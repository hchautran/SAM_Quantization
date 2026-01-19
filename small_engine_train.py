import os
import logging
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from functools import partial
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import torch.optim as optim
import argparse
import random
import itertools 
import time

from train.utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from train.utils.loss_mask import loss_masks
from data_utils import OnlineDataset
from train.train import compute_iou, compute_boundary_iou, MaskDecoderHQ
from segment_anything import SamPredictor,sam_model_registry
import train.utils.misc as misc

from utils.utils import show_mask_image
from prunning_rate.samprune import image_encoder_monkey_patch_train
from utils.quant_utils import (
    quantize_activation_per_token_absmax,
)
from processors import (
    get_encoder_processor,
    EncoderRecenterAttentionProcessor,
    EncoderAttentionProcessor,
    DecoderDoNothingProcessor,
)
# from profiler import InferenceProfiler, compare_inference_speed
from segment_anything.modeling.image_encoder import Attention as EncoderSamAttention
from segment_anything.modeling.transformer import  Attention as  DecoderAttention
from seginw.segment_anything.modeling.image_encoder import Attention as EncoderAttention 
from train.segment_anything_training.modeling.image_encoder import Attention as EncoderAttentionTraining
from sam_engine import override_args, Evaluator , create_calib_dataloaders, setup_logger, get_default_datasets, plot_output
from prunning_rate.samprune import DiffPruneRateAttention
import wandb

def print_model_structure(model, title="Model Structure"):
    print(f"\n{title}")
    print("=" * len(title))
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")
    print("=" * len(title))

def evaluate(args, sam, valid_dataloaders, visualize=False):
    # Handle DDP wrapper - access the actual model
    if hasattr(sam, 'module'):
        actual_sam = sam.module
    else:
        actual_sam = sam
        
    actual_sam.eval()
    print("Validating...")
    test_stats = {}

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader))

        for data_val in metric_logger.log_every(valid_dataloader, 5):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label']

            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.cuda()
                labels_ori = labels_ori.cuda()

            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
            
            labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])
            input_keys = ['box']
            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=actual_sam.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            with torch.no_grad():
                # Use the DDP-wrapped model for consistent behavior
                batched_output = sam(batched_input, multimask_output=False)
           
            # Extract masks from SAM output
            masks_hq ,_= batched_output
            
            # Handle mask dimensions
            if masks_hq.dim() == 5:  # [batch, 1, 1, H, W]
                masks_hq = masks_hq.squeeze(1).squeeze(1)  # [batch, H, W]
            elif masks_hq.dim() == 4:  # [batch, 1, H, W]
                masks_hq = masks_hq.squeeze(1)  # [batch, H, W]
            if masks_hq.dtype == torch.bool:
                masks_hq = masks_hq.float()
            iou = compute_iou(masks_hq.unsqueeze(1), labels_ori)
            boundary_iou = compute_boundary_iou(masks_hq.unsqueeze(1), labels_ori)

            if visualize:
                print("visualize")
                os.makedirs(args.output, exist_ok=True)
                masks_hq_vis = (F.interpolate(masks_hq.detach().unsqueeze(1), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
                for ii in range(len(imgs)):
                    base = data_val['imidx'][ii].item()
                    print('base:', base)
                    save_base = os.path.join(args.output, str(k)+'_'+ str(base))
                    imgs_ii = imgs[ii].astype(dtype=np.uint8)
                    show_iou = torch.tensor([iou.item()])
                    show_boundary_iou = torch.tensor([boundary_iou.item()])
                    show_anns(masks_hq_vis[ii], None, labels_box[ii].cpu(), None, save_base , imgs_ii, show_iou, show_boundary_iou)

            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)

        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)

    return test_stats

def train(args, sam_hq, target_flop ,optimizer, train_dataloaders, valid_dataloaders, lr_scheduler, ratio=0.1):
    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num
    train_num = len(train_dataloaders)

    learning_rate = args.learning_rate
    lr_drop = args.lr_drop_epoch
    
    sam_hq.train()
    _ = sam_hq.to(device="cuda")
    sam_hq = torch.nn.parallel.DistributedDataParallel(sam_hq, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    
    training_start_time = time.time()
    for epoch in range(epoch_start, epoch_num): 
        epoch_start_time = time.time()
        print("epoch:   ", epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)
        
        # Start timing for this epoch
        epoch_start_time = time.time()

        for data in metric_logger.log_every(train_dataloaders, 1000):
            inputs, labels = data['image'], data['label']
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()
            
            # input prompt
            input_keys = ['box', 'point', 'noise_mask']
            labels_box = misc.masks_to_boxes(labels[:, 0, :, :])
            try:
                labels_points = misc.masks_sample_points(labels[:, 0, :, :])
            except:
                # less than 10 points
                input_keys = ['box', 'noise_mask']
            labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256)

            batched_input = []
            for b_i in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam_hq.device).permute(2, 0, 1).contiguous()
                dict_input['image'] = input_image 
                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i+1]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None, :]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = imgs[b_i].shape[:2]
                batched_input.append(dict_input)

            # Forward pass through SAM-HQ model
            masks_hq, total_encoder_flops = sam_hq(batched_input, multimask_output=False)
            
            # Remove batch dimension if present (masks are already batched)
            if masks_hq.dim() == 5:  # [batch, 1, 1, H, W]
                masks_hq = masks_hq.squeeze(1).squeeze(1)  # [batch, H, W]
            elif masks_hq.dim() == 4:  # [batch, 1, H, W]
                masks_hq = masks_hq.squeeze(1)  # [batch, H, W]
            if masks_hq.dtype == torch.bool:
                masks_hq = masks_hq.float()
            masks_hq = masks_hq.unsqueeze(1)  # [batch, 1, H, W]
            
            

            loss_flops = (total_encoder_flops/1e11 - target_flop)**2
            
            loss_flops = loss_flops * ratio

            loss_mask, loss_dice = loss_masks(masks_hq, labels/255.0, len(masks_hq))
            loss = loss_mask + loss_dice + loss_flops
            
            wandb.log({
                "train_step/loss": loss.item(),
                "train_step/loss_mask": loss_mask.item(), 
                "train_step/loss_dice": loss_dice.item(),
                "train_step/loss_flops": loss_flops.item(),
                "train_step/total_encoder_flops": total_encoder_flops.item(),
                "train_step/epoch": epoch,
            })

            loss_dict = {"loss_mask": loss_mask, "loss_dice": loss_dice, "loss_flops": loss_flops}

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)

        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        lr_scheduler.step()
        test_stats = evaluate(args, sam_hq, valid_dataloaders)
        train_stats.update(test_stats)
        
        epoch_time = time.time() - epoch_start_time
        total_training_time = time.time() - training_start_time
        
        wandb_log_dict = {"epoch": epoch}
        wandb_log_dict.update({f"epoch/{k}": v for k, v in train_stats.items()})
        wandb_log_dict["epoch/time_seconds"] = epoch_time
        wandb_log_dict["epoch/total_training_time_seconds"] = total_training_time
        wandb.log(wandb_log_dict)
        sam_hq.train()  

        if epoch % args.model_save_fre == 0:
            model_name = "/sam_hq_epoch_"+str(epoch)+args_yaml.model.model_type+"target_flopsize_"+str(target_flop)+"_ratio_"+str(ratio)+"-lr"+str(learning_rate)+ "-lr_drop" + str(lr_drop) +".pth"
            print('come here save at', args.output + model_name)
            misc.save_on_master(sam_hq.module.state_dict(), args.output + model_name)
    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")
def analyze_model_head_pruning_and_flops(predictor_model, manual_local_heads=None, manual_global_heads=None):
    """
    Analyze head pruning ratios and calculate FLOPs for attention operations.
    
    Args:
        predictor_model: The SAM model with DiffPruneRateAttention modules
        manual_local_heads: Number of local heads to keep manually (optional)
        manual_global_heads: Number of global heads to keep manually (optional)
        
    Returns:
        dict: Dictionary containing head statistics and FLOPs information
    """
    # Initialize counters
    total_kept_heads_local = 0
    total_original_heads_local = 0
    total_kept_heads_global = 0
    total_original_heads_global = 0
    
    # For FLOPs calculation
    total_flops = 0
    total_baseline_flops = 0  # FLOPs without pruning
    
    nu_layer_global = 0
    nu_layer_local = 0
    
    # Collect per-layer information
    layer_info = []
    
    # Iterate through all modules to find DiffPruneRateAttention modules
    for name, module in predictor_model.named_modules():
        if isinstance(module, DiffPruneRateAttention):
            # Get head information
            kept_heads = int(module.prune_ddp.update_kept_head_number())
            original_heads = module.prune_ddp.head_number
            
            # Determine if this is a local or global attention
            is_local = original_heads % 100 == 0
            
            # For FLOPs calculation, we assume fixed dimensions
            # B//nu_images = 25 for local, 1 for global
            batch_factor = 25 if is_local else 1
            if is_local: 
                nu_layer_local += 1
            else: 
                nu_layer_global += 1
            # Using typical attention map dimensions for SAM
            H, W = (14, 14) if is_local else (64, 64)
            
            # Calculate FLOPs for this attention module
            qkv_flops = module._calculate_qkv_flops(batch_factor, H, W)
            proj_flops = module._calculate_projection_flops(batch_factor, H, W)
            attention_flops = module._calculate_attention_flops(H, W, kept_heads)
            baseline_attention_flops = module._calculate_attention_flops(H, W, original_heads)
            
            module_flops = qkv_flops + proj_flops + attention_flops
            module_baseline_flops = qkv_flops + proj_flops + baseline_attention_flops
            
            # Update totals
            total_flops += module_flops
            total_baseline_flops += module_baseline_flops
            
            # Store layer information
            layer_info.append({
                'name': name,
                'kept_heads': kept_heads,
                'original_heads': original_heads,
                'is_local': is_local,
                'flops': module_flops,
                'baseline_flops': module_baseline_flops
            })
            
            # Update head counters
            if is_local:
                total_kept_heads_local += kept_heads
                total_original_heads_local += original_heads
            else:
                total_kept_heads_global += kept_heads
                total_original_heads_global += original_heads
    
    # Calculate overall ratios
    overall_local_ratio = total_kept_heads_local / total_original_heads_local if total_original_heads_local > 0 else 0
    overall_global_ratio = total_kept_heads_global / total_original_heads_global if total_original_heads_global > 0 else 0
    flops_reduction = (1 - total_flops / total_baseline_flops) * 100 if total_baseline_flops > 0 else 0
    
    # Calculate manual heads if not provided
    avg_local_heads = total_kept_heads_local / nu_layer_local if nu_layer_local > 0 else 0
    avg_global_heads = total_kept_heads_global / nu_layer_global if nu_layer_global > 0 else 0
    
    # Round up to nearest integer (ceiling)
    if manual_local_heads is None:
        manual_local_heads = int(avg_local_heads + 0.999)  # Equivalent to math.ceil()
    if manual_global_heads is None:
        manual_global_heads = int(avg_global_heads + 0.999)  # Equivalent to math.ceil()
    
    # Recalculate manual FLOPs
    total_manual_flops = 0
    layer_idx_local = 0
    layer_idx_global = 0
    
    for name, module in predictor_model.named_modules():
        if isinstance(module, DiffPruneRateAttention):
            original_heads = module.prune_ddp.head_number
            is_local = original_heads % 100 == 0
            
            batch_factor = 25 if is_local else 1
            H, W = (14, 14) if is_local else (64, 64)
            
            # Calculate FLOPs with manual head counts
            qkv_flops = module._calculate_qkv_flops(batch_factor, H, W)
            proj_flops = module._calculate_projection_flops(batch_factor, H, W)
            
            # Use manual head counts
            if is_local:
                manual_attention_flops = module._calculate_attention_flops(H, W, manual_local_heads)
                layer_idx_local += 1
            else:
                manual_attention_flops = module._calculate_attention_flops(H, W, manual_global_heads)
                layer_idx_global += 1
            
            manual_module_flops = qkv_flops + proj_flops + manual_attention_flops
            total_manual_flops += manual_module_flops
    
    # Calculate manual FLOPs reduction
    manual_flops_reduction = None
    if total_baseline_flops > 0:
        manual_flops_reduction = (1 - total_manual_flops / total_baseline_flops) * 100
    
    return {
        'layer_info': layer_info,
        'head_stats': {
            'local_kept': total_kept_heads_local,
            'local_total': total_original_heads_local,
            'local_ratio': overall_local_ratio,
            'global_kept': total_kept_heads_global,
            'global_total': total_original_heads_global,
            'global_ratio': overall_global_ratio
        },
        'flops_stats': {
            'total_flops': total_flops,
            'baseline_flops': total_baseline_flops,
            'reduction_percent': flops_reduction,
            'manual_flops': total_manual_flops,
            'manual_reduction_percent': manual_flops_reduction
        },
        'manual_settings': {
            'manual_local_heads': manual_local_heads,
            'manual_global_heads': manual_global_heads,
            'avg_local_heads': avg_local_heads,
            'avg_global_heads': avg_global_heads
        }
    }
def print_head_pruning_and_flops_info(predictor_model):
    """
    Print detailed head pruning ratios and FLOPs information for the model.
    
    Args:
        predictor_model: The SAM model with DiffPruneRateAttention modules
    """
    print("\n=== Head Pruning Ratios ===")
    
    # Analyze model
    analysis = analyze_model_head_pruning_and_flops(predictor_model)
    
    # Print per-layer information
    for layer in analysis['layer_info']:
        layer_type = "Local" if layer['is_local'] else "Global"
        print(f"Layer {layer['name']}: {layer['kept_heads']}/{layer['original_heads']} ({layer_type})")
    
    # Print overall head statistics
    head_stats = analysis['head_stats']
    if head_stats['local_total'] > 0:
        print(f"\nOverall Local: {head_stats['local_kept']}/{head_stats['local_total']} heads kept "
              f"({head_stats['local_ratio']:.2%}) pruning rate: {1-head_stats['local_ratio']:.2%}")
    
    if head_stats['global_total'] > 0:
        print(f"\nOverall Global: {head_stats['global_kept']}/{head_stats['global_total']} heads kept "
              f"({head_stats['global_ratio']:.2%}) pruning rate: {1-head_stats['global_ratio']:.2%}")
    
    # Print manual head information
    manual_settings = analysis['manual_settings']
    print(f"\nManual Settings:")
    print(f"Average Local Heads: {manual_settings['avg_local_heads']:.2f}")
    print(f"Average Global Heads: {manual_settings['avg_global_heads']:.2f}")
    print(f"Manual Local Heads (rounded): {manual_settings['manual_local_heads']}")
    print(f"Manual Global Heads (rounded): {manual_settings['manual_global_heads']}")
    
    # Print FLOPs information
    flops_stats = analysis['flops_stats']
    print("\n=== FLOPs Information (Attention Only) ===")
    print(f"Total Attention FLOPs (with parameter pruning): {flops_stats['total_flops']/1e9:.2f} GFLOPs")
    print(f"Total Attention FLOPs (baseline): {flops_stats['baseline_flops']/1e9:.2f} GFLOPs")
    if flops_stats['reduction_percent'] >= 0:
        print(f"Attention FLOPs Reduction (parameter pruning): {flops_stats['reduction_percent']:.2f}%")
    
    # Print manual pruning FLOPs information if provided
    if flops_stats['manual_flops'] is not None:
        print(f"\nTotal Attention FLOPs (with manual pruning): {flops_stats['manual_flops']/1e9:.2f} GFLOPs")
        if flops_stats['manual_reduction_percent'] >= 0:
            print(f"Attention FLOPs Reduction (manual pruning): {flops_stats['manual_reduction_percent']:.2f}%")
        
        # Compare the two pruning approaches
        if flops_stats['total_flops'] and flops_stats['manual_flops']:
            flops_diff = flops_stats['manual_flops'] - flops_stats['total_flops']
            if flops_diff > 0:
                print(f"Manual pruning uses {flops_diff/1e9:.2f} GFLOPs more than parameter pruning")
            else:
                print(f"Manual pruning uses {abs(flops_diff)/1e9:.2f} GFLOPs less than parameter pruning")
    
    print("===========================\n")
class training_engine:
    """Main engine class for orchestrating quantization experiments"""

    def __init__(self, strategy_name: str, mode_train: bool,args, datasets=None) -> None:
        
        # if misc.is_main_process():
        #     print("chiiii")
        
        self.stat = {}
        self.strategy_name = strategy_name
        self.train = mode_train
        self.args= args
        # Setup datasets
        if datasets is None:
            datasets = get_default_datasets()
            
        
        if self.train:
            
            valid_im_gt_list = get_im_gt_name_dict([datasets[0]], flag="valid")
            for dataset_dict in valid_im_gt_list:
                dataset_dict["im_path"] = dataset_dict["im_path"][-10:]
                dataset_dict["gt_path"] = dataset_dict["gt_path"][-10:]
            self.valid_dataloaders, self.vals_datasets = create_dataloaders(
                valid_im_gt_list,
                my_transforms=[Resize([1024, 1024])],
                batch_size=self.args.train_prune_rate.batch_size_valid,
                training= False
            )
            
            train_im_gt_list = get_im_gt_name_dict([datasets[0]], flag="train")
            for dataset_dict in train_im_gt_list:
                dataset_dict["im_path"] = dataset_dict["im_path"][:500]
                dataset_dict["gt_path"] = dataset_dict["gt_path"][:500]
            self.train_dataloaders, self.train_datasets = create_dataloaders(
                train_im_gt_list,
                my_transforms=[RandomHFlip(),
                               LargeScaleJitter()],
                batch_size=self.args.train_prune_rate.batch_size_train,
                training= True
            )
        else:
            valid_im_gt_list = get_im_gt_name_dict([datasets[1]], flag="valid")
            self.dataloaders, self.datasets = create_calib_dataloaders(
                valid_im_gt_list,
                my_transforms=[Resize([1024, 1024])],
                batch_size=1,
            )
       
       
        
    def monkey_patch(self, predictor, encoder_config=None,args_yaml= None ,train=False):
        print("Applying encoder quantization...")
        image_encoder_monkey_patch_train(
            predictor.model,
            processor=encoder_config.get('processor'),
            args_yaml= args_yaml,
            train = train,
        )
    def setup_and_calibrate_processors(self, predictor, num_calib_samples=32, encoder_processor=EncoderAttentionProcessor,args_yaml=None):
        """
        Setup and calibrate processors for encoder and/or decoder.

        Args:
            predictor: SamPredictor instance
            num_calib_samples: Number of samples for calibration

        Returns:
            Tuple of (encoder_processor, decoder_processor)
        """      
        print("Setting up encoder processor...")
        # encoder_processor = EncoderAttentionProcessor()
        encoder_processor.set_params(args_yaml)
        encoder_processor.calibrate(
            predictor=predictor,
            modules=( EncoderAttentionTraining, EncoderAttention, EncoderSamAttention),
            num_samples=num_calib_samples
        )

        print(f"Encoder processor calibrated on {num_calib_samples} samples")

        return encoder_processor
    def eval_hq44k(self, predictor: SamPredictor, num_samples=None, plot_figures=False):
        """Delegate to evaluator component"""
        checkpoint_path = "/home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/pretrained_checkpoint/prune_rate/sam_hq_epoch_40_vit_b_target_flopsize_1.5_ratio_5startlr0.1_lr_drop_10.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        predictor.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        
        
        print_head_pruning_and_flops_info(predictor.model) # second attribute is kept local head number , third attribute is kept global head number- to compare FLops
        
        sam = predictor.model
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        sam = sam.to(self.device)
        sam.eval()  
        self.evaluator = Evaluator(self.accelerator, self.dataloaders, self.datasets)
        
        return self.evaluator.eval_hq44k(predictor, num_samples, plot_figures)
    def train_model(self, predictor, args_yaml):
        sam = predictor.model
        
        print("--- define optimizer ---")
        # Collect ONLY selected_probability parameters for the optimizer
        trainable_params = []
        for name, param in sam.named_parameters():
            if 'selected_probability' in name:
                trainable_params.append(param)
                print(f"Training parameter: {name}")
            # DON'T set requires_grad=False for other parameters!
        
        if not trainable_params:
            raise ValueError("No selected_probability parameters found!")
        
        # Create optimizer with ONLY selected_probability parameters
        # This ensures only these parameters get updated, even though others have gradients
        optimizer = optim.Adam(trainable_params, 
                            lr=args_yaml.train_prune_rate.learning_rate, 
                            betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args_yaml.train_prune_rate.lr_drop_epoch)
        lr_scheduler.last_epoch = args_yaml.train_prune_rate.start_epoch
        if args_yaml.model.model_type == "vit_b":
            target_flop = 1.5
            ratio= 5
        elif args_yaml.model.model_type == "vit_l":
            target_flop = 15.8
            ratio= 0.05
        elif args_yaml.model.model_type == "vit_h":
            target_flop = 32
            ratio= 0.025 
        wandb.init(project="sam-hq-training", name=f"experiment_{self.strategy_name}-model_{args_yaml.model.model_type}-targetflop_{target_flop}-ratio_{ratio}-lr-{args_yaml.train_prune_rate.learning_rate}-lr_drop_{args_yaml.train_prune_rate.lr_drop_epoch}")
        train(args_yaml.train_prune_rate, sam, target_flop, optimizer, self.train_dataloaders, self.valid_dataloaders, lr_scheduler, ratio)
    def train_model_minimize_entropy_scores(self, predictor, args_yaml):
        sam = predictor.model
        
        print("--- define optimizer ---")
        # Collect ONLY selected_probability parameters for the optimizer
        trainable_params = []
        for name, param in sam.named_parameters():
            if 'selected_probability' in name:
                trainable_params.append(param)
                print(f"Training parameter: {name}")
            # DON'T set requires_grad=False for other parameters!
        
        if not trainable_params:
            raise ValueError("No selected_probability parameters found!")
        
        # Create optimizer with ONLY selected_probability parameters
        # This ensures only these parameters get updated, even though others have gradients
        optimizer = optim.Adam(trainable_params, 
                            lr=args_yaml.train_prune_rate.learning_rate, 
                            betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args_yaml.train_prune_rate.lr_drop_epoch)
        lr_scheduler.last_epoch = args_yaml.train_prune_rate.start_epoch
        
        wandb.init(project="sam-hq-training", name=f"experiment_{self.strategy_name}-model_{args_yaml.model.model_type}-lr-{args_yaml.train_prune_rate.learning_rate}-lr_drop_{args_yaml.train_prune_rate.lr_drop_epoch}")
        train(args_yaml.train_prune_rate, sam, target_flop, optimizer, self.train_dataloaders, self.valid_dataloaders, lr_scheduler, ratio)
if __name__ == '__main__':
    
    

    parser = argparse.ArgumentParser(description='SAM Quantization Engine')
    parser.add_argument('--encoder_processor', default='base',
                        help='Enable encoder quantization')
    parser.add_argument('--num-calib-samples', type=int, default=16,
                        help='Number of calibration samples')
    parser.add_argument('--num-samples', type=int, default=400,
                        help='Number of evaluation samples')
    parser.add_argument('--target', type=str, default='decoder',
                        choices=['decoder', 'encoder', 'both'],
                        help='Target for k_preserve experiments')
    parser.add_argument('--processor', type=str, default='PRUNE_RATE',
                       choices=['BASE','PRUNE_RATE'],
                       help='Processor to use')
    parser.add_argument("--config-file", type=str, default=None,),
    parser.add_argument('--train', default=False ,action='store_true')
    
    args = parser.parse_args()
    args_yaml = OmegaConf.load(args.config_file)
    args_yaml = override_args(args, args_yaml)

    if args.train:
        misc.init_distributed_mode(args_yaml.train_prune_rate)
        print('world size: {}'.format(args_yaml.train_prune_rate.world_size))
        print('rank: {}'.format(args_yaml.train_prune_rate.rank))
        print('local_rank: {}'.format(args_yaml.train_prune_rate.local_rank))

        seed = args_yaml.train_prune_rate.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    model_type = args_yaml.model.model_type
    checkpoint_path = args_yaml.model.hq_checkpoint
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')
    # import ipdb; ipdb.set_trace()
    predictor = SamPredictor(sam)

    # Initialize engine
    engine = training_engine('hq44k',args.train, args_yaml)
    print(args.processor)
    enc_processor = get_encoder_processor(args.processor)
    encoder_processor = engine.setup_and_calibrate_processors(
        predictor,
        num_calib_samples=args.num_calib_samples,
        encoder_processor=enc_processor,
        args_yaml= args_yaml,
    )
    # Apply quantization
    encoder_config = {
        'processor': encoder_processor,
    } if encoder_processor else None
    engine.monkey_patch(predictor, encoder_config,args_yaml, args.train)
    # print_model_structure(predictor.model,"Final structure ")
    # exit()
    if args.train:
        engine.train_model(predictor=predictor, args_yaml=args_yaml)
    else:
        results = engine.eval_hq44k(predictor=predictor, num_samples=args.num_samples, plot_figures=False)
    # 