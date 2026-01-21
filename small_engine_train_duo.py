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
from train.utils.loss_mask import loss_masks, get_uncertain_point_coords_with_randomness, calculate_uncertainty, point_sample
from data_utils import OnlineDataset
from train.train import compute_iou, compute_boundary_iou, MaskDecoderHQ
from segment_anything import SamPredictor,sam_model_registry
import train.utils.misc as misc

from utils.utils import show_mask_image
from prunning_rate.sampruneduo import image_encoder_monkey_patch_train_duo
from prunning_rate.samprunediff_duo import image_encoder_monkey_patch_train_duo_diff
from utils.quant_utils import (
    quantize_activation_per_token_absmax,
)

# from profiler import InferenceProfiler, compare_inference_speed
from segment_anything.modeling.image_encoder import Attention as EncoderSamAttention
from segment_anything.modeling.transformer import  Attention as  DecoderAttention
from seginw.segment_anything.modeling.image_encoder import Attention as EncoderAttention 
from train.segment_anything_training.modeling.image_encoder import Attention as EncoderAttentionTraining
from sam_engine import override_args, Evaluator , create_calib_dataloaders, setup_logger, get_default_datasets, plot_output
import wandb

from processors import (
    get_encoder_processor,
    EncoderRecenterAttentionProcessor,
    EncoderAttentionProcessor,
    DecoderDoNothingProcessor,
)


def get_full_attention_heads(model):
    full_attention_heads = []
    # import ipdb; ipdb.set_trace()
    for layer in model.module.image_encoder.blocks:
        module = layer.attn
        if not hasattr(module, "full_attention_heads"):
            continue
        full_attention_heads.append(module.full_attention_heads)
    return full_attention_heads

def get_head_probability(model):
    full_head_probability=[]
    for layer in model.module.image_encoder.blocks:
        module = layer.attn
        if not hasattr(module, "prune_ddp"):
            continue
        head_probability = module.prune_ddp.get_head_probability_diff_duo()
        full_head_probability.append(head_probability)
    return full_head_probability
def print_model_structure(model, title="Model Structure"):
    print(f"\n{title}")
    print("=" * len(title))
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")
    print("=" * len(title))

def l1_loss(x):
    numel = x.numel()
    l1 = x.abs().sum()
    return l1 / numel

def distill_loss_masks_bce(
    masks_hq_teacher,
    masks_hq_prune,
    oversample_ratio=3.0,
    num_points=112 * 112,
    importance_sample_ratio=0.75,
    T=2.0,
):
    """
    Distillation loss for mask logits using uncertainty-based point sampling
    and soft BCE (teacher -> student).

    Args:
        masks_hq_teacher: Tensor [B, 1, H, W], raw logits (teacher)
        masks_hq_prune:   Tensor [B, 1, H, W], raw logits (student)
        oversample_ratio: same as normal mask loss
        num_points:       number of sampled points
        importance_sample_ratio: uncertainty sampling ratio
        T: temperature for distillation

    Returns:
        Scalar distillation loss
    """

    # 1. Sample uncertain points from teacher
    with torch.no_grad():
        point_coords = get_uncertain_point_coords_with_randomness(
            masks_hq_teacher,
            lambda logits: calculate_uncertainty(logits),
            num_points,
            oversample_ratio,
            importance_sample_ratio,
        )

        # teacher logits at sampled points
        teacher_point_logits = point_sample(
            masks_hq_teacher,
            point_coords,
            align_corners=False,
        ).squeeze(1)  # [B, P]

        # soft teacher targets
        teacher_prob = torch.sigmoid(teacher_point_logits / T)

    # 2. Student logits at the same points
    student_point_logits = point_sample(
        masks_hq_prune,
        point_coords,
        align_corners=False,
    ).squeeze(1)  # [B, P]

    # 3. Soft BCE distillation loss
    distill_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        student_point_logits / T,
        teacher_prob,
        reduction="mean",
    ) * (T * T)

    return distill_loss


def print_pruned_heads_info(model, threshold, global_threshold, logger, model_type="vit_b"):
    """
    Print information about pruned heads and compute average pruned heads
    for local-threshold layers vs global-threshold layers.

    Supports two cases:
    (A) Existing: module.full_attention_heads (values in [0,1])  -> pruned if < threshold
    (B) New Duo: module.prune_ddp.get_head_probability_diff_duo() -> kept if > threshold
             so pruned if <= threshold (or < threshold, choose one consistently)
    """

    actual_model = model.module if hasattr(model, "module") else model

    total_pruned_heads = 0
    total_heads = 0

    local_pruned_heads = 0
    global_pruned_heads = 0
    local_layers = 0
    global_layers = 0

    local_prune_percent_sum = 0.0
    global_prune_percent_sum = 0.0

    print(f"\n{'='*60}")
    print(f"Pruned Heads Information")
    print(f"Local thr: {threshold} | Global thr: {global_threshold}")
    print(f"Model type: {model_type}")
    print(f"{'='*60}\n")

    for name, module in actual_model.named_modules():

        # --- NEW: decide if this module is eligible by either mechanism ---
        has_full = hasattr(module, "full_attention_heads")
        has_duo  = hasattr(module, "prune_ddp") and hasattr(module.prune_ddp, "get_head_probability_diff_duo")

        if not (has_full or has_duo):
            continue

        # Decide threshold type (same as before)
        if model_type == "vit_b":
            use_global = any(tok in name for tok in [".2", ".5", "8", "11"])
        elif model_type == "vit_l":
            use_global = any(tok in name for tok in [".5", "11", "17", "23"])
        elif model_type == "vit_h":
            use_global = any(tok in name for tok in [".7", "15", "23", "31"])
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        thr_used = global_threshold if use_global else threshold
        tag = "GLOBAL" if use_global else "LOCAL"

        # =========================
        # (A) OLD PATH: full_attention_heads
        # =========================
        if has_full and not has_duo:
            # head_weights = module.full_attention_heads.clamp(0, 1)
            head_weights = module.full_attention_heads
            num_total = int(head_weights.numel())

            # old meaning: pruned if head weight is below threshold
            pruned_mask = head_weights < thr_used

        # =========================
        # (B) NEW PATH: Duo probability diff
        # =========================
        else:
            # new meaning: kept if prob_diff > threshold -> pruned otherwise
            single_mask_probability = module.prune_ddp.get_head_probability_diff_duo()

            # make sure it's a tensor on CPU-friendly dtype
            if not torch.is_tensor(single_mask_probability):
                single_mask_probability = torch.tensor(single_mask_probability)

            single_mask_probability = single_mask_probability.detach()
            # logger.info(f"single_mask_probability: {single_mask_probability} ")
            num_total = int(single_mask_probability.numel())

            # kept mask is > thr, so pruned is NOT kept
            kept_mask = single_mask_probability > thr_used
            pruned_mask = ~kept_mask

        # layer counters
        if use_global:
            global_layers += 1
        else:
            local_layers += 1

        # compute stats
        # NOTE: pruned_mask is expected to be 1D; if not, flatten safely
        pruned_mask_flat = pruned_mask.reshape(-1)

        num_pruned = int(pruned_mask_flat.sum().item())
        prune_percent = num_pruned / num_total if num_total > 0 else 0.0

        # indices (optional)
        pruned_indices = torch.where(pruned_mask_flat)[0].detach().cpu().numpy()

        # Print per-module info
        print(f"Module: {name}  [{tag} thr={thr_used}]")
        print(f"  Total heads: {num_total}")
        print(f"  Pruned heads: {num_pruned} ({prune_percent*100:.1f}%)")
        print()

        logger.info(f"Module: {name}  [{tag} thr={thr_used}]")
        logger.info(f"  Total heads: {num_total}")
        logger.info(f"  Pruned heads: {num_pruned} ({prune_percent*100:.1f}%)")

        # Accumulate totals
        total_pruned_heads += num_pruned
        total_heads += num_total

        if use_global:
            global_pruned_heads += num_pruned
            global_prune_percent_sum += prune_percent
        else:
            local_pruned_heads += num_pruned
            local_prune_percent_sum += prune_percent

    # Compute averages
    avg_local_heads = local_pruned_heads / local_layers if local_layers > 0 else 0.0
    avg_global_heads = global_pruned_heads / global_layers if global_layers > 0 else 0.0

    avg_local_percent = local_prune_percent_sum / local_layers if local_layers > 0 else 0.0
    avg_global_percent = global_prune_percent_sum / global_layers if global_layers > 0 else 0.0

    overall_prune_percent = total_pruned_heads / total_heads if total_heads > 0 else 0.0

    # Print summary
    print(f"{'='*60}")
    print("Overall Summary")
    print(f"{'='*60}")
    print(f"Total heads: {total_heads}")
    print(f"Total pruned heads: {total_pruned_heads}")
    print(f"Overall pruning rate: {overall_prune_percent*100:.2f}%")

    print(f"\n{'='*60}")
    print("Local vs Global Pruning Statistics")
    print(f"{'='*60}")

    print("Local pruning:")
    print(f"  Layers: {local_layers}")
    print(f"  Total pruned heads: {local_pruned_heads}")
    print(f"  Avg pruned heads / layer: {avg_local_heads:.2f}")
    print(f"  Avg prune percentage / layer: {avg_local_percent*100:.2f}%")

    print("\nGlobal pruning:")
    print(f"  Layers: {global_layers}")
    print(f"  Total pruned heads: {global_pruned_heads}")
    print(f"  Avg pruned heads / layer: {avg_global_heads:.2f}")
    print(f"  Avg prune percentage / layer: {avg_global_percent*100:.2f}%")

    print(f"{'='*60}\n")

    logger.info(f"{'='*60}")
    logger.info("Overall Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total heads: {total_heads}")
    logger.info(f"Total pruned heads: {total_pruned_heads}")
    logger.info(f"Overall pruning rate: {overall_prune_percent*100:.2f}%")

    logger.info(f"\n{'='*60}")
    logger.info("Local vs Global Pruning Statistics")
    logger.info(f"{'='*60}")

    logger.info("Local pruning:")
    logger.info(f"  Layers: {local_layers}")
    logger.info(f"  Total pruned heads: {local_pruned_heads}")
    logger.info(f"  Avg pruned heads / layer: {avg_local_heads:.2f}")
    logger.info(f"  Avg prune percentage / layer: {avg_local_percent*100:.2f}%")

    logger.info("\nGlobal pruning:")
    logger.info(f"  Layers: {global_layers}")
    logger.info(f"  Total pruned heads: {global_pruned_heads}")
    logger.info(f"  Avg pruned heads / layer: {avg_global_heads:.2f}")
    logger.info(f"  Avg prune percentage / layer: {avg_global_percent*100:.2f}%")

    logger.info(f"{'='*60}\n")

    stats = {
        "total_heads": total_heads,
        "total_pruned_heads": total_pruned_heads,
        "overall_pruning_rate": overall_prune_percent,

        "local_layers": local_layers,
        "local_pruned_heads": local_pruned_heads,
        "avg_pruned_heads_per_local_layer": avg_local_heads,
        "avg_prune_percent_per_local_layer": avg_local_percent,

        "global_layers": global_layers,
        "global_pruned_heads": global_pruned_heads,
        "avg_pruned_heads_per_global_layer": avg_global_heads,
        "avg_prune_percent_per_global_layer": avg_global_percent,
    }

    return total_pruned_heads, total_heads



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
            masks_hq = batched_output
            
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

def train(args, sam_hq, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
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
            # teacher_image_embeddings, teacher_interm_embeddings, prune_image_embeddings, prune_interm_embeddings = sam_hq(batched_input, multimask_output=False)
            # image_embeddings_distill_loss
            # import ipdb ; ipdb.set_trace()

            masks_hq = sam_hq(batched_input, multimask_output=False)
            
            # Remove batch dimension if present (masks are already batched)
            if masks_hq.dim() == 5:  # [batch, 1, 1, H, W]
                masks_hq = masks_hq.squeeze(1).squeeze(1)  # [batch, H, W]
            elif masks_hq.dim() == 4:  # [batch, 1, H, W]
                masks_hq = masks_hq.squeeze(1)  # [batch, H, W]
            if masks_hq.dtype == torch.bool:
                masks_hq = masks_hq.float()
            masks_hq = masks_hq.unsqueeze(1)  # [batch, 1, H, W]
            
            if args.training_method == "duo":
                full_attention_heads = get_full_attention_heads(sam_hq)

                full_attention_heads = [
                    h.to(sam_hq.device)
                    for h in full_attention_heads
                ]

                reg_loss = l1_loss(torch.cat(full_attention_heads).float())

            elif args.training_method == "diffduo" :
                head_probability = get_head_probability(sam_hq)
                head_probability = [
                    h.to(sam_hq.device)
                    for h in head_probability
                ]
                reg_loss = l1_loss(torch.cat(head_probability).float())
          

            ############################################################
            ### Calculate loss of pruning output from the lable

            # loss_mask, loss_dice = loss_masks(masks_hq, labels/255.0, len(masks_hq))
            # loss = loss_mask + loss_dice +  args.reg_weight * reg_loss


            # wandb.log({
            #     "train_step/loss": loss.item(),
            #     "train_step/loss_mask": loss_mask.item(), 
            #     "train_step/loss_dice": loss_dice.item(),
            #     "train_step/reg_loss": reg_loss.item(),
            #     "train_step/epoch": epoch,
            # })

            # loss_dict = {"loss_mask": loss_mask, "loss_dice": loss_dice}

            #############################################################
            ### Calculate loss of Distillation loss 

            nu_image =masks_hq.shape[0]//2
            masks_hq_teacher = masks_hq[:nu_image]
            masks_hq_prune = masks_hq[nu_image:]

            loss_distill = distill_loss_masks_bce(
                masks_hq_teacher,
                masks_hq_prune,
                T=2.0,
            )
            loss = loss_distill  +  args.reg_weight * reg_loss
            wandb.log({
                "train_step/loss": loss.item(),
                "train_step/loss_distill": loss_distill.item(), 
                "train_step/reg_loss": reg_loss.item(),
                "train_step/epoch": epoch,
            })

            loss_dict = {"loss_distill": loss_distill}

            #############################################################
            
            

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

        # test_stats = evaluate(args, sam_hq, valid_dataloaders)
        # train_stats.update(test_stats)
        
        epoch_time = time.time() - epoch_start_time
        total_training_time = time.time() - training_start_time
        
        wandb_log_dict = {"epoch": epoch}
        wandb_log_dict.update({f"epoch/{k}": v for k, v in train_stats.items()})
        wandb_log_dict["epoch/time_seconds"] = epoch_time
        wandb_log_dict["epoch/total_training_time_seconds"] = total_training_time
        wandb.log(wandb_log_dict)
        sam_hq.train()  

        if epoch % args.model_save_fre == 0 and epoch != 0:
            if args.training_method == "diffduo" :
                model_name = "/diffduo_sam_hq_epoch_torchnograd_distill" + str(epoch) + "_" + str(args_yaml.model.model_type) + "_reg-weight_" + str(args.reg_weight) + "_lr" + str(learning_rate) + "_lr_drop" + str(lr_drop) + ".pth"
            elif  args.training_method == "duo" :
                model_name = "/duo_sam_hq_epoch_torchnograd_distill" + str(epoch) + "_" + str(args_yaml.model.model_type) + "_reg-weight_" + str(args.reg_weight) + "_lr" + str(learning_rate) + "_lr_drop" + str(lr_drop) + ".pth"

            print('come here save at', args.output + model_name)
            misc.save_on_master(sam_hq.module.state_dict(), args.output + model_name)
    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")

class training_engine:
    """Main engine class for orchestrating quantization experiments"""

    def __init__(self, strategy_name: str, mode_train: bool, args, datasets=None) -> None:
        
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
            
            valid_im_gt_list = get_im_gt_name_dict([datasets[2]], flag="valid")
            for dataset_dict in valid_im_gt_list:
                dataset_dict["im_path"] = dataset_dict["im_path"][-10:]
                dataset_dict["gt_path"] = dataset_dict["gt_path"][-10:]
            self.valid_dataloaders, self.vals_datasets = create_dataloaders(
                valid_im_gt_list,
                my_transforms=[Resize([1024, 1024])],
                batch_size=self.args.train_prune_rate.batch_size_valid,
                training= False
            )
            
            train_im_gt_list = get_im_gt_name_dict([datasets[1]], flag="train")
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
            valid_im_gt_list = get_im_gt_name_dict(datasets[2:3], flag="valid")
            self.dataloaders, self.datasets = create_calib_dataloaders(
                valid_im_gt_list,
                my_transforms=[Resize([1024, 1024])],
                batch_size=self.args.train_prune_rate.batch_size_valid,
            )
       
       
        
    def monkey_patch(self, predictor,processor, encoder_config=None,train=False):
        if self.args.train_prune_rate.training_method == "duo":
            print("Applying encoder quantization...")
            image_encoder_monkey_patch_train_duo(
                predictor.model,
                processor=processor,
                args_yaml= self.args,
                train = train,
            )
        elif self.args.train_prune_rate.training_method == "diffduo":
            image_encoder_monkey_patch_train_duo_diff(
                predictor.model,
                processor=processor,
                args_yaml= self.args,
                train = train,
            )

    def eval_hq44k(self, predictor: SamPredictor, processor= None, num_samples=None, checkpoint_evaluation=None, plot_figures=False):
        """Delegate to evaluator component"""
        

        checkpoint_path = checkpoint_evaluation #"./pretrained_checkpoint/prune_rate/duo_sam_hq_epoch_torchnograd10_vit_b_reg-weight_0.5_lr0.1_lr_drop2.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        predictor.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        if self.args.quantization.use_percentage:
            processor.calculate_pruned_heads_per_layer_percent_based(predictor)
        # Print pruned heads information
        if "distill" in checkpoint_path:
            states='distillation'+self.args.model.model_type
        else:
            states='torch_nograd'
        if self.args.train_prune_rate.training_method == "diffduo":
            states = "diffduo" + states
        logger = setup_logger("./logs", states)

        threshold = self.args.train_prune_rate.threshold
        global_threshold = self.args.train_prune_rate.threshold_globle

        logger.info(f"{'='*250}")
        logger.info('Local threshold: {}'.format(threshold))
        logger.info('Global threshold: {}'.format(global_threshold))
        logger.info('Number of Sample {}'.format(num_samples))
        logger.info(f"{'='*120}")
        if not self.args.quantization.use_percentage:
            pruned_count, total_count = print_pruned_heads_info(predictor.model, threshold, global_threshold, logger, self.args.model.model_type)    
        logger.info("Local percent: {}".format(self.args.quantization.percent_entropy))
        logger.info("Global percent: {}".format(self.args.quantization.percent_entropy_global))
        # exit()

        sam = predictor.model
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        sam = sam.to(self.device)
        sam.eval()  
        self.evaluator = Evaluator(self.accelerator, self.dataloaders, self.datasets)
        
        results= self.evaluator.eval_hq44k(predictor, num_samples, plot_figures)
        logger.info(f"{'='*120}")
        keys_list = list(results.keys()) 
        for i in range(len(results)):
            logger.info(f'{keys_list[i]}: {results[keys_list[i]]}')
        logger.info(f"{'='*250}")
        return results
    def train_model(self, predictor, args_yaml):
        sam = predictor.model
        
        print("--- define optimizer ---")
        # Collect ONLY selected_probability parameters for the optimizer
        trainable_params = []
        for name, param in sam.named_parameters():
            if 'full_attention_heads' in name or "selected_probability" in name:
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
        if args_yaml.train_prune_rate.training_method == "duo":
            wandb.init(project="sam-hq-training-duo", name=f"experiment_{self.strategy_name}-distill-model__{args_yaml.model.model_type}-lr-{args_yaml.train_prune_rate.learning_rate}-lr_drop_{args_yaml.train_prune_rate.lr_drop_epoch}-reg_weight_{args_yaml.train_prune_rate.reg_weight}")
        elif args_yaml.train_prune_rate.training_method == "diffduo":
            wandb.init(project="sam-hq-training-diffduo", name=f"experiment_{self.strategy_name}-distill-model__{args_yaml.model.model_type}-lr-{args_yaml.train_prune_rate.learning_rate}-lr_drop_{args_yaml.train_prune_rate.lr_drop_epoch}-reg_weight_{args_yaml.train_prune_rate.reg_weight}")
        train(args_yaml.train_prune_rate, sam,  optimizer, self.train_dataloaders, self.valid_dataloaders, lr_scheduler)
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
    parser.add_argument('--checkpoint-evaluation', type=str)
    
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
    
    
    processor = None
    if args_yaml.train_prune_rate.training_method == "diffduo":
        processor = get_encoder_processor("PRUNE_RATE")
        processor.calibrate(
            predictor=predictor,
            modules=( EncoderAttentionTraining, EncoderAttention, EncoderSamAttention),
            num_samples=args.num_calib_samples
        )
        processor.set_params(args_yaml)
    if args_yaml.train_prune_rate.training_method == "duo":
        processor = get_encoder_processor("PRUNE_RATE_DUO")
        processor.set_params(args_yaml)
        


    encoder_config =  None
    engine.monkey_patch(predictor,processor, encoder_config, args.train)
    # print_model_structure(predictor.model,"Final structure ")
    # exit()
    if args.train:
        engine.train_model(predictor=predictor, args_yaml=args_yaml)
    else:
        results = engine.eval_hq44k(predictor=predictor,processor=processor , num_samples=args.num_samples, checkpoint_evaluation= args.checkpoint_evaluation, plot_figures=False)


    # 