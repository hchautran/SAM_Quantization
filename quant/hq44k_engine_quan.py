# %%
    
import os
import yaml
import cv2
import torch
from typing import Tuple
from engine import InferenceStrategy, Engine
import numpy as np
from tqdm.auto import tqdm
from train.utils.dataloader import get_im_gt_name_dict, create_dataloaders, Resize
from train.utils.misc import   F, random
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou, show_anns, MaskDecoderHQ
from train.segment_anything_training import sam_model_registry
# from segment_anything import sam_model_registry
import cv2
from tqdm.auto import tqdm
import json
import time
from omegaconf import OmegaConf
import argparse
import logging
import ipdb
import os
import sys
import importlib
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
quarot_path = os.path.join(project_root, 'quarot')
qgemm_dir = os.path.join(project_root, 'quant', 'qgemm')
rtn_cuda_dir = os.path.join(qgemm_dir, 'cuda_rtn_gptq') 
sys.path.insert(0, project_root)  
sys.path.insert(0, quarot_path) 
sys.path.insert(0, qgemm_dir)  
sys.path.insert(0, rtn_cuda_dir) 
from quarot import parser_gen

from distribution_sam import get_channel_distribution_modify
import RTN_quantization.utils as rtn_utils
from RTN_quantization import per_tensor_channel_group,gptq_utils
from RTN_quantization.utils import QuantizationConfig, replace_linear_with_quantized,replace_attention_with_quantized
import rotate_sam
from quantizer import replace_linear_with_int4 ,save_cuda_quantized_model, replace_linear_with_int4_gptq
from torch import nn
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from segment_anything.utils.transforms import ResizeLongestSide
def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

    # parser.add_argument("--output", type=str, required=True, 
                        # help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    # parser.add_argument("--checkpoint", type=str, required=True, 
                        # help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=12, type=int)
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")
    parser.add_argument('--logging_path', type=str, default='./logs')
 
    
    return parser.parse_args()

def setup_logger(path_log,state):
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(path_log, f'{state}.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
def print_model_structure(model, title="Model Structure"):
    print(f"\n{title}")
    print("=" * len(title))
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")
    print("=" * len(title))
def print_model_parameters(model, model_name="Model"):
    """
    Print the number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        model_name: Name to display for the model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"{model_name} Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Non-trainable: {non_trainable_params:,}")
    print(f"  Memory (approx): {total_params * 4 / 1024**2:.2f} MB")  # Assuming float32
    print("-" * 50)

class Hq44kInferenceStrategy(InferenceStrategy):
    def __init__(self, args):
        # build sam encoder
        
        self.checkpoint = args.model.checkpoint
        self.model_type = args.model.model_type
        self.restore_model = args.model.restore_model
        self.device = args.model.device
        self.predictor = None
        self.quant_rtn = args.quantization.quanrtn
        self.quant_gptq = args.quantization.quangptq
        self.quant_smooth = args.quantization.quansmooth
        self.quant_ro  = args.quantization.quanro
        self.act_scales_file = args.quantization.act_scales_file
        self.act_quant = args.quantization.act_quant
        self.weight_quant = args.quantization.weight_quant
        self.n_bits = args.quantization.n_bits
        self.quantize_output = args.quantization.quantize_output
        self.quantize_decoder = args.quantization.quandecoder
        self.rtn_cuda = args.quantization.rtn_cuda
        self.gptq_cuda = args.quantization.gptq_cuda
        self.low_high_density =args.quantization.low_high_density
        self.up_down_RTN = args.quantization.up_down_RTN
        self.qkT_v = args.quantization.qkT_v
        if self.low_high_density != "none":
            self.percent = args.quantization.percent
        if self.quant_gptq or self.gptq_cuda:
            self.args_gptq = args.gptq
        if self.rtn_cuda:
            self.save_rtn_cuda = args.quantization.save_rtn_cuda
        if self.quant_rtn and 'rtn_ro_config'  in args:
            self.rtn_ro = args.rtn_ro_config
        else:
            self.rtn_ro = None
        
        if self.quant_ro:
            self.rot_args = args.quarot_inf
        else:
            self.rot_args = None
    def build_predictor(self):
        self.hq_mask_decoder = MaskDecoderHQ(self.model_type) 
        self.predictor = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        # print_model_parameters(self.predictor, model_name="SAM Image Encoder")
        # print_model_parameters(self.hq_mask_decoder, model_name="HQ Mask Decoder")
        # exit()
        if self.restore_model:
            print("restore model from:", self.restore_model)
            self.hq_mask_decoder.load_state_dict(torch.load(self.restore_model))
        
        if self.quant_smooth:
            assert self.act_scales_file is not None, "Run Smooth_sam.py to generate act_scales_file"
            act_scales = torch.load(self.act_scales_file)
            self.predictor = rtn_utils.smooth_sam(self.predictor, act_scales, alpha=0.5)
            if self.quantize_decoder:
                self.hq_mask_decoder = rtn_utils.smooth_sam(self.hq_mask_decoder, act_scales, alpha=0.5)
        elif self.quant_ro:
            
            rotate_sam.rotate_sam(self.predictor,self.rot_args,self.rtn_ro)
            self.quant_rtn = False
            if self.quantize_decoder:
                rotate_sam.rotate_sam(self.hq_mask_decoder,self.rot_args,self.rtn_ro,decoder= True)
                if not self.quant_gptq:
                    self.quantize_decoder = False
        if self.quant_gptq or self.gptq_cuda:
            from train.utils.dataloader import OnlineDataset
            from torchvision import transforms
            import numpy as np
            
            # Dataset configuration
            dataset_coift_val = {"name": "COIFT",
                    "im_dir": "./data/thin_object_detection/COIFT/images",
                    "gt_dir": "./data/thin_object_detection/COIFT/masks",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

            # Get file list
            valid_im_gt_list = get_im_gt_name_dict([dataset_coift_val], flag="valid")
            
            # Create dataset with transforms
            transform = transforms.Compose([Resize([1024,1024])])
            gos_dataset = OnlineDataset(valid_im_gt_list, transform=transform, eval_ori_resolution=True)
            
            # Create simple dataloader without DistributedSampler
            valid_dataloader = torch.utils.data.DataLoader(
                gos_dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=0,
                drop_last=False
            )
            
            print(f"Created dataloader with {len(valid_dataloader)} samples for GPTQ")
            
            # Run GPTQ quantization
            quantizer = gptq_utils.gptq_fwrd_sam(
                self.predictor, 
                valid_dataloader, 
                self.args_gptq.device, 
                self.args_gptq
            )
            
            # modules_to_exclude = ["pos_embed", "cls_token", "patch_embed", "neck", "fpn", "mask_tokens", "iou_token", "output_upscaling", "output_hypernetworks_mlps"]
            modules_to_exclude = []
            if self.gptq_cuda:
                replace_linear_with_int4_gptq(self.predictor,quantizer, exclude_modules=modules_to_exclude)
            else:
                # Use new QuantizationConfig API
                config = QuantizationConfig(
                    n_bits_w=self.n_bits,
                    n_bits_a=self.args_gptq.ac_bits,
                    weight_quant=self.weight_quant,
                    act_quant=self.act_quant,
                    quantize_output=self.quantize_output,
                    quantize_weight=False  # weight already quantized in gptq
                )
                replace_linear_with_quantized(self.predictor, config, modules_to_exclude)
            
                
            if self.quantize_decoder:
              
                self.predictor = self.predictor.to('cuda')
                self.hq_mask_decoder.eval()
                
                # Collect encoder outputs by processing the dataloader properly
                batched_outputs = []
                interm_embeddings_list = []
                for i,data_batch in enumerate(valid_dataloader):
                    if i >= self.args_gptq.nsamples:
                        break
                    print(f"Processing sample {i+1}/{self.args_gptq.nsamples} for GPTQ image encoder outputs", end='\r')
                    _, inputs_val, labels_val, _, _ = data_batch['imidx'], data_batch['image'], data_batch['label'], data_batch['shape'], data_batch['ori_label']
                    
                    # Convert to device
                    if torch.cuda.is_available():
                        inputs_val = inputs_val.cuda()
                        labels_val = labels_val.cuda()
                    
                    # Prepare inputs in the same format as inference time
                    imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
                    labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])
                    
                    batched_input = []
                    for b_i in range(len(imgs)):
                        dict_input = dict()
                        # Convert to tensor and move to CUDA to match model's device
                        input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device='cuda').permute(2, 0, 1).contiguous()
                        dict_input['image'] = input_image 
                        dict_input['boxes'] = labels_box[b_i:b_i+1]  # Keep boxes on original device
                        dict_input['original_size'] = imgs[b_i].shape[:2]
                        batched_input.append(dict_input)
                    
                    # Get encoder outputs
                    batch_output, batch_interm = self.predictor(batched_input, multimask_output=False)
                    batched_outputs.extend(batch_output)
                    interm_embeddings_list.extend(batch_interm)
                    
                    # Limit to nsamples for GPTQ
                    if len(batched_outputs) >= self.args_gptq.nsamples:
                        batched_outputs = batched_outputs[:self.args_gptq.nsamples]
                        interm_embeddings_list = interm_embeddings_list[:self.args_gptq.nsamples]
                        break
                
                # Process the collected outputs
                batch_len = len(batched_outputs)
                encoder_embedding = torch.cat([batched_outputs[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0).to(self.args_gptq.device)
                image_pe = [batched_outputs[i_l]['image_pe'].to(self.args_gptq.device) for i_l in range(batch_len)]
                sparse_embeddings = [batched_outputs[i_l]['sparse_embeddings'].to(self.args_gptq.device) for i_l in range(batch_len)]
                dense_embeddings = [batched_outputs[i_l]['dense_embeddings'].to(self.args_gptq.device) for i_l in range(batch_len)]
                interm_embeddings_device = [interm_emb.to(self.args_gptq.device) if isinstance(interm_emb, torch.Tensor) else interm_emb for interm_emb in interm_embeddings_list]

                quantizer = gptq_utils.gptq_fwrd_sam_maskdecoder(self.hq_mask_decoder,
                    image_embeddings=encoder_embedding,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    hq_token_only=False,
                    interm_embeddings=interm_embeddings_device,
                    dev=self.args_gptq.device,
                    args=self.args_gptq)
                # modules_to_exclude = ["pos_embed", "cls_token", "patch_embed", "neck", "fpn", "mask_tokens", "iou_token", "output_upscaling","output_hypernetworks_mlps"]
                moduel_to_exculde =[]
                if self.gptq_cuda:
                    replace_linear_with_int4_gptq(self.hq_mask_decoder,quantizer, exclude_modules=modules_to_exclude)
                else:
                    # Use new QuantizationConfig API
                    config = QuantizationConfig(
                        n_bits_w=self.n_bits,
                        n_bits_a=self.args_gptq.ac_bits,
                        weight_quant=self.weight_quant,
                        act_quant=self.act_quant,
                        quantize_output=self.quantize_output,
                        quantize_weight=False  # weight already quantized in gptq
                    )
                    replace_linear_with_quantized(self.hq_mask_decoder, config, modules_to_exclude)
                
                del batched_outputs
                del interm_embeddings_list
                del encoder_embedding
                del image_pe
                del sparse_embeddings
                del dense_embeddings
                del interm_embeddings_device
            
         
            del valid_dataloader
            if quantizer:
                del quantizer
            import gc
            gc.collect()    
            torch.cuda.empty_cache()
            # print(f"Memory after cleanup: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
           
        if self.quant_rtn:
            modules_to_exclude = []
            if self.quantize_decoder:
                # Use new QuantizationConfig API
                config = QuantizationConfig(
                    n_bits_w=4,
                    n_bits_a=8,
                    weight_quant=self.weight_quant,
                    act_quant=self.act_quant,
                    quantize_output=self.quantize_output,
                    up_down_RTN=self.up_down_RTN
                )
                replace_linear_with_quantized(self.hq_mask_decoder, config, modules_to_exclude)
        
        if self.rtn_cuda:
            # modules_to_exclude = ["pos_embed", "cls_token", "patch_embed", "neck", "fpn", "mask_tokens", "iou_token", "output_upscaling","output_hypernetworks_mlps"]
            modules_to_exclude=[]
            replace_linear_with_int4(self.predictor,  exclude_modules=modules_to_exclude)
            if self.quantize_decoder:
                replace_linear_with_int4(self.hq_mask_decoder,  exclude_modules=modules_to_exclude)
                
            if self.save_rtn_cuda:
                save_cuda_quantized_model(self.predictor, save_dir="./pretrained_checkpoint", model_name="sam_int4_full")
                if self.quantize_decoder:
                    save_cuda_quantized_model(self.hq_mask_decoder, save_dir="./pretrained_checkpoint", model_name="hq_decoder_int4_full")
        if self.low_high_density != "none":
            modules_to_exclude=["mask_decoder"]
            quantizehigh_ = self.low_high_density != "low"

            # Use new QuantizationConfig API for predictor
            config = QuantizationConfig(
                n_bits_w=4,
                n_bits_a=4,
                weight_quant="per_channel",
                act_quant="low_high_density_activation",
                quantize_output=False,
                quantize_weight=True,
                quantizehigh=quantizehigh_,
                percent=self.percent
            )
            replace_linear_with_quantized(self.predictor, config, modules_to_exclude)

            if self.quantize_decoder:
                # Use new QuantizationConfig API for decoder
                config_decoder = QuantizationConfig(
                    n_bits_w=4,
                    n_bits_a=4,
                    weight_quant="per_channel",
                    act_quant="low_high_density_activation",
                    quantize_output=self.quantize_output,
                    quantize_weight=True,
                    quantizehigh=quantizehigh_,
                    percent=self.percent
                )
                replace_linear_with_quantized(self.hq_mask_decoder, config_decoder, modules_to_exclude)   
        if self.qkT_v:
            replace_attention_with_quantized(self.predictor)
            if self.quantize_decoder:
                raise NotImplementedError("QkT_v for decoder not implemented yet")
        # self.plot_distribution()
            
        # print_model_structure(self.predictor, title="Final Structure")
        # print_model_structure(self.hq_mask_decoder, title="Final HQ Mask Decoder Structure")
        # exit()
    def plot_distribution(self):
        act = ''
        if self.quant_rtn:
            act += "rtn"
        if self.quant_smooth:
            act += "smooth"
        if self.quant_ro:
            act += "ro_"
        get_channel_distribution_modify(self.predictor,self.hq_mask_decoder,model_type="vit_l",act = act, rot_args = self.rot_args)
        
    def set_image(self, image_dir:str):
        raise NotImplementedError("")

    def set_video(self, video_dir:str):
        raise NotImplementedError("Video inference is not supported for SAM")
        
    @torch.inference_mode()
    @torch.no_grad()
    def inference(self, inputs:dict):
        self.hq_mask_decoder.eval()
        # encoder image and prompts
        batched_output, interm_embeddings = self.predictor(inputs, multimask_output=False) 
        batch_len = len(batched_output)
        encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
        image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
        sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
        dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

        # decode high quality mask
        masks_sam, masks_hq = self.hq_mask_decoder(
            image_embeddings=encoder_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            hq_token_only=False,
            interm_embeddings=interm_embeddings,
        )
        

        return masks_sam, masks_hq 
    def inference_visual(self, inputs:dict):
        self.hq_mask_decoder.eval()
        # encoder image and prompts
        # if self.quant_ro:
        
        
        self.predictor.to(self.device)
        self.hq_mask_decoder.to(self.device)
       
        for i in range(len(inputs)):
            for key, value in inputs[i].items():
                if isinstance(value, torch.Tensor):
                    inputs[i][key] = value.to(self.device)
                elif isinstance(value, list):
                    inputs[i][key] = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in value]
                elif isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        if isinstance(nested_value, torch.Tensor):
                            inputs[i][key][nested_key] = nested_value.to(self.device)
        batched_output, interm_embeddings = self.predictor(inputs, multimask_output=False) 
        batch_len = len(batched_output)
        encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
        image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
        sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
        dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
        
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2) # early-layer ViT feature, after 1st global attention block in ViT
        hq_features = self.hq_mask_decoder.embedding_encoder(encoder_embedding) + self.hq_mask_decoder.compress_vit_feat(vit_features)

        batch_len = len(encoder_embedding)
        masks = []
        iou_preds = []

        for i_batch in range(batch_len):
            mask, iou_pred = self.hq_mask_decoder.predict_masks(
                image_embeddings=encoder_embedding[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_embeddings[i_batch],
                dense_prompt_embeddings=dense_embeddings[i_batch],
                hq_feature = hq_features[i_batch].unsqueeze(0)
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks,0)
        iou_preds = torch.cat(iou_preds,0)
        
        mask_slice = slice(0, 1)
        masks_sam = masks[:,mask_slice]
        masks_hq = masks[:,slice(self.hq_mask_decoder.num_mask_tokens-1, self.hq_mask_decoder.num_mask_tokens), :, :]
        iou_preds= iou_preds[:,mask_slice]
        return masks_sam, masks_hq, iou_preds
        
        
    def visualize(self, prompts:dict, masks:torch.Tensor, scores:torch.Tensor, result_path:str):
        raise NotImplementedError("")
        



#%%
class Hq44kSamEngine(Engine):
    def __init__(self, strategy:InferenceStrategy):
        super().__init__(strategy)
        self.strategy.build_predictor()
        dataset_dis = {"name": "DIS5K-TR",
                    "im_dir": "./data/DIS5K/DIS-TR/im",
                    "gt_dir": "./data/DIS5K/DIS-TR/gt",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_thin = {"name": "ThinObject5k-TR",
                    "im_dir": "./data/thin_object_detection/ThinObject5K/images_train",
                    "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_fss = {"name": "FSS",
                    "im_dir": "./data/cascade_psp/fss_all",
                    "gt_dir": "./data/cascade_psp/fss_all",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_duts = {"name": "DUTS-TR",
                    "im_dir": "./data/cascade_psp/DUTS-TR",
                    "gt_dir": "./data/cascade_psp/DUTS-TR",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_duts_te = {"name": "DUTS-TE",
                    "im_dir": "./data/cascade_psp/DUTS-TE",
                    "gt_dir": "./data/cascade_psp/DUTS-TE",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_ecssd = {"name": "ECSSD",
                    "im_dir": "./data/cascade_psp/ecssd",
                    "gt_dir": "./data/cascade_psp/ecssd",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_msra = {"name": "MSRA10K",
                    "im_dir": "./data/cascade_psp/MSRA_10K",
                    "gt_dir": "./data/cascade_psp/MSRA_10K",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        # valid set
        dataset_coift_val = {"name": "COIFT",
                    "im_dir": "./data/thin_object_detection/COIFT/images",
                    "gt_dir": "./data/thin_object_detection/COIFT/masks",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_hrsod_val = {"name": "HRSOD",
                    "im_dir": "./data/thin_object_detection/HRSOD/images",
                    "gt_dir": "./data/thin_object_detection/HRSOD/masks",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_thin_val = {"name": "ThinObject5k-TE",
                    "im_dir": "./data/thin_object_detection/ThinObject5K/images",
                    "gt_dir": "./data/thin_object_detection/ThinObject5K/masks",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_dis_val = {"name": "DIS5K-VD",
                    "im_dir": "./data/DIS5K/DIS-VD/im",
                    "gt_dir": "./data/DIS5K/DIS-VD/gt",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        self.train_datasets = [dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra]
        self.valid_datasets = [
            dataset_dis_val, dataset_coift_val, 
            # dataset_hrsod_val,
            #  dataset_thin_val
        ] 

        
    def train(self, args:dict):
        pass

    def distribute(self, dist_args):
        if torch.cuda.is_available():
            self.strategy.predictor.to(device=dist_args.device)
            self.strategy.hq_mask_decoder.to(device=dist_args.device)
            self.strategy.predictor = torch.nn.parallel.DistributedDataParallel(self.strategy.predictor, device_ids=[dist_args.gpu], find_unused_parameters=dist_args.find_unused_params)
            self.strategy.hq_mask_decoder = torch.nn.parallel.DistributedDataParallel(self.strategy.hq_mask_decoder, device_ids=[dist_args.gpu], find_unused_parameters=dist_args.find_unused_params)
        else:
            raise NotImplementedError("Distributed training supported on this machine")


    @torch.no_grad()
    def demo(self,):
        pass 

# %%
    @torch.no_grad()
    def evaluate(self, args, model_args ,visualize:bool=False):
        state="hq44k_"
        if model_args.quantization.quanrtn:
            state +="rtn_"+model_args.quantization.up_down_RTN
        if model_args.quantization.quansmooth:
            state += "smooth"
        if model_args.quantization.quanro:
            state += "ro"
        if model_args.quantization.quandecoder:
            state += "dec"
        if model_args.quantization.quangptq:
            state += "gptq"
        if model_args.quantization.rtn_cuda:
            state += "rtncuda"
        if model_args.quantization.gptq_cuda:
            state += "gptqcuda"
        if model_args.quantization.low_high_density != "none":
            state+= "lh_"+ model_args.quantization.low_high_density
        if model_args.quantization.qkT_v:
            state += "qkTv"
        logger =setup_logger(args.logging_path,state)
        
        misc.init_distributed_mode(args)
        print('world size: {}'.format(args.world_size))
        print('rank: {}'.format(args.rank))
        print('local_rank: {}'.format(args.local_rank))
        print("args: " + str(args) + '\n')
        logger.info('world size: {}'.format(args.world_size))
        logger.info('rank: {}'.format(args.rank))
        logger.info('local_rank: {}'.format(args.local_rank))
        logger.info("args: " + str(args) + '\n')
        logger.info('model_args: ' + str(model_args) + '\n')
        logger.info("=" * 100)
        
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.distribute(args)

        device = self.strategy.predictor.device
        valid_im_gt_list = get_im_gt_name_dict(self.valid_datasets, flag="valid")
        valid_dataloaders, _ = create_dataloaders(
            valid_im_gt_list,
            my_transforms = [
                        Resize(args.input_size)
                    ],
            batch_size=args.batch_size_valid,
            training=False
        )
        test_stats = {}
        for k in range(len(valid_dataloaders)):
            
            metric_logger = misc.MetricLogger(delimiter="  ")
            valid_dataloader = valid_dataloaders[k]
            print('valid_dataloader len:', len(valid_dataloader))
            logger.info(f"\nValidating {self.valid_datasets[k]['name']}:")
            progress_bar = tqdm(total=len(valid_dataloader), desc=f"Validating {self.valid_datasets[k]['name']}")
            start = time.time()
            for i,data_val in enumerate(metric_logger.log_every(valid_dataloader, 2)):
                if i >5:
                    break
                _, inputs_val, labels_val, _, labels_ori = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label']
                # prepare image & prompts 
                if torch.cuda.is_available():
                    inputs_val = inputs_val.cuda()
                    labels_val = labels_val.cuda()
                    labels_ori = labels_ori.cuda()

                imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
                labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])

                batched_input = []
                for b_i in range(len(imgs)):
                    dict_input = dict()
                    input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=device).permute(2, 0, 1).contiguous()
                    dict_input['image'] = input_image 
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                    dict_input['original_size'] = imgs[b_i].shape[:2]
                    batched_input.append(dict_input)

                _, masks_hq = self.strategy.inference(batched_input)
                # compute metric & update
                iou = compute_iou(masks_hq,labels_ori)
                boundary_iou = compute_boundary_iou(masks_hq,labels_ori)
                loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
                loss_dict_reduced = misc.reduce_dict(loss_dict)
                metric_logger.update(**loss_dict_reduced)
                progress_bar.update(1)
            total_time = time.time() - start

            print('=' * 100) 
            logger.info('=' * 100)
            logger.info(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))} ({total_time / len(valid_dataloader):.4f} s / it)")      
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            logger.info(f"Averaged stats: {metric_logger}")
            resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
            test_stats.update(resstat)
            test_stats['total_time'] = f'{total_time:2f}'
            logger.info(f"{resstat}")
            
            print(test_stats)
            logger.info(f"test stats: {test_stats}")
            logger.info(f"Finished validating {self.valid_datasets[k]['name']}")
            logger.info("=" * 100)
        logger.info("\n" + "=" * 100)
        logger.info("FINAL EVALUATION SUMMARY:")
        logger.info(f"Final test stats: {test_stats}")
        logger.info("=" * 100)
        return test_stats
 # %%       
    @torch.no_grad()
    def visual_eval(self, args, model_args):
        state=""
        if model_args.quantization.quanrtn:
            state +="rtn "
        if model_args.quantization.quansmooth:
            state += "smooth "
        if model_args.quantization.quanro:
            state += "ro "
        if model_args.quantization.quandecoder:
            state += "decoder "
        if model_args.quantization.quangptq:
            state += "gptq "
        if model_args.quantization.rtn_cuda:
            state += "rtncuda "
        if model_args.quantization.gptq_cuda:
            state += "gptqcuda "
            
        device = self.strategy.predictor.device 
        for i in range(8):
            print("image:   ",i)
            # hq_token_only: False means use hq output to correct SAM output. 
            #                True means use hq output only. 
            #                Default: False
            hq_token_only = False 
            # To achieve best visualization effect, for images contain multiple objects (like typical coco images), we suggest to set hq_token_only=False
            # For images contain single object, we suggest to set hq_token_only = True
            # For quantiative evaluation on COCO/YTVOS/DAVIS/UVO/LVIS etc., we set hq_token_only = False

            image = cv2.imread('/home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/sam-hq/demo/input_imgs/example'+str(i)+'.png')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes_add = generate_random_bboxes(0, image.shape[:2])

            if i==0:
                continue
                input_box = np.array([[4,13,1007,1023]])
                input_point, input_label = None, None
            elif i==1:
                continue
                input_box = np.array([[306, 132, 925, 893]])
                input_point, input_label = None, None
                hq_token_only = True
            elif i==2:
                continue
                input_point = np.array([[495,518],[217,140]])
                input_label = np.ones(input_point.shape[0])
                input_box = None
                hq_token_only = True
            elif i==3:
                continue
                input_point = np.array([[221,482],[498,633],[750,379]])
                input_label = np.ones(input_point.shape[0])
                input_box = None
            elif i==4:
                continue
                input_box = np.array([[64,76,940,919]])
                input_point, input_label = None, None
                hq_token_only = True
            elif i==5:
                # continue
                input_point = np.array([[373,363], [452, 575]])
                input_label = np.ones(input_point.shape[0])
                input_box = None
            elif i==6:
                # continue
                input_box = np.array([[181, 196, 757, 495]])
                input_point, input_label = None, None
            elif i==7:
                # multi box input
          
                transform_ = ResizeLongestSide(self.strategy.predictor.image_encoder.img_size)
                input_box = torch.tensor([[45,260,515,470], [310,228,424,296]],device=device)
                input_box = torch.cat([input_box, torch.tensor(boxes_add, device=device)], dim=0)
                transformed_box = transform_.apply_boxes_torch(input_box, image.shape[:2])
                input_point, input_label = None, None

            batch_box = False if input_box is None else len(input_box)>1 
            result_path = 'demo/hq_sam_result/'
            os.makedirs(result_path, exist_ok=True)
            os.makedirs(result_path + f"comparision/{state}/", exist_ok=True)

            # Format input for SAM model
            batched_input = format_input_for_sam(
                image=image, 
                input_point=input_point, 
                input_label=input_label, 
                input_box=input_box,
                device=device
            )

            ori_outputpath = result_path + "original_sam/"
            if not batch_box: 
                _, masks, scores = self.strategy.inference_visual(batched_input)
              
           
                postprocess_mask= postprocess_masks(masks=masks,image_encoder_image_size= self.strategy.predictor.image_encoder.img_size, input_size=batched_input[0]['image'].shape[1:], original_size= batched_input[0]['original_size']) # original_size = image.shape[:2]
                masks = (postprocess_mask>0)[0].bool().unsqueeze(0)
                
                masks = masks.clone().detach()
                show_res(masks,scores,input_point, input_label, input_box, result_path + 'example'+str(i), image)
                if state != "":
                    # Load and compare with original results
                    original_file = os.path.join(ori_outputpath, f'example{i}_baseline.pkl')
                    if os.path.exists(original_file):
                        compare_and_visualize(
                            original_file=original_file,
                            state=state,
                            current_masks=masks,
                            current_scores=scores,
                            input_point=input_point,
                            input_label=input_label,
                            input_box=input_box,
                            image=image,
                            save_path=result_path + f"comparision/{state}/" + f'comparison_example{i}_{state}.png'
                        )
                    else:
                        print(f"Warning: Baseline file {original_file} not found. Cannot create comparison.")
                else:
                    # Save output mask and score as baseline when no quantization
                    
                    os.makedirs(ori_outputpath, exist_ok=True)
                    baseline_file = os.path.join(ori_outputpath, f'example{i}_baseline.pkl')
                    save_baseline_results(masks, scores, baseline_file)
                    print(f"Saved baseline results to {baseline_file}")
                    
            else:
                _, masks, scores = self.strategy.inference_visual(batched_input)
                masks = masks.squeeze(1).cpu().numpy()
                scores = scores.squeeze(1).cpu().numpy()
                input_box = input_box.cpu().numpy()
                show_res_multi(masks, scores, input_point, input_label, input_box, result_path + 'example'+str(i), image)
                if state != "":
                    # Load and compare with original results for multi-box case
                    original_file = os.path.join(ori_outputpath, f'example{i}_baseline.pkl')
                    if os.path.exists(original_file):
                        compare_and_visualize(
                            original_file=original_file,
                            state=state,
                            current_masks=masks,
                            current_scores=scores,
                            input_point=input_point,
                            input_label=input_label,
                            input_box=input_box,
                            image=image,
                            save_path=result_path + f"comparision/{state}/" +f'comparison_example{i}_{state}.png'
                        )
                    else:
                        print(f"Warning: Baseline file {original_file} not found. Cannot create comparison.")
                else:
                    # Save output masks and scores as baseline when no quantization
                    os.makedirs(ori_outputpath, exist_ok=True)
                    baseline_file = os.path.join(ori_outputpath, f'example{i}_baseline.pkl')
                    save_baseline_results(masks, scores, baseline_file)
                    print(f"Saved baseline results to {baseline_file}")
                    

# %%
def generate_random_bboxes(n, image_shape):

    height, width = image_shape[:2]
    bboxes = []
    
    for _ in range(n):
        # Generate two random points
        x1 = np.random.randint(0, width)
        y1 = np.random.randint(0, height)
        x2 = np.random.randint(0, width)
        y2 = np.random.randint(0, height)
        
        # Ensure x1 < x2 and y1 < y2 (top-left, bottom-right format)
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        # Ensure minimum box size of 1x1
        if x_max == x_min:
            x_max = min(x_min + 1, width - 1)
        if y_max == y_min:
            y_max = min(y_min + 1, height - 1)
            
        bboxes.append([x_min, y_min, x_max, y_max])
    
    return np.array(bboxes)
def postprocess_masks(
        masks: torch.Tensor,
        image_encoder_image_size,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (image_encoder_image_size, image_encoder_image_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks.cpu()
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())
        score_value = score.item() if isinstance(score, torch.Tensor) else score
        print(f"Score: {score_value:.3f}")
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.savefig(filename +'.png',bbox_inches='tight',pad_inches=-0.1)
    plt.close()

def format_input_for_sam(image, input_point=None, input_label=None, input_box=None, device='cuda'):
    input_image = torch.as_tensor(image.astype(dtype=np.uint8), device=device).permute(2, 0, 1).contiguous()

    dict_input = {
        'image': input_image,
        'original_size': image.shape[:2]  # (H, W)
    }
    
    # Format points with consistent dimensions
    if input_point is not None and input_label is not None:
        # Ensure points are in correct format: (batch_size, num_points, 2)
        if isinstance(input_point, np.ndarray):
            input_point = torch.as_tensor(input_point, device=device, dtype=torch.float)
        if isinstance(input_label, np.ndarray):
            input_label = torch.as_tensor(input_label, device=device, dtype=torch.int)
            
        # Ensure 3D tensor format for points: (1, N, 2)
        if input_point.dim() == 2:
            input_point = input_point.unsqueeze(0)  # Add batch dimension
        if input_label.dim() == 1:
            input_label = input_label.unsqueeze(0)  # Add batch dimension
            
        dict_input['point_coords'] = input_point
        dict_input['point_labels'] = input_label
    
    # Format boxes with consistent dimensions
    if input_box is not None:
        if isinstance(input_box, np.ndarray):
            input_box = torch.as_tensor(input_box, device=device, dtype=torch.float)
        
        # Ensure 2D tensor format for boxes: (N, 4)
        if input_box.dim() == 1:
            input_box = input_box.unsqueeze(0)  # Add batch dimension
            
        dict_input['boxes'] = input_box

    return [dict_input]
def compare_and_visualize(original_file, state, current_masks, current_scores, input_point, input_label, input_box, image, save_path):
    """
    Compare original model outputs with current model outputs and create visualization
    """
    # Create comparison directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Load original results
    with open(original_file, 'rb') as f:
        original_data = pickle.load(f)
        original_masks = original_data['masks']
        original_scores = original_data['scores']
    
    # Create 2x3 subplot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Comparison Analysis', fontsize=25)
    
    # 1. Original image (top-left)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=20)
    axes[0, 0].axis('off')
    
    # 2. Original image with original mask (top-middle)
    axes[0, 1].imshow(image)
    if len(original_masks) > 0:
        show_mask(original_masks[0], axes[0, 1])
        if input_box is not None:
            box_to_show = input_box[0] if len(input_box.shape) > 1 else input_box
            show_box(box_to_show, axes[0, 1])
        if input_point is not None and input_label is not None:
            show_points(input_point, input_label, axes[0, 1])
    axes[0, 1].set_title('Original Model Output', fontsize=20)
    axes[0, 1].axis('off')
    
    # 3. Original image with modified model mask (top-right)
    axes[0, 2].imshow(image)
    if len(current_masks) > 0:
        show_mask(current_masks[0], axes[0, 2])
        if input_box is not None:
            box_to_show = input_box[0] if len(input_box.shape) > 1 else input_box
            show_box(box_to_show, axes[0, 2])
        if input_point is not None and input_label is not None:
            show_points(input_point, input_label, axes[0, 2])
    axes[0, 2].set_title('Modified Model Output-'+state, fontsize=20)
    axes[0, 2].axis('off')
    
    # 4. Mask difference (bottom-left)
    if len(original_masks) > 0 and len(current_masks) > 0:
        original_mask_np = original_masks[0].cpu().numpy() if isinstance(original_masks[0], torch.Tensor) else original_masks[0]
        current_mask_np = current_masks[0].cpu().numpy() if isinstance(current_masks[0], torch.Tensor) else current_masks[0]
        
        # Squeeze to remove single dimensions and ensure 2D
        original_mask_np = np.squeeze(original_mask_np)
        current_mask_np = np.squeeze(current_mask_np)
        
        # Create binary difference mask (black and white)
        mask_diff = (original_mask_np != current_mask_np).astype(np.uint8)
        
        # Display as black (0) and white (1) image
        im = axes[1, 0].imshow(mask_diff, cmap='gray', vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[1, 0], shrink=0.6)
    axes[1, 0].set_title('Mask Difference\n(Black: Same, White: Different)',fontsize=20)
    axes[1, 0].axis('off')
    
    # 5. Score comparison bar chart (bottom-middle)
    if len(original_scores) > 0 and len(current_scores) > 0:
        categories = ['Original Model', 'Modified Model']
        original_score = original_scores[0].item() if isinstance(original_scores[0], torch.Tensor) else original_scores[0]
        current_score = current_scores[0].item() if isinstance(current_scores[0], torch.Tensor) else current_scores[0]
        
        scores = [original_score, current_score]
        colors = ['blue', 'orange']
        bars = axes[1, 1].bar(categories, scores, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Score', fontsize=15)
        axes[1, 1].set_title('Score Comparison', fontsize=20)
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
    
    # 6. Metrics information (bottom-right)
    axes[1, 2].axis('off')
    if len(original_masks) > 0 and len(current_masks) > 0 and len(original_scores) > 0 and len(current_scores) > 0:
        # Extract scalar values from tensors
        original_score = original_scores[0].item() if isinstance(original_scores[0], torch.Tensor) else original_scores[0]
        current_score = current_scores[0].item() if isinstance(current_scores[0], torch.Tensor) else current_scores[0]
        
        # Calculate metrics
        score_diff = abs(original_score - current_score)
        degradation_pct = (score_diff / original_score) * 100 if original_score != 0 else 0

        # Convert tensors to numpy arrays and flatten
        orig_flat = original_masks[0].cpu().numpy().flatten().astype(float) if isinstance(original_masks[0], torch.Tensor) else original_masks[0].flatten().astype(float)
        curr_flat = current_masks[0].cpu().numpy().flatten().astype(float) if isinstance(current_masks[0], torch.Tensor) else current_masks[0].flatten().astype(float)
        mask_mse = mean_squared_error(orig_flat, curr_flat)
        
        # Create text summary
        metrics_text = f"""Model Comparison Metrics:

                        Score Difference: {score_diff:.4f}

                        Degradation: {degradation_pct:.2f}%

                        Mask MSE: {mask_mse:.6f}

                        Original Score: {original_score:.4f}
                        Modified Score: {current_score:.4f}"""
        
        axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes, 
                    fontsize=12, verticalalignment='top')
    
    axes[1, 2].set_title('Metrics Summary',fontsize=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_baseline_results(masks, scores, save_path):
    """Save original model results as baseline"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    baseline_data = {
        'masks': masks,
        'scores': scores
    }
    with open(save_path, 'wb') as f:
        pickle.dump(baseline_data, f)
def reset_everything():
    """Complete reset between runs"""
    # 1. Clear all loggers
    logging.shutdown()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.Logger.manager.loggerDict.clear()

    # 2. Destroy distributed process group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    # 3. Clear CUDA cache
    torch.cuda.empty_cache()

    # 4. Force garbage collection
    import gc
    gc.collect()
# %%


if __name__ == "__main__":
    import csv
    from datetime import datetime

    model_args = OmegaConf.load('./quant/config/hq44k/qkT_v.yaml')
    args = get_args_parser()
    engine = Hq44kSamEngine(Hq44kInferenceStrategy(model_args))
    test_stats = engine.evaluate(args, model_args)
    exit()
    
    # Create results directory and CSV file
    os.makedirs('./output/density_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'./output/density_results/results_{timestamp}.csv'

    # Store all results
    all_results = []

    for token_type in ['low', 'high']:
        for percent in [50, 60, 70, 80, 100]:
            print(f"\n{'='*80}")
            print(f"Testing: token_type={token_type}, percent={percent}%")
            print(f"{'='*80}")

            model_args.quantization.percent = percent
            model_args.quantization.low_high_density = token_type

            engine = Hq44kSamEngine(Hq44kInferenceStrategy(model_args))
            test_stats = engine.evaluate(args, model_args)

            # Add configuration to results
            result_row = {
                'token_type': token_type,
                'percent': percent,
                **test_stats
            }
            all_results.append(result_row)

            print(f"\nResults: {test_stats}")

            # Cleanup
            del engine
            reset_everything()

    # Write results to CSV
    if all_results:
        # Get all unique keys from all results
        fieldnames = ['token_type', 'percent'] + [k for k in all_results[0].keys() if k not in ['token_type', 'percent']]

        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n{'='*80}")
        print(f"Results saved to: {csv_filename}")
        print(f"{'='*80}")

        # Print summary table
        print("\nRESULTS SUMMARY:")
        print(f"{'Token Type':<12} {'Percent':<10} {'IoU Metrics'}")
        print('-'*80)
        for result in all_results:
            iou_metrics = {k: v for k, v in result.items() if 'iou' in k.lower() and k != 'total_time'}
            iou_str = ', '.join([f"{k}={v:.4f}" for k, v in iou_metrics.items()])
            print(f"{result['token_type']:<12} {result['percent']:<10} {iou_str}")

    exit()
    # model_args.quantization.up_down_RTN= "RTN"
    test_stats_list = []
    randomsnum=100
    
    original_seed = args.seed
    for i in range(randomsnum):
        print(f"\n=== Running iteration {i+1}/{randomsnum} ===")

        # Reset everything except for the first run
        args.seed = original_seed + i
        if i > 0:
            reset_everything()
        engine = Hq44kSamEngine(Hq44kInferenceStrategy(model_args))
        test_stats=engine.evaluate(args,model_args)
        test_stats_list.append(test_stats)
        
        del engine
        # calculate the average and standard deviation of the results
        if i % 20 == 0:
            if len(test_stats_list) > 1:
                # Extract metric names (excluding 'total_time')
                metric_names = [key for key in test_stats_list[0].keys() if key != 'total_time']

                # Calculate statistics for each metric
                stats_summary = {}
                for metric in metric_names:
                    values = [test_stats[metric] for test_stats in test_stats_list]
                    values_array = np.array(values)

                    mean_val = np.mean(values_array)
                    std_val = np.std(values_array, ddof=1)  # ddof=1 for sample standard deviation
                    var_val = np.var(values_array, ddof=1)   # ddof=1 for sample variance

                    stats_summary[metric] = {
                        'mean': mean_val,
                        'std': std_val,
                        'variance': var_val,
                        'values': values
                    }

                    state="hq44k_"
                    if model_args.quantization.quanrtn:
                        state +="rtn_"+model_args.quantization.up_down_RTN
                    if model_args.quantization.quansmooth:
                        state += "smooth"
                    if model_args.quantization.quanro:
                        state += "ro"
                    if model_args.quantization.quandecoder:
                        state += "dec"
                    if model_args.quantization.quangptq:
                        state += "gptq"
                    if model_args.quantization.rtn_cuda:
                        state += "rtncuda"
                    if model_args.quantization.gptq_cuda:
                        state += "gptqcuda"
                    if model_args.quantization.low_high_density != "none":
                        state+= "lh_"+ model_args.quantization.low_high_density
                    logger =setup_logger(args.logging_path,state)
                    
                    # Log the statistics
                    logger.info(f"{metric}:")
                    logger.info(f"  Values: {values}")
                    logger.info(f"  Mean: {mean_val:.6f}")
                    logger.info(f"  Std Dev: {std_val:.6f}")
                    logger.info(f"  Variance: {var_val:.6f}")

                # Print summary to console as well
                print("\n" + "=" * 100)
                print("STATISTICAL SUMMARY:")
                print("=" * 100)
                for metric in metric_names:
                    stats = stats_summary[metric]
                    print(f"\n{metric}:")
                    print(f"  Mean: {stats['mean']:.6f}")
                    print(f"  Std Dev: {stats['std']:.6f}")
                    print(f"  Variance: {stats['variance']:.6f}")

                logger.info("\n" + "=" * 100)
                logger.info("STATISTICAL SUMMARY:")
                for metric in metric_names:
                    stats = stats_summary[metric]
                    logger.info(f"{metric} - Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}, Var: {stats['variance']:.6f}")
                logger.info("=" * 100)


    # engine.visual_eval(args,model_args)

# %%

