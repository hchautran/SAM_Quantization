import torch
import torch.nn.functional  as F
import torch.nn as nn
import copy
from typing import List, Tuple, Union
from .ddp import DiffPruneRate

from training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData
from training.model.sam2 import SAM2Train
from sam2.modeling.backbones.image_encoder import ImageEncoder
from sam2.modeling.backbones.hieradet import MultiScaleAttention, MultiScaleBlock, Hiera, do_pool
from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)
from .sam2pruneduo import duplicate_batched_video_datapoint, separate_duplicated_output

class DuoDiffPruneRateMultiScaleAttention(MultiScaleAttention):
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent MultiScaleAttention class."""
        super().__init__(*args, **kwargs)
    def introduce_prune_diff(self,head_number,prune_granularity):
        self.prune_ddp = DiffPruneRate(head_number,prune_granularity)
    def set_processor(self, processor, module_name,args,train_rate_prune=False):
        self.processor = processor
        self.module_name = module_name 
        self.training = train_rate_prune
        self.threshold = args.threshold
        self.global_threshold = args.threshold_globle
        self.model_type = args.model_type
        self.prune_global = args.prune_global
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if self.training:
            B_2, H, W, _ = x.shape

            assert B_2 % 2 == 0
            B= B_2 // 2
            
            # qkv with shape (3, B, nHead, H * W, C)
            qkv = self.qkv(x).reshape(B_2, H * W, 3, self.num_heads, -1)
            
            qkv_non_prune, qkv_prune = qkv[:B], qkv[B:]
        
            with torch.no_grad():
                # q, k, v with shape (B * nHead, H * W, C)
                q, k, v = qkv_non_prune.permute(2, 0, 3, 1, 4).reshape(3, B * self.num_heads, H * W, -1).unbind(0)
                x = F.scaled_dot_product_attention(q, k, v)
            q_prune, k_prune, v_prune = qkv_prune.permute(2, 0, 3, 1, 4).reshape(3, B * self.num_heads, H * W, -1).unbind(0)
            x_prune= v_prune.mean(-2, keepdim=True).expand(-1, x.shape[-2], x.shape[-1])

            # Get sorted head indices from processor
            sorted_indicies = self.processor.final_entropy_stats.get(self.module_name, None)            
            # Create reordering indices for B * num_heads dimension 
            nu_heads = k.shape[0]
            nuheads_per_image= len(sorted_indicies)
            nu_images = nu_heads // nuheads_per_image


            # calculate number of batch of only one image
            sorted_indicies_mul_images = []
            for i in range(nu_images):
                # Add offset for each image in the batch
                offset = i * nuheads_per_image
                batch_indices = [idx + offset for idx in sorted_indicies]
                sorted_indicies_mul_images.extend(batch_indices)
            
            with torch.no_grad():
                sorted_x = x[sorted_indicies_mul_images, :, :]
            sorted_x_prune = x_prune[sorted_indicies_mul_images, :, :]

            # Get trainable mask from DiffPruneRate
            batch_masks_probability = []
            for i in range(nu_images):
                single_mask_probablity = self.prune_ddp.get_head_probability_diff_duo()
                batch_masks_probability.append(single_mask_probablity)
            prune_ordered_probality_mask = torch.cat(batch_masks_probability, dim=0).view(-1, 1, 1)
            sorted_x_prune = (1-prune_ordered_probality_mask) * sorted_x_prune + prune_ordered_probality_mask * sorted_x
            inverse_indices = torch.argsort(torch.tensor(sorted_indicies_mul_images))

            x = sorted_x[inverse_indices, :, :]
            x_prune = sorted_x_prune[inverse_indices, :, :]
            
            with torch.no_grad():
                x = x.reshape(B, self.num_heads, H * W, -1)
                x = x.transpose(1, 2)
                x = x.reshape(B, H, W, -1)
            x_prune = x_prune.reshape(B, self.num_heads, H * W, -1)
            x_prune = x_prune.transpose(1, 2)
            x_prune = x_prune.reshape(B, H, W, -1)
            x = torch.cat([x, x_prune], dim=0)
            x= self.proj(x)
            

            return x
            
        else:
            B, H, W, _ = x.shape

            # Compute QKV using SAM2 format
            # qkv with shape (B, H * W, 3, nHead, head_dim)
            qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
            # Permute to (3, B, nheads, H*W, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            # Reshape to (3, B*nheads, H*W, head_dim) for easier masking
            qkv = qkv.reshape(3, B * self.num_heads, H * W, -1)
            # Unbind to get q, k, v: each (B*nheads, H*W, head_dim)
            q, k, v = qkv.unbind(0)
            
            nu_images = q.shape[0] // len(self.processor.final_entropy_stats.get(self.module_name, None))

            should_prune = True
            
            if not self.prune_global:
                if self.model_type == "hiera_b_plus" and any(num in self.module_name for num in [".12.", ".20.", ".16."]):
                    should_prune = False
                
            if should_prune:
                single_mask_probability = self.prune_ddp.get_head_probability_diff_duo()

                if self.model_type == "hiera_b_plus":
                    if not any(num in self.module_name for num in [".12.", ".20.", ".16."]):
                        mask = single_mask_probability > self.threshold
                    else:
                        mask = single_mask_probability > self.global_threshold

                
                prune_kept_num = mask.sum().item()
                
                non_prune_mask = self.processor.final_entropy_stats.get(self.module_name, None)[:prune_kept_num]
                prune_mask = self.processor.final_entropy_stats.get(self.module_name, None)[prune_kept_num:]
                multi_non_prune_mask = copy.deepcopy(self.processor.final_entropy_stats.get(self.module_name, None)[:prune_kept_num])
                multi_prune_mask = copy.deepcopy(self.processor.final_entropy_stats.get(self.module_name, None)[prune_kept_num:])
                if nu_images >1:
                    for i in range(1,nu_images):
                        offset = i * len(self.processor.final_entropy_stats.get(self.module_name, None))
                        batch_non_prune_indices = [idx + offset for idx in non_prune_mask]
                        batch_prune_indices = [idx + offset for idx in prune_mask]
                        multi_non_prune_mask.extend(batch_non_prune_indices)
                        multi_prune_mask.extend(batch_prune_indices)


                
                q_attn = q[multi_non_prune_mask, :, :].unsqueeze(1)
                k_attn = k[multi_non_prune_mask, :, :].unsqueeze(1)
                v_attn = v[multi_non_prune_mask, :, :].unsqueeze(1)
                v_pruned = v[multi_prune_mask, :, :]
                
                # Compute attention for non-pruned heads
                x_attn= self.processor.scaled_dot_product_attention_m(q_attn, k_attn, v_attn)
                # x_attn = F.scaled_dot_product_attention(q_attn, k_attn, v_attn)
                x_attn = x_attn.reshape(-1, H * W, x_attn.size(-1))
                
                # Initialize output tensor
                x = torch.zeros_like(v).to(v.device)
                # Fill pruned heads with mean of their values
                x[multi_prune_mask] = v_pruned.mean(-2, keepdim=True).expand(-1, x_attn.shape[-2], x_attn.shape[-1])
                x[multi_non_prune_mask] = x_attn
            else:
                # No pruning or invalid masks
                x_attn= self.processor.scaled_dot_product_attention_m(q.unsqueeze(1),k.unsqueeze(1),v.unsqueeze(1))
                # x_attn = F.scaled_dot_product_attention(q.unsqueeze(1),k.unsqueeze(1),v.unsqueeze(1))
                x_attn = x_attn.reshape(-1, H * W, x_attn.size(-1))
                x = x_attn
         
            x = x.reshape(B, self.num_heads, H * W, -1)
            x = x.transpose(1, 2)
            x = x.reshape(B, H, W, -1)
            x = self.proj(x)
        
      
        return x

class DiffDuoPruneSAM2TrainDistillation(SAM2Train):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input: BatchedVideoDatapoint):
        double_input = duplicate_batched_video_datapoint(input)
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(double_input.flat_img_batch)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        _, prune_backbone_out = separate_duplicated_output(backbone_out)
        backbone_out = self.prepare_prompt_inputs(prune_backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)
        return previous_stages_out


def monkey_patch_train_sam2_diff_duo(model, processor=None, model_type=None,args= None, train=False):
    """
    Apply monkey-patching to SAM2 model for pruning and training.

    Args:
        model: SAM2 model to patch.
        processor: Processing strategy for activations.
        device: Device to use for training (default: "cuda").
        model_type
        train: Whether to enable training mode.
    """
    Match_head = {"hiera_b_plus": {".0.": 2048, ".1.": 2048, ".2.": 4096, ".3.": 4096, ".4.": 4096, ".5.": 8192, ".6.": 200, 
                           ".7.": 200, ".8.": 200, ".9.": 200, ".10.": 200, ".11.": 200, ".12.": 8, ".13.": 200, ".14.": 200, 
                           ".15.": 200, ".16.": 8, ".17.": 200, ".18.": 200, ".19.": 200, ".20.": 8, ".21.": 400, ".22.": 400, ".23.": 400} ,
               }
    
    if model_type not in Match_head:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Freeze all original SAM2 parameters
    for name, module in model.named_modules():
        if isinstance(module, MultiScaleAttention) and not module.q_pool:
            # Replace MultiScaleAttention with DiffPruneRateMultiScaleAttention
            module.__class__ = DuoDiffPruneRateMultiScaleAttention
            module.set_processor(processor, name, args, train)

            # Match the number of heads for the current layer
            matched_heads = None
            for key, num_heads in Match_head[model_type].items():
                if key in name:
                    matched_heads = num_heads
                    break

            if matched_heads is not None:
                module.introduce_prune_diff(matched_heads, 1)
            else:
                raise ValueError(f"Layer {name} does not have a matching head configuration in Match_head.")

        # if isinstance(module, MultiScaleBlock):
        #     # Replace MultiScaleBlock with DiffPruneRateMultiScaleBlock
        #     module.__class__ = DiffPruneRateMultiScaleBlock

        # if isinstance(module, Hiera):
        #     # Replace Hiera with DiffPruneRateHiera
        #     module.__class__ = DiffPruneRateHiera
        # if isinstance(module, ImageEncoder):
        #     module.__class__ = DiffPruneImageEncoder
        if train:
            if isinstance(module, SAM2Train):
                module.__class__ = DiffDuoPruneSAM2TrainDistillation

    # Enable gradients only for `selected_probability` parameters if training
    if train:
        for name, param in model.named_parameters():
            if "selected_probability" in name:
                param.requires_grad = True
                print(f"Enabled training for: {name}")
            else:
                param.requires_grad = False