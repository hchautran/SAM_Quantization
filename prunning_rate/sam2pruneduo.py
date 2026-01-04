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


class DuoPruneRateMultiScaleAttention(MultiScaleAttention):
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent MultiScaleAttention class."""
        super().__init__(*args, **kwargs)
    
    def introduce_full_attention_heads(self,head_number, train_rate_prune= False,initial_value=1.0, device='cpu', dtype=torch.float32):
        self.full_attention_heads = nn.Parameter(
                torch.ones(
                    head_number,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                * initial_value
            )
    
    def set_processor(self, processor, module_name, args,train_rate_prune=False):
        # self.processor = processor
        self.module_name = module_name 
        self.training = train_rate_prune
        self.batch_size = args.batch_size_train
        self.threshold = args.threshold  # You can set this to any desired value
        self.global_threshold = args.threshold_globle
        self.model_type = args.model_type
    
    
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

            nu_heads = k.shape[0]
            nuheads_per_image= len(self.full_attention_heads)
            nu_images = nu_heads // nuheads_per_image
            attention_weights_multiple_images = torch.cat([self.full_attention_heads for _ in range(nu_images)])
            
            attention_weights = (
                attention_weights_multiple_images.clamp(0, 1)
                .view(-1, 1, 1)  # Shape: (total_nu_heads, 1, 1)
            )
            
            x_prune = (1 - attention_weights) * x_prune + attention_weights * x


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

            # Apply pruning mask if available (same as SAM1)

            if hasattr(self, 'full_attention_heads') and self.full_attention_heads is not None:
                # Create pruning mask: True for heads to prune (below threshold)
                head_weights = self.full_attention_heads.clamp(0, 1)

                if self.model_type == "hiera_b_plus":
                    if not any(num in self.module_name for num in [".12.", ".20.", ".16."]):
                        prune_mask = head_weights < self.threshold
                    else:
                        prune_mask = head_weights < self.global_threshold
                
                # Repeat mask for batch dimension
                prune_mask = prune_mask.repeat(q.shape[0] // prune_mask.shape[0])
                # Select heads based on mask
                q_attn = q[~prune_mask, :, :].unsqueeze(1)
                k_attn = k[~prune_mask, :, :].unsqueeze(1)
                v_attn = v[~prune_mask, :, :].unsqueeze(1)
                v_pruned = v[prune_mask, :, :]
            else:
                q_attn, k_attn, v_attn = q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
                prune_mask = None
        
            x_attn = F.scaled_dot_product_attention(q_attn, k_attn, v_attn)
            # x_attn = self.scaled_dot_product_attention_m(q_attn, k_attn, v_attn)
            x_attn = x_attn.reshape(-1, H * W, x_attn.size(-1))

            # Merge outputs if pruning was applied
            if prune_mask is not None and isinstance(prune_mask, torch.Tensor):
                x = torch.zeros_like(v).to(v.device)
                # Fill pruned heads with mean of V
                x[prune_mask] = v_pruned.mean(-2, keepdim=True).expand(-1, x_attn.shape[-2], x_attn.shape[-1])
                # Fill kept heads with attention output
                x[~prune_mask] = x_attn
            else:
                x = x_attn

            x = x.reshape(B, self.num_heads, H * W, -1)
            x = x.transpose(1, 2)
            x = x.reshape(B, H, W, -1)

            # Apply output projection
            x = self.proj(x)

            return x




def duplicate_batched_video_datapoint(input: BatchedVideoDatapoint) -> BatchedVideoDatapoint:
    """
    Duplicates a BatchedVideoDatapoint instance by concatenating its data along the batch dimension.
    """
    # Duplicate tensors along the batch dimension (dim=1 for img_batch, masks, etc.)
    duplicated_img_batch = torch.cat([input.img_batch, input.img_batch], dim=1)
    duplicated_masks = torch.cat([input.masks, input.masks], dim=1)
    duplicated_obj_to_frame_idx = torch.cat([input.obj_to_frame_idx, input.obj_to_frame_idx], dim=1)

    # Duplicate metadata
    duplicated_metadata = BatchedVideoMetaData(
        unique_objects_identifier=torch.cat(
            [input.metadata.unique_objects_identifier, input.metadata.unique_objects_identifier], dim=1
        ),
        frame_orig_size=torch.cat(
            [input.metadata.frame_orig_size, input.metadata.frame_orig_size], dim=1
        ),
    )

    # Create a new BatchedVideoDatapoint with the duplicated data
    duplicated_input = BatchedVideoDatapoint(
        img_batch=duplicated_img_batch,
        obj_to_frame_idx=duplicated_obj_to_frame_idx,
        masks=duplicated_masks,
        metadata=duplicated_metadata,
        dict_key=input.dict_key,  # Keep the same dict_key
        batch_size=input.batch_size,  # Keep the same batch_size (number of frames)
    )

    return duplicated_input


def separate_duplicated_output(encoder_output):
    """
    Separate the output from ImageEncoder when input was duplicated (in1, in1).
    
    Args:
        encoder_output: Dictionary with keys "vision_features", "vision_pos_enc", "backbone_fpn"
    
    Returns:
        Tuple of two dictionaries, each with the same format as input
    """
    batch_size = encoder_output["vision_features"].shape[0]
    assert batch_size % 2 == 0, "Batch size must be even for duplicated input"
    
    mid_point = batch_size // 2
    
    # Split vision_features (single tensor)
    vision_features_1 = encoder_output["vision_features"][:mid_point]
    vision_features_2 = encoder_output["vision_features"][mid_point:]
    
    # Split vision_pos_enc (list of tensors)
    vision_pos_enc_1 = [pos[:mid_point] for pos in encoder_output["vision_pos_enc"]]
    vision_pos_enc_2 = [pos[mid_point:] for pos in encoder_output["vision_pos_enc"]]
    
    # Split backbone_fpn (list of tensors)
    backbone_fpn_1 = [feat[:mid_point] for feat in encoder_output["backbone_fpn"]]
    backbone_fpn_2 = [feat[mid_point:] for feat in encoder_output["backbone_fpn"]]
    
    output_1 = {
        "vision_features": vision_features_1,
        "vision_pos_enc": vision_pos_enc_1,
        "backbone_fpn": backbone_fpn_1,
    }
    
    output_2 = {
        "vision_features": vision_features_2,
        "vision_pos_enc": vision_pos_enc_2,
        "backbone_fpn": backbone_fpn_2,
    }
    
    return output_1, output_2

class DuoPruneSAM2TrainDistillation(SAM2Train):
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


def monkey_patch_train_sam2_duo(model, processor=None, model_type=None,args= None, train=False):
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
            module.__class__ = DuoPruneRateMultiScaleAttention
            module.set_processor(processor, name, args, train)

            # Match the number of heads for the current layer
            matched_heads = None
            for key, num_heads in Match_head[model_type].items():
                if key in name:
                    matched_heads = num_heads
                    break

            if matched_heads is not None:
                module.introduce_full_attention_heads(matched_heads, 1)
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
                module.__class__ = DuoPruneSAM2TrainDistillation

    # Enable gradients only for `selected_probability` parameters if training
    if train:
        for name, param in model.named_parameters():
            if "full_attention_heads" in name:
                param.requires_grad = True
                print(f"Enabled training for: {name}")
            else:
                param.requires_grad = False