import torch
import torch.nn.functional  as F
import copy
from typing import List, Tuple, Union
from .ddp import DiffPruneRate

from training.utils.data_utils import BatchedVideoDatapoint
from training.model.sam2 import SAM2Train
from sam2.modeling.backbones.image_encoder import ImageEncoder
from sam2.modeling.backbones.hieradet import MultiScaleAttention, MultiScaleBlock, Hiera, do_pool
from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)


class DiffPruneRateMultiScaleAttention(MultiScaleAttention):
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent MultiScaleAttention class."""
        super().__init__(*args, **kwargs)
    
    def introduce_prune_diff(self, head_number, prune_granularity):
        self.prune_ddp = DiffPruneRate(head_number, prune_granularity)
    
    def set_processor(self, processor, module_name, train_rate_prune=False):
        self.processor = processor
        self.module_name = module_name 
        self.training = train_rate_prune
    
    def _calculate_qkv_flops(self, B, H, W):
        """Calculate FLOPs for QKV linear transformation."""
        # Input: (B, H*W, dim), Weight: (dim, 3*dim_out)
        dim = self.qkv.in_features
        dim_out = self.dim_out
        return B * H * W * dim * 3 * dim_out
    
    def _calculate_attention_flops(self, H, W, active_heads):
        """Calculate FLOPs for scaled dot product attention with pruned heads."""
        head_dim = self.dim_out // self.num_heads
        seq_len = H * W
        
        # Scaled dot product attention FLOPs
        # Q @ K^T + scaling + softmax + @ V
        qk_flops = active_heads * seq_len * seq_len * head_dim
        softmax_flops = active_heads * seq_len * seq_len
        attn_v_flops = active_heads * seq_len * seq_len * head_dim
        
        return qk_flops + softmax_flops + attn_v_flops
    
    def _calculate_projection_flops(self, B, H, W):
        """Calculate FLOPs for final projection."""
        # Input: (B, H*W, dim_out), Weight: (dim_out, dim_out)
        return B * H * W * self.dim_out * self.dim_out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B, H, W, _ = x.shape
        
        # # qkv with shape (B, H * W, 3, nHead, C)
        # qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # qkv = qkv.reshape(3, B * self.num_heads, H * W, -1)
        # # Unbind to get q, k, v: each (B*nheads, H*W, head_dim)
        # q, k, v = qkv.unbind(0)


        if self.training:
            B, H, W, _ = x.shape
        
            # qkv with shape (B, H * W, 3, nHead, C)
            qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            qkv = qkv.reshape(3, B * self.num_heads, H * W, -1)
            # Unbind to get q, k, v: each (B*nheads, H*W, head_dim)
            q, k, v = qkv.unbind(0)


            prune_kept_num = self.prune_ddp.update_kept_head_number()
            # Get sorted head indices from processor
            sorted_indices = self.processor.final_entropy_stats.get(self.module_name, None)
            nu_heads = k.shape[0]
            nuheads_per_image= len(sorted_indices)
            nu_images = nu_heads // nuheads_per_image
            # calculate number of batch of only one image
            sorted_indicies_mul_images = []
            for i in range(nu_images):
                # Add offset for each image in the batch
                offset = i * nuheads_per_image
                batch_indices = [idx + offset for idx in sorted_indices]
                sorted_indicies_mul_images.extend(batch_indices)
            
            # Reorder q, k, v according to importance scores (head dimension is index 2)

            q_reordered = q[sorted_indicies_mul_images, :, :]
            k_reordered = k[sorted_indicies_mul_images, :, :]
            v_reordered = v[sorted_indicies_mul_images, :, :]
            # Get trainable mask from DiffPruneRate
            batch_masks = []
            for i in range(nu_images):
                single_mask = self.prune_ddp.get_head_mask(nuheads_per_image).to(q.device)
                batch_masks.append(single_mask)
            prune_mask = torch.cat(batch_masks, dim=0)
            # Apply mask to q and k (element-wise multiplication for training)
            q_masked = (q_reordered * prune_mask.unsqueeze(-1).unsqueeze(-1)).unsqueeze(1)
            k_masked = (k_reordered * prune_mask.unsqueeze(-1).unsqueeze(-1)).unsqueeze(1)
            v_masked = v_reordered.unsqueeze(1)  # v doesn't need masking for attention computation
            # Transpose for scaled_dot_product_attention: [B, nheads, H*W, C]
            x_attn = F.scaled_dot_product_attention(q_masked,k_masked,v_masked)
            x_attn = x_attn.reshape(-1, H * W, x_attn.size(-1))
            inverse_indices = torch.argsort(torch.tensor(sorted_indicies_mul_images))
            x = x_attn[inverse_indices, :, :]
            
        else:
            if self.processor is not None and hasattr(self.processor, 'final_entropy_stats'):
                if len(self.processor.final_entropy_stats) > 0:
                    return self.processor.process_val(x, self, self.module_name) 

            #########################################################
            B, H, W, _ = x.shape
        
            # qkv with shape (B, H * W, 3, nHead, C)
            qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            qkv = qkv.reshape(3, B * self.num_heads, H * W, -1)
            # Unbind to get q, k, v: each (B*nheads, H*W, head_dim)
            q, k, v = qkv.unbind(0)

            nu_images = q.shape[0] // len(self.processor.final_entropy_stats.get(self.module_name, None))
            prune_kept_num = int(self.prune_ddp.update_kept_head_number())
            
            should_prune = True
            if not self.processor.prune_global:
                if self.processor.model_type == "hiera_b_plus" and any(num in self.module_name for num in [".12", ".16", ".20"]):
                    should_prune = False
                
            
            if should_prune:
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
        # Calculate FLOPs
        qkv_flops = self._calculate_qkv_flops(B // nu_images, H, W)
        project_flops = self._calculate_projection_flops(B // nu_images, H, W)
        attention_flops = self._calculate_attention_flops(H, W, prune_kept_num)
        total_flops = qkv_flops + project_flops + attention_flops
        
        # print("module name:",self.module_name)
        # print("H,W:",H,W)
        # print("original_heads", self.prune_ddp.head_number)
        
        # print("number kept heads/ total heads:",prune_kept_num,"/",B*self.num_heads)
        # print("B// nu_images:",B//nu_images)
        # print("prune_kept_num:",prune_kept_num)
        # print("qkv flops:",qkv_flops)
        # print("projectflops flops:",project_flops)
        # print("attention_flops: ",attention_flops)
      
        return x, total_flops

class DiffPruneRateMultiScaleBlock(MultiScaleBlock):
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent Block class."""
        super().__init__(*args, **kwargs)

    def _calculate_mlp_flops(self, x: torch.Tensor) -> int:
        """Calculate FLOPs for the MLP block."""
        B, H, W, C = x.shape
        sequence_length = B * H * W

        # Initialize total FLOPs
        total_flops = 0

        # Iterate through each layer in the MLP
        input_dim = C
        for i, layer in enumerate(self.mlp.layers):
            output_dim = layer.out_features
            # FLOPs for a single linear layer: input_dim * output_dim * sequence_length
            total_flops += input_dim * output_dim * sequence_length
            input_dim = output_dim  # Update input_dim for the next layer

        return total_flops
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        if self.attn.q_pool:
            x = self.attn(x)
            flops= 0  
        else :
            x ,flops= self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, flops
class DiffPruneRateHiera(Hiera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add positional embedding
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        total_flops = 0
        for i, blk in enumerate(self.blocks):
            x, flops = blk(x)
            total_flops += flops
            if (i == self.stage_ends[-1]) or (
                i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
        return (outputs, total_flops)

class DiffPruneImageEncoder(ImageEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, sample: torch.Tensor):
        # Forward through backbone
        outputs, total_flops = self.trunk(sample)
        features, pos = self.neck(outputs)
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
            "total_flops": total_flops,
        }
        return output
    
class DiffPruneSAM2Train(SAM2Train):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input: BatchedVideoDatapoint):
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(input.flat_img_batch)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        total_flops = backbone_out['total_flops']
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)

        return previous_stages_out,total_flops
def monkey_patch_train_sam2(model, processor=None, model_type=None, train=False):
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
            module.__class__ = DiffPruneRateMultiScaleAttention
            module.set_processor(processor, name, train)

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

        if isinstance(module, MultiScaleBlock):
            # Replace MultiScaleBlock with DiffPruneRateMultiScaleBlock
            module.__class__ = DiffPruneRateMultiScaleBlock

        if isinstance(module, Hiera):
            # Replace Hiera with DiffPruneRateHiera
            module.__class__ = DiffPruneRateHiera
        if isinstance(module, ImageEncoder):
            module.__class__ = DiffPruneImageEncoder
        if isinstance(module, SAM2Train):
            module.__class__ = DiffPruneSAM2Train

    # Enable gradients only for `selected_probability` parameters if training
    if train:
        for name, param in model.named_parameters():
            if "selected_probability" in name:
                param.requires_grad = True
                print(f"Enabled training for: {name}")
            else:
                param.requires_grad = False