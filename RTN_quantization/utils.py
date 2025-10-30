import torch
import torch.nn as nn
from .Smooth import smooth_ln_fcs
from matplotlib import pyplot as plt
import os
import sys
import numpy as np
from dataclasses import dataclass
from segment_anything.modeling.image_encoder import Attention
from .per_tensor_channel_group import quantize_activation_low_high_density_activation_index, quantize_activation_per_token_absmax, quantize_weight_per_channel_absmax
# Add the sam-hq directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up to SAM_Quantization
sam_hq_path = os.path.join(project_root, "sam-hq")

sys.path.insert(0, sam_hq_path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Optional


class AttentionQ(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.percent=50
        self.n_bits_act=4
        self.n_bits_w= 4
    def take_values(self, percent, n_bits_act, n_bits_w):
        self.percent= percent
        self.n_bits_act= n_bits_act
        self.n_bits_w= n_bits_w

    def forward(self, x: torch.Tensor) :
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        # quantize attn and v
        
        attn = attn.softmax(dim=-1) # B * nHead, H * W, H * W
     
        attn, indices = quantize_activation_low_high_density_activation_index(attn, n_bits=self.n_bits_act,percent=self.percent, quantizehigh=True, )
        # import ipdb; ipdb.set_trace()
       
        v=quantize_activation_low_high_density_activation_index(v, n_bits=self.n_bits_act, percent=self.percent, quantizehigh=True, indices=indices)[0]
      
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x
class AttentionQ_token(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bits_act=4

    def take_values(self,  n_bits_act):
        self.n_bits_act= n_bits_act


    def forward(self, x: torch.Tensor) :
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        # quantize attn and v
        
        attn = attn.softmax(dim=-1) # B * nHead, H * W, H * W
        # import ipdb; ipdb.set_trace()
   
        attn = quantize_activation_per_token_absmax(attn, n_bits=self.n_bits_act)
       
        v = quantize_weight_per_channel_absmax(v.permute(0, 2, 1), n_bits=self.n_bits_act).permute(0, 2, 1)
      
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x
def replace_attention_with_quantized(model, channel = True ,percent=50, n_bits=4):
    device = next(model.parameters()).device
    for i, block in enumerate(model.image_encoder.blocks[:]):
        original_attn = block.attn
        dim = original_attn.qkv.in_features
        num_heads = original_attn.num_heads
        qkv_bias = hasattr(original_attn.qkv, 'bias') and original_attn.qkv.bias is not None
        use_rel_pos = original_attn.use_rel_pos
        h_size = (original_attn.rel_pos_h.shape[0] + 1) // 2
        w_size = (original_attn.rel_pos_w.shape[0] + 1) // 2
        input_size = (h_size, w_size)
        if channel:
            custom_attn = AttentionQ(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                input_size=input_size
            )
            custom_attn.take_values(percent, n_bits, n_bits)
        else:
            custom_attn = AttentionQ_token(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                input_size=input_size
            )
            custom_attn.take_values(n_bits)
        custom_attn.to(device)
        # Copy weights from original attention
        custom_attn.qkv.weight.data = original_attn.qkv.weight.data.to(device)
        custom_attn.qkv.bias.data = original_attn.qkv.bias.data.to(device)
        custom_attn.proj.weight.data = original_attn.proj.weight.data.to(device)
        custom_attn.proj.bias.data = original_attn.proj.bias.data.to(device)
        if use_rel_pos:
            custom_attn.rel_pos_h = original_attn.rel_pos_h
            custom_attn.rel_pos_w = original_attn.rel_pos_w
        block.attn = custom_attn
    import gc
    gc.collect()

@dataclass
class QuantizationConfig:
    """Configuration for quantization parameters.

    This class encapsulates all quantization-related parameters to avoid
    passing many individual arguments through function calls.

    Attributes:
        n_bits_w: Number of bits for weight quantization (default: 8)
        n_bits_a: Number of bits for activation quantization (default: 8)
        weight_quant: Weight quantization strategy ("per_channel", "per_tensor", "per_group")
        act_quant: Activation quantization strategy ("per_token", "per_tensor", "per_group", "low_high_density")
        quantize_output: Whether to quantize output activations (default: False)
        quantize_weight: Whether to quantize weights (default: True)
        group_size: Group size for group-based quantization (default: None)
        order: Channel reordering indices for selective quantization (default: None)
        topk: Top-k channels to preserve in selective quantization (default: None)
        quantizehigh: For density-based quantization, whether to quantize high-density tokens (default: True)
        up_down_RTN: Rounding strategy ("RTN", "up", "down", "random") (default: "RTN")
        percent: Percentage of channels/tokens to quantize (default: 100)
    """
    n_bits_w: int = 8
    n_bits_a: int = 8
    weight_quant: str = "per_channel"
    act_quant: str = "per_token"
    quantize_output: bool = False
    quantize_weight: bool = True
    group_size: Optional[int] = None
    order: Optional[torch.Tensor] = None
    topk: Optional[torch.Tensor] = None
    quantizehigh: bool = True
    up_down_RTN: str = "none"
    percent: float = 100

    def get_w8a8linear_class(self):
        """Determine the appropriate W8A8Linear subclass based on configuration.

        Returns:
            The appropriate W8A8Linear subclass to use
        """
        from .per_tensor_channel_group import (
            W8A8LinearPerChannel,
            W8A8LinearPerTensor,
            W8A8LinearPerGroup,
            W8A8LinearDensityBased,
            W8A8LinearSelectiveChannel
        )

        # Selective channel quantization (has order or topk)
        if self.order is not None or self.topk is not None:
            return W8A8LinearSelectiveChannel

        # Density-based quantization
        if self.act_quant == "low_high_density_activation":
            return W8A8LinearDensityBased

        # Group-based quantization
        if self.weight_quant == "per_group" or self.act_quant == "per_group":
            return W8A8LinearPerGroup

        # Per-tensor quantization
        if self.weight_quant == "per_tensor" and self.act_quant == "per_tensor":
            return W8A8LinearPerTensor

        # Default: Per-channel weight, per-token activation
        return W8A8LinearPerChannel

    def to_kwargs(self) -> dict:

        return {
            'n_bits_w': self.n_bits_w,
            'n_bits_a': self.n_bits_a,
            'weight_quant': self.weight_quant,
            'act_quant': self.act_quant,
            'quantize_output': self.quantize_output,
            'group_size': self.group_size,
            'quantize_weight': self.quantize_weight,
            'order': self.order,
            'topk': self.topk,
            'quantizehigh': self.quantizehigh,
            'up_down_RTN': self.up_down_RTN,
            'percent': self.percent
        }


def replace_linear_with_quantized(
    module: nn.Module,
    config: QuantizationConfig,
    module_name_to_exclude: Optional[list] = None,
    parent_name: str = ""
):
    """Replace linear layers with quantized versions using configuration object.

    Args:
        module: The module to traverse
        config: QuantizationConfig object containing all quantization parameters
        module_name_to_exclude: List of module names to exclude from quantization
        parent_name: The hierarchical name of the parent module (used internally)

    Example:
        >>> config = QuantizationConfig(
        ...     n_bits_w=4,
        ...     n_bits_a=8,
        ...     weight_quant="per_channel",
        ...     act_quant="per_token"
        ... )
        >>> replace_linear_with_quantized(model, config, ["lm_head"])
    """
    if module_name_to_exclude is None:
        module_name_to_exclude = []

    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
           any([x == name for x in module_name_to_exclude]) and not \
           any([x in parent_name for x in module_name_to_exclude]):

            quantized_class = config.get_w8a8linear_class()
            new_module = quantized_class.from_float(child, **config.to_kwargs())
            setattr(module, name, new_module)

        else:
            # Recursively process child modules
            name_layer = parent_name + "." + name if parent_name else name
            replace_linear_with_quantized(
                child, config, module_name_to_exclude, name_layer
            )




    

@torch.no_grad()
def smooth_sam(model, act_scales, do_smooth_decoder=True, alpha=0.5, num_samples=512, do_smooth_attn_encoder= True):
    """
    Apply SmoothQuant to SAM model focusing on image encoder and prompt encoder.
    
    Args:
        model: SAM model instance
        batched_input: List of input dictionaries for calibration
        alpha: Smoothing balance factor (0.0 to 1.0)
        num_samples: Number of samples for calibration
    """
    if hasattr(model, 'transformer'):
        print("applying smooth to the decoder")
        transformer = model.transformer
        for i, layer in enumerate(transformer.layers):  # Ensure this aligns correctly
            layer_name = f"mask_decoder.transformer.layers.{i}"
            mlp_name = f"{layer_name}.mlp.lin1"
            if mlp_name in act_scales:
                smooth_ln_fcs(layer.norm2, [layer.mlp.lin1], act_scales[mlp_name], alpha)
                print(f"Applied MLP smoothing (norm2 -> mlp.lin1) to {layer_name}")
    
    # Target image encoder blocks specifically (based on SAM structure)
    if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'blocks'):
        for i, block in enumerate(model.image_encoder.blocks):
            block_name = f"image_encoder.blocks.{i}"
            if do_smooth_attn_encoder:
                # Attention smoothing: norm1 -> attn.qkv
                if hasattr(block, 'norm1') and hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                    qkv_name = f"{block_name}.attn.qkv"
                    if qkv_name in act_scales:
                        qkv_input_scales = act_scales[qkv_name]
                        smooth_ln_fcs(block.norm1, [block.attn.qkv], qkv_input_scales, alpha)
                        print(f"Applied attention smoothing to {block_name}")

            if hasattr(block, 'norm2') and hasattr(block, 'mlp'):
                mlp_first_layer = None
                mlp_name = None
                for attr_name in ['fc1', 'lin1', 'dense', 'linear']:
                    if hasattr(block.mlp, attr_name):
                        mlp_first_layer = getattr(block.mlp, attr_name)
                        mlp_name = f"{block_name}.mlp.{attr_name}"
                        break      
                if mlp_first_layer is not None and mlp_name in act_scales:
                    mlp_input_scales = act_scales[mlp_name]
                    smooth_ln_fcs(block.norm2, [mlp_first_layer], mlp_input_scales, alpha)
                    print(f"Applied MLP smoothing to {block_name}")

    
    # Handle prompt encoder layers
    if hasattr(model, 'prompt_encoder'):
        for name, module in model.prompt_encoder.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"prompt_encoder.{name}"
                if full_name in act_scales:
                    parent_path = name.split('.')[:-1]
                    parent_module = model.prompt_encoder
                    for part in parent_path:
                        parent_module = getattr(parent_module, part)
                    for child_name, child_module in parent_module.named_children():
                        if isinstance(child_module, nn.LayerNorm):
                            linear_input_scales = act_scales[full_name]
                            smooth_ln_fcs(child_module, [module], linear_input_scales, alpha)
                            print(f"Applied prompt encoder smoothing: {full_name}")
                            break
                            
    # Handle mask decoder layers
    
    if do_smooth_decoder and  hasattr(model, 'mask_decoder'):
        print("Processing mask decoder layers...")
        
        # Handle transformer layers in mask decoder
        if hasattr(model.mask_decoder, 'transformer'):
            transformer = model.mask_decoder.transformer
        
            for i, layer in enumerate(transformer.layers):
                layer_name = f"mask_decoder.transformer.layers.{i}"
                ## TODO: Find the way to smooth residual layers  
                # norm1 -> cross_attn_token_to_image
                # if hasattr(layer, 'norm1') and hasattr(layer, 'cross_attn_token_to_image'):
                #     cross_q_name = f"{layer_name}.cross_attn_token_to_image.q_proj"
                #     if cross_q_name in act_scales:
                #         cross_layers = [
                #             layer.cross_attn_token_to_image.q_proj,
                #             layer.cross_attn_token_to_image.k_proj,
                #             layer.cross_attn_token_to_image.v_proj
                #         ]
                #         smooth_ln_fcs(layer.norm1, cross_layers, act_scales[cross_q_name], alpha)
                #         print(f"Applied smoothing (norm1 -> cross_attn_token_to_image) to {layer_name}")
                
                # norm2 -> mlp.lin1
                if hasattr(layer, 'norm2') and hasattr(layer, 'mlp') and hasattr(layer.mlp, 'lin1'):
                    mlp_name = f"{layer_name}.mlp.lin1"
                    if mlp_name in act_scales:
                        smooth_ln_fcs(layer.norm2, [layer.mlp.lin1], act_scales[mlp_name], alpha)
                        print(f"Applied MLP smoothing (norm2 -> mlp.lin1) to {layer_name}")
                
                # norm3 -> cross_attn_image_to_token
                # if hasattr(layer, 'norm3') and hasattr(layer, 'cross_attn_image_to_token'):
                #     img_to_token_q_name = f"{layer_name}.cross_attn_image_to_token.q_proj"
                #     if img_to_token_q_name in act_scales:
                #         img_to_token_layers = [
                #             layer.cross_attn_image_to_token.q_proj,
                #             layer.cross_attn_image_to_token.k_proj,
                #             layer.cross_attn_image_to_token.v_proj
                #         ]
                #         smooth_ln_fcs(layer.norm3, img_to_token_layers, act_scales[img_to_token_q_name], alpha)
                #         print(f"Applied smoothing (norm3 -> cross_attn_image_to_token) to {layer_name}")
                        
                # norm4 of final layer -> final_attn_token_to_image
                # Check if this is the last layer (i == len(transformer.layers) - 1)
                # if i == len(transformer.layers) - 1:
                #     if hasattr(layer, 'norm4') and hasattr(transformer, 'final_attn_token_to_image'):
                #         final_attn_name = "mask_decoder.transformer.final_attn_token_to_image.q_proj"
                #         if final_attn_name in act_scales:
                #             final_attn_layers = [
                #                 transformer.final_attn_token_to_image.q_proj,
                #                 transformer.final_attn_token_to_image.k_proj,
                #                 transformer.final_attn_token_to_image.v_proj
                #             ]
                #             smooth_ln_fcs(layer.norm4, final_attn_layers, act_scales[final_attn_name], alpha)
                #             print(f"Applied final attention smoothing (norm4 of final layer -> final_attn_token_to_image)")
                                    
    print("SmoothQuant smoothing completed for SAM model")
    return model


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
        
        print(f"Score: {score:.3f}")
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
    


