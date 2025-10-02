# Standard library imports
from collections import defaultdict
from typing import Optional, Tuple

# Third-party imports
import torch
import torch.nn as nn

# SAM model imports
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling.image_encoder import (
    Attention,
    Block,
    ImageEncoderViT,
)

# Local imports
from quant_utils import (
    ImageEncoderProcessor,
    quantize_activation_per_token_absmax,
)
from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
from RTN_quantization import per_tensor_channel_group
from RTN_quantization.utils import  replace_linear_with_target_and_quantize
from utils import inference_image, to_numpy
from segment_anything.modeling.image_encoder import (
    window_partition,
    window_unpartition,
)



# ============================================================================
# Utility Functions
# ============================================================================



def re_cal_attn(q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Recalculate attention scores from query and key tensors.

    Args:
        q: Query tensor of shape (B, N_heads, H*W, C_per_head) or (B*N_heads, H*W, C_per_head)
        k: Key tensor of shape (B, N_heads, H*W, C_per_head) or (B*N_heads, H*W, C_per_head)
        scale: Scaling factor for attention

    Returns:
        Attention scores of shape (B, N_heads, H*W, H*W) or (B*N_heads, H*W, H*W)
    """
    attn = q @ k.transpose(-2, -1)
    attn = attn * scale
    attn = torch.softmax(attn, dim=-1)
    return attn


# ============================================================================
# Observer Classes
# ============================================================================


class AttentionObserver(Attention):
    """
    Attention layer observer for SAM encoder with quantization support and activation tracking.

    This class extends the standard Attention layer from SAM encoder to support:
    - Quantization of activations using ImageEncoderProcessor
    - Tracking and returning attention scores
    - Compatible with ViT-based image encoder
    """

    attention_score = defaultdict(list)

    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent Attention class."""
        super().__init__(*args, **kwargs)
        # Quantization attributes
        self.n_bits = 8
        self.name = ""
        self.processor = None

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention tracking and optional quantization.

        Args:
            x: Input tensor of shape (B, H, W, C)

        Returns:
            Tuple of (output, attn, q, k, v)
        """
        B, H, W, C = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C_per_head)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        # Apply processor if available (skip if processor doesn't actually modify tensors)
        if self.processor is not None and hasattr(self.processor, 'stat') and self.name in self.processor.stat:
            # Only do expensive reshaping if we actually have calibration stats for this layer
            # Reshape to per-head format for processor
            c_per_head = C // self.num_heads
            breakpoint()
            q_heads = q.reshape(B, self.num_heads, H * W, c_per_head)
            k_heads = k.reshape(B, self.num_heads, H * W, c_per_head)
            v_heads = v.reshape(B, self.num_heads, H * W, c_per_head)

            q_heads, k_heads, v_heads = self.processor.process(
                q_heads, k_heads, v_heads, self.name, self.n_bits
            )

            # Reshape back
            q = q_heads.reshape(B * self.num_heads, H * W, c_per_head)
            k = k_heads.reshape(B * self.num_heads, H * W, c_per_head)
            v = v_heads.reshape(B * self.num_heads, H * W, c_per_head)

        # Compute attention
        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)
        output = (
            (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        )
        output = self.proj(output)

        return output, attn, q, k, v

    @staticmethod
    def clear_dict():
        """Clear the attention score dictionary."""
        AttentionObserver.attention_score = defaultdict(list)


class BlockObserver(Block):
    """
    Observer wrapper for SAM encoder Block (Transformer blocks).

    Extends Block to return attention scores and intermediate values
    for debugging and analysis purposes.
    """

    attention_dict = {}

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention tracking.

        Args:
            x: Input tensor of shape (B, H, W, C)

        Returns:
            Tuple of (output, attn, q, k, v)
        """

        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x, attn, q, k, v = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x, attn, q, k, v


class ImageEncoderViTObserver(ImageEncoderViT):
    """
    Observer wrapper for ImageEncoderViT that tracks attention scores.

    This class extends ImageEncoderViT to capture and store attention scores,
    queries, keys, and values from all attention layers for analysis.
    """

    attention_score = defaultdict(list)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Forward pass with attention tracking.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tuple of (output, interm_embeddings)
        """
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        interm_embeddings = []
        for idx, blk in enumerate(self.blocks):
            x, attn, q, k, v = blk(x)

            # Store attention information
            # ImageEncoderViTObserver.attention_score[f"block_{idx}_attn"].append(to_numpy(attn))
            # ImageEncoderViTObserver.attention_score[f"block_{idx}_q"].append(to_numpy(q))
            # ImageEncoderViTObserver.attention_score[f"block_{idx}_k"].append(to_numpy(k))
            # ImageEncoderViTObserver.attention_score[f"block_{idx}_v"].append(to_numpy(v))

            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x, interm_embeddings

    @staticmethod
    def clear_dict():
        """Clear the attention score dictionary."""
        ImageEncoderViTObserver.attention_score = defaultdict(list)


# ============================================================================
# Quantization Functions
# ============================================================================


# def replace_linear_with_target_and_quantize(
#     module,
#     target_class,
#     n_bit_w,
#     n_bit_ac,
#     module_name_to_exclude,
#     weight_quant="per_channel",
#     act_quant="per_token",
#     quantize_output=False,
#     group_size=None,
#     quantize_weight=True,
#     k_preserve=None,
# ):
#     """
#     Replace linear layers in attention modules with quantized versions.

#     Args:
#         module: Module to process
#         target_class: Target quantized linear class
#         n_bit_w: Weight quantization bits
#         n_bit_ac: Activation quantization bits
#         module_name_to_exclude: List of module names to skip
#         weight_quant: Weight quantization strategy
#         act_quant: Activation quantization strategy
#         quantize_output: Whether to quantize output
#         group_size: Group size for quantization
#         quantize_weight: Whether to quantize weights
#         k_preserve: Number of channels to preserve in selective quantization
#     """

#     def _process_module_recursive(current_module, current_path=""):
#         """Recursively process modules."""
#         for name, child in current_module.named_children():
#             # Build full path including parent names
#             full_path = f"{current_path}.{name}" if current_path else name

#             if isinstance(child, AttentionObserver):
#                 # Get order statistics if using selective quantization
#                 order = None
#                 topk = None

#                 has_processor_stat = (
#                     hasattr(child, 'processor') and
#                     child.processor is not None and
#                     hasattr(child.processor, 'stat')
#                 )

#                 if has_processor_stat and weight_quant == 'selective_channel':
#                     stat_data = None
#                     for stat_key in child.processor.stat.keys():
#                         if stat_key in full_path or full_path in stat_key:
#                             stat_data = child.processor.stat[stat_key]
#                             print(f"Matched statistics: {stat_key} -> {full_path}")
#                             break

#                     if stat_data and 'order' in stat_data:
#                         order = stat_data['order']
#                         if k_preserve is not None and k_preserve > 0:
#                             topk = list(range(min(k_preserve, order.size(-1))))
#                         print(f"Found order statistics for {full_path}, order shape: {order.shape}, topk={topk}")

#                 # Process linear layers within attention modules
#                 for linear_name, linear_module in child.named_children():
#                     if isinstance(linear_module, nn.Linear) and linear_name not in module_name_to_exclude:
#                         actual_weight_quant = weight_quant
#                         actual_order = None
#                         actual_topk = None

#                         # Apply selective quantization to QKV projection
#                         if weight_quant == 'selective_channel' and order is not None and linear_name == 'qkv':
#                             actual_weight_quant = 'selective_channel'
#                             actual_order = order
#                             actual_topk = topk
#                             print(f"Applying selective quantization to {full_path}.{linear_name}")

#                         print(f"Processing module: {full_path}.{linear_name}")

#                         new_module = target_class.from_float(
#                             linear_module,
#                             n_bits_w=n_bit_w,
#                             n_bits_ac=n_bit_ac,
#                             weight_quant=actual_weight_quant,
#                             act_quant=act_quant,
#                             quantize_output=quantize_output,
#                             group_size=group_size,
#                             quantize_weight=quantize_weight,
#                             order=actual_order,
#                             topk=actual_topk,
#                         )
#                         setattr(child, linear_name, new_module)

#             else:
#                 # Recursively process nested modules
#                 _process_module_recursive(child, full_path)

#     # Start recursive processing
#     _process_module_recursive(module)


def image_encoder_monkey_patch(
    model,
    processor=None,
    n_bits=8,
    weight_quant="per_channel",
    k_preserve=0,
):
    """
    Apply monkey-patching to SAM image encoder for quantization and observation.

    Args:
        model: SAM model to patch
        processor: Processing strategy for activations
        n_bits: Number of bits for quantization
        weight_quant: Weight quantization strategy
        k_preserve: Number of channels to preserve in selective quantization
    """
    # Replace classes with observer versions using monkey patching
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.__class__ = AttentionObserver
            module.processor = processor
            module.n_bits = n_bits
            module.name = name
        if isinstance(module, Block):
            module.__class__ = BlockObserver
        if isinstance(module, ImageEncoderViT):
            module.__class__ = ImageEncoderViTObserver

    modules_to_exclude = [
        "pos_embed",
        "cls_token",
        "patch_embed",
        "neck",
        "fpn",
        "rel_pos_h",
        "rel_pos_w",
    ]

    replace_linear_with_target_and_quantize(
        module=model.image_encoder,
        parent_name="",
        target_class=per_tensor_channel_group.W8A8Linear,
        n_bit_w=n_bits,
        n_bit_ac=n_bits,
        module_name_to_exclude=modules_to_exclude,
        weight_quant=weight_quant,
        act_quant="per_token",
        quantize_output=False,
    )


# ============================================================================
# Main Execution
# ============================================================================


if __name__ == "__main__":
    # Configuration
    model_type = "vit_l"
    num_calib_samples = 8
    checkpoint_path = "./pretrained_checkpoint/sam_hq_vit_l.pth"

    # Initialize model and predictor
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to("cuda")
    predictor = SamPredictor(sam)

    # Setup processor with calibration for image encoder
    processor = ImageEncoderProcessor("encoder_attn")
    processor.calibrate(
        predictor=predictor,
        modules=(Block,),
        num_samples=num_calib_samples,
    )

    # Apply quantization with monkey-patching
    image_encoder_monkey_patch(
        predictor.model,
        processor=processor,
        n_bits=4,
        weight_quant="selective_channel",
        k_preserve=4,
    )

    # Run inference
    results = inference_image(
        predictor,
        image_dir="./input_imgs/",
        example_idx=3,
        show_image=True,
    )

    # Access attention scores
    print("\nCaptured attention information:")
    for key in ImageEncoderViTObserver.attention_score.keys():
        print(f"  {key}: {len(ImageEncoderViTObserver.attention_score[key])} items")

    print("\nImage encoder quantization completed successfully!")