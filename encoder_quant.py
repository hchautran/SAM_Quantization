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
from RTN_quantization.utils import replace_linear_with_quantized, QuantizationConfig
from utils import inference_image, to_numpy
from segment_anything.modeling.image_encoder import (
    window_partition,
    window_unpartition,
)





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
    debug = False

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
            if ImageEncoderViTObserver.debug:
                ImageEncoderViTObserver.attention_score[f"block_{idx}_attn"].append(to_numpy(attn))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_q"].append(to_numpy(q))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_k"].append(to_numpy(k))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_v"].append(to_numpy(v))

            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x, interm_embeddings

    @staticmethod
    def clear_dict():
        """Clear the attention score dictionary."""
        ImageEncoderViTObserver.attention_score = defaultdict(list)



def image_encoder_monkey_patch(
    model,
    processor=None,
    n_bits=8,
    weight_quant="per_channel",
    act_quant="per_token",
    debug=False
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
            ImageEncoderViTObserver.debug = debug 

    modules_to_exclude = [
        "pos_embed",
        "cls_token",
        "patch_embed",
        "neck",
        "fpn",
        "rel_pos_h",
        "rel_pos_w",
    ]

    config = QuantizationConfig(
        n_bits_w=n_bits,
        n_bits_a=n_bits,
        weight_quant=weight_quant,
        act_quant=act_quant,
        quantize_output=False,
    )
    replace_linear_with_quantized(
        module=model.image_encoder,
        config=config,
        module_name_to_exclude=modules_to_exclude,
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
        weight_quant="per_channel",
        act_quant="per_token",
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