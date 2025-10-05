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
  
        B, H, W, C = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        x = x.reshape(B, H*W, C)
        bias = self.qkv.bias[None, None, ...]
        x_mean = x.mean(1, keepdim=True)
        x_hat= (x - x_mean)
        # qkv     = self.qkv(x)       # shape: (B, H*W, 3*num_heads*dim)
        qkv_hat = self.qkv(x_hat)
        qkv_mean = self.qkv(x_mean) - bias


        qkv_hat = qkv_hat.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv_mean = qkv_mean.reshape(B, 1, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q_hat, k_hat, v_hat = qkv_hat.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        q_mean, _, v_mean = qkv_mean.reshape(3, B * self.num_heads, 1, -1).unbind(0)
        if ImageEncoderViTObserver.debug:
            qkv = qkv.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
            # Assert that q_ori = q + q_mean (and similarly for k, v)
            assert torch.allclose(q, q_hat+q_mean, rtol=1e-4, atol=1e-4), "q_ori != q + q_mean"
            # assert torch.allclose(k, k_hat+k_mean, rtol=1e-4, atol=1e-4), "k_ori != k + k_mean"
            assert torch.allclose(v, v_hat+v_mean, rtol=1e-4, atol=1e-4), "v_ori != v + v_mean"
            attn_ori = (q * self.scale) @ k.transpose(-2, -1)

        # q_hat = q_hat+q_mean
        # k_hat = k_hat+k_mean
        

        # Compute attention
        attn = (q_hat * self.scale) @ k_hat.transpose(-2, -1)
        attn_mean = (q_mean * self.scale) @ k_hat.transpose(-2, -1)
        attn = attn + attn_mean

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q_hat+q_mean, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )
            if ImageEncoderViTObserver.debug:
                attn_ori = add_decomposed_rel_pos(
                    attn_ori, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
                )

        attn = attn.softmax(dim=-1)
        if ImageEncoderViTObserver.debug:
            attn_ori = attn_ori.softmax(dim=-1)
        output = (
            (attn @ (v_hat+v_mean)).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        )
        output = self.proj(output)

        return output, attn, attn_mean, attn_ori, q_hat, k_hat, v_hat, q_mean, v_mean 

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

        # x, attn, q, k, v = self.attn(x)
        x, attn, attn_mean, q, k, v, q_mean, v_mean = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return  x, attn, attn_mean, q, k, v, q_mean, v_mean


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
            if ImageEncoderViTObserver.debug:
                ImageEncoderViTObserver.attention_score[f"block_{idx}_x"].append(to_numpy(x))
            x, attn, attn_mean, q, k, v, q_mean, v_mean = blk(x)

            # Store attention information
            if ImageEncoderViTObserver.debug:
                ImageEncoderViTObserver.attention_score[f"block_{idx}_attn"].append(to_numpy(attn))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_attn_mean"].append(to_numpy(attn_mean))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_q"].append(to_numpy(q))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_k"].append(to_numpy(k))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_v"].append(to_numpy(v))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_q_mean"].append(to_numpy(q_mean))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_v_mean"].append(to_numpy(v_mean))

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
    layers=[0,5, 17],
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
    layer_idx =0
    ImageEncoderViTObserver.debug = debug 
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.__class__ = AttentionObserver
            module.processor = processor
            module.n_bits = n_bits
            module.name = name
        if isinstance(module, Block):
            module.__class__ = BlockObserver
        if isinstance(module, ImageEncoderViT):
            if  layer_idx in layers:
                module.__class__ = ImageEncoderViTObserver
            layer_idx += 1

    modules_to_exclude = [
        "pos_embed",
        "cls_token",
        "patch_embed",
        "neck",
        "fpn",
        "rel_pos_h",
        "rel_pos_w",
    ]

    # config = QuantizationConfig(
    #     n_bits_w=n_bits,
    #     n_bits_a=n_bits,
    #     weight_quant=weight_quant,
    #     act_quant=act_quant,
    #     quantize_output=False,
    # )
    # replace_linear_with_quantized(
    #     module=model.image_encoder,
    #     config=config,
    #     module_name_to_exclude=modules_to_exclude,
    # )


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