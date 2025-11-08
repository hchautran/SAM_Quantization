# Standard library imports
import math
from collections import defaultdict
from typing import Optional, Tuple

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch import Tensor
# SAM model imports
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling.transformer import (
    Attention,
    TwoWayAttentionBlock,
    TwoWayTransformer,
)

# Local imports
from .base import AttentionProcessor
from .encoder import (
    EncoderAttentionProcessorSmoothMeanQ,
    EncoderAttentionProcessorSmooth,
    EncoderAttentionProcessorHighLow,
)
from RTN_quantization import per_tensor_channel_group
from RTN_quantization.utils import QuantizationConfig, replace_linear_with_quantized
from utils.utils import inference_image

from utils.utils import quantize_activation_per_highblock_abmax, find_O_qha

# ============================================================================
# Utility Functions
# ============================================================================


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w



@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    # print('got here', n_bits)
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

def separate_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    Separate tensor into multiple attention heads.

    Args:
        x: Input tensor of shape (B, N_tokens, C)
        num_heads: Number of attention heads

    Returns:
        Tensor of shape (B, N_heads, N_tokens, C_per_head)
    """
    b, n, c = x.shape
    x = x.reshape(b, n, num_heads, c // num_heads)
    return x.transpose(1, 2)


def recombine_heads(x: torch.Tensor) -> torch.Tensor:
    """
    Recombine multiple attention heads back into single tensor.

    Args:
        x: Input tensor of shape (B, N_heads, N_tokens, C_per_head)

    Returns:
        Tensor of shape (B, N_tokens, C)
    """
    b, n_heads, n_tokens, c_per_head = x.shape
    x = x.transpose(1, 2)
    return x.reshape(b, n_tokens, n_heads * c_per_head)


def re_cal_attn(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Recalculate attention scores from query and key tensors.

    Args:
        q: Query tensor of shape (B, N_heads, N_tokens, C_per_head)
        k: Key tensor of shape (B, N_heads, N_tokens, C_per_head)

    Returns:
        Attention scores of shape (B, N_heads, N_tokens, N_tokens)
    """
    _, _, _, c_per_head = q.shape
    attn = q @ k.permute(0, 1, 3, 2)
    attn = attn / math.sqrt(c_per_head)
    attn = torch.softmax(attn, dim=-1)
    return attn


# ============================================================================
# Observer Classes
# ============================================================================


class TwoWayTransformerObserverElementLow(TwoWayTransformer):
    """
    Observer wrapper for TwoWayTransformer that tracks attention scores.

    This class extends TwoWayTransformer to capture and store attention scores,
    queries, keys, and values from all attention layers for analysis.
    """

    attention_score = defaultdict(list)
    weights = {}  # Store weights separately, extracted once
    debug = False

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            (queries,
             keys , 
             p2p_O,  
             p2p_O_qha, 
             p2p_O_qfull, 
             p2i_O,  
             p2i_O_qha, 
             p2i_O_qfull, 
             i2p_O,  
             i2p_O_qha, 
             i2p_O_qfull )= layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
            if TwoWayTransformerObserverElementLow.debug:
                
                # store all the value
                TwoWayTransformerObserverElementLow.attention_score["p2p_O"].append(p2p_O)
                TwoWayTransformerObserverElementLow.attention_score["p2p_O_qha"].append(p2p_O_qha)
                TwoWayTransformerObserverElementLow.attention_score["p2p_O_qfull"].append(p2p_O_qfull)
                TwoWayTransformerObserverElementLow.attention_score["p2i_O"].append(p2i_O)
                TwoWayTransformerObserverElementLow.attention_score["p2i_O_qha"].append(p2i_O_qha)
                TwoWayTransformerObserverElementLow.attention_score["p2i_O_qfull"].append(p2i_O_qfull)
                TwoWayTransformerObserverElementLow.attention_score["i2p_O"].append(i2p_O)
                TwoWayTransformerObserverElementLow.attention_score["i2p_O_qha"].append(i2p_O_qha)
                TwoWayTransformerObserverElementLow.attention_score["i2p_O_qfull"].append(i2p_O_qfull)
                
                

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out, final_O,  final_O_qha, final_O_qfull= self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        if TwoWayTransformerObserverElementLow.debug:
            TwoWayTransformerObserverElementLow.attention_score["final_O"].append(final_O)
            TwoWayTransformerObserverElementLow.attention_score["final_O_qha"].append(final_O_qha)
            TwoWayTransformerObserverElementLow.attention_score["final_O_qfull"].append(final_O_qfull)
        return queries, keys

    

    def clear_dict():
        TwoWayTransformerObserverElementLow.attention_score = defaultdict(list)
        TwoWayTransformerObserverElementLow.weights = {}


class TwoWayAttentionBlockObserverElementLow(TwoWayAttentionBlock):
    """
    Observer wrapper for TwoWayAttentionBlock that captures attention metrics.

    Extends TwoWayAttentionBlock to return attention scores and intermediate
    values for debugging and analysis purposes.
    """

    attention_dict = {}

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries, p2p_O,  p2p_O_qha, p2p_O_qfull = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out , p2p_O,  p2p_O_qha, p2p_O_qfull = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out , p2i_O,  p2i_O_qha, p2i_O_qfull = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out , i2p_O,  i2p_O_qha, i2p_O_qfull= self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return (queries, keys , p2p_O,  p2p_O_qha, p2p_O_qfull, p2i_O,  p2i_O_qha, p2i_O_qfull, i2p_O,  i2p_O_qha, i2p_O_qfull)

class AttentionObserverElementLow(Attention):
    """
    Attention layer with quantization support and activation tracking.

    This class extends the standard Attention layer to support:
    - Quantization of activations
    - Optional processing strategies for Q, K, V
    - Tracking and returning attention scores
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.high_element = True  # Default to high element tracking
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        # take the high output
        attn_qha , indicies = quantize_activation_per_highblock_abmax(attn, n_bits=4, percent=0.5, block_size= 1, high= self.high_element) 
        O_qha = find_O_qha(qattn=attn_qha, v=v, indices = indicies, n_bits=4, block_size=1).unsqueeze(0)
        # Get output
        O = attn @ v
        
        attn = quantize_activation_per_token_absmax(attn, n_bits=4)
        v = quantize_weight_per_channel_absmax(v.permute(0,1, 3, 2), n_bits=4).permute(0, 1,3, 2)
        O_qfull = attn @ v
        
        out = self._recombine_heads(O_qha)
        out = self.out_proj(out)

        return out, O,  O_qha, O_qfull


# ============================================================================
# Quantization Functions
# ============================================================================




def mask_decoder_monkey_patch(
    model,
    processor: AttentionProcessor = None,
    n_bits=8,
    weight_quant="per_channel",
    k_preserve=0,
    debug=False,
    high_element: bool = True,
):
    """
    Apply monkey-patching to SAM mask decoder for quantization and observation.

    Args:
        model: SAM model to patch
        processor: Processing strategy for activations
        n_bits: Number of bits for quantization
        weight_quant: Weight quantization strategy
        k_preserve: Number of channels to preserve
    """
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.__class__ = AttentionObserverElementLow
            module.high_element = high_element
        if isinstance(module, TwoWayAttentionBlock):
            module.__class__ = TwoWayAttentionBlockObserverElementLow
        if isinstance(module, TwoWayTransformer):
            module.__class__ = TwoWayTransformerObserverElementLow
            TwoWayTransformerObserverElementLow.debug = debug 
            

    modules_to_exclude = [
        "encoder",
    ]

    config = QuantizationConfig(
        n_bits_w=n_bits,
        n_bits_a=n_bits,
        weight_quant=weight_quant,
        act_quant="per_token",
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
    processor = None,
    mask_decoder_monkey_patch(
        predictor.model,
        processor,
        n_bits=4,
        weight_quant="selective_channel",
        k_preserve=4,
        debug= True,
    )

    

    # Run inference
    results = inference_image(
        predictor,
        image_dir="./input_imgs/",
        example_idx=3,
        show_image=True,
    )
    print("\nCaptured attention information:")
    for key in TwoWayTransformerObserverElementLow.attention_score.keys():
        print(f"  {key}: {len(TwoWayTransformerObserverElementLow.attention_score[key])} items")