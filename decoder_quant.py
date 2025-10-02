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

# SAM model imports
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling.transformer import (
    Attention,
    TwoWayAttentionBlock,
    TwoWayTransformer,
)

# Local imports
from quant_utils import (
    AttnBasedProcessor,
    DoNothingProcessor,
    ProcessStrategy,
    quantize_activation_per_token_absmax,
)
from RTN_quantization import per_tensor_channel_group
from RTN_quantization.utils import QuantizationConfig
from utils import inference_image



# ============================================================================
# Utility Functions
# ============================================================================




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


class TwoWayTransformerObserver(TwoWayTransformer):
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
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor,
    ):
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            TwoWayTransformerObserver.attention_score["pre_p"].append(queries)
            TwoWayTransformerObserver.attention_score["pre_i"].append(keys)
            (
                queries,
                keys,
                p2p_attn,
                p2p_q,
                p2p_k,
                p2p_v,
                p2p_q_pre,
                p2p_k_pre,
                p2p_v_pre,
                p2i_attn,
                p2i_q,
                p2i_k,
                p2i_v,
                p2i_q_pre,
                p2i_k_pre,
                p2i_v_pre,
                i2p_attn,
                i2p_q,
                i2p_k,
                i2p_v,
                i2p_q_pre,
                i2p_k_pre,
                i2p_v_pre,
            ) = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

            # Post-projection activations
            if TwoWayTransformerObserver.debug:
                TwoWayTransformerObserver.attention_score["p2p_q"].append(p2p_q)
                TwoWayTransformerObserver.attention_score["p2p_k"].append(p2p_k)
                TwoWayTransformerObserver.attention_score["p2p_v"].append(p2p_v)

                TwoWayTransformerObserver.attention_score["i2p_q"].append(i2p_q)
                TwoWayTransformerObserver.attention_score["i2p_k"].append(i2p_k)
                TwoWayTransformerObserver.attention_score["i2p_v"].append(i2p_v)

                TwoWayTransformerObserver.attention_score["p2i_q"].append(p2i_q)
                TwoWayTransformerObserver.attention_score["p2i_k"].append(p2i_k)
                TwoWayTransformerObserver.attention_score["p2i_v"].append(p2i_v)

                # Pre-projection activations
                TwoWayTransformerObserver.attention_score["p2p_q_pre"].append(p2p_q_pre)
                TwoWayTransformerObserver.attention_score["p2p_k_pre"].append(p2p_k_pre)
                TwoWayTransformerObserver.attention_score["p2p_v_pre"].append(p2p_v_pre)

                TwoWayTransformerObserver.attention_score["i2p_q_pre"].append(i2p_q_pre)
                TwoWayTransformerObserver.attention_score["i2p_k_pre"].append(i2p_k_pre)
                TwoWayTransformerObserver.attention_score["i2p_v_pre"].append(i2p_v_pre)

                TwoWayTransformerObserver.attention_score["p2i_q_pre"].append(p2i_q_pre)
                TwoWayTransformerObserver.attention_score["p2i_k_pre"].append(p2i_k_pre)
                TwoWayTransformerObserver.attention_score["p2i_v_pre"].append(p2i_v_pre)

                TwoWayTransformerObserver.attention_score["p2p_attn"].append(p2p_attn)
                TwoWayTransformerObserver.attention_score["i2p_attn"].append(i2p_attn)
                TwoWayTransformerObserver.attention_score["p2i_attn"].append(p2i_attn)

        # Apply the final attenion layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out, final_attn, final_q, final_k, final_v, final_q_pre, final_k_pre, final_v_pre = (
            self.final_attn_token_to_image(q=q, k=k, v=keys)
        )
        if TwoWayTransformerObserver.debug:
            TwoWayTransformerObserver.attention_score["final_attn"] = final_attn
            TwoWayTransformerObserver.attention_score["final_q"] = final_q
            TwoWayTransformerObserver.attention_score["final_k"] = final_k
            TwoWayTransformerObserver.attention_score["final_v"] = final_v
            TwoWayTransformerObserver.attention_score["final_q_pre"] = final_q_pre
            TwoWayTransformerObserver.attention_score["final_k_pre"] = final_k_pre
            TwoWayTransformerObserver.attention_score["final_v_pre"] = final_v_pre
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys

    @staticmethod
    def extract_weights(model):
        """Extract QKV weights from all attention layers once."""
        weights = {}
        for layer_idx, layer in enumerate(model.mask_decoder.transformer.layers):
            # Self-attention weights
            weights[f"p2p_q_w_layer{layer_idx}"] = layer.self_attn.q_proj.weight.data
            weights[f"p2p_k_w_layer{layer_idx}"] = layer.self_attn.k_proj.weight.data
            weights[f"p2p_v_w_layer{layer_idx}"] = layer.self_attn.v_proj.weight.data

            # Cross-attention (token to image) weights
            weights[f"p2i_q_w_layer{layer_idx}"] = layer.cross_attn_token_to_image.q_proj.weight.data
            weights[f"p2i_k_w_layer{layer_idx}"] = layer.cross_attn_token_to_image.k_proj.weight.data
            weights[f"p2i_v_w_layer{layer_idx}"] = layer.cross_attn_token_to_image.v_proj.weight.data

            # Cross-attention (image to token) weights
            weights[f"i2p_q_w_layer{layer_idx}"] = layer.cross_attn_image_to_token.q_proj.weight.data
            weights[f"i2p_k_w_layer{layer_idx}"] = layer.cross_attn_image_to_token.k_proj.weight.data
            weights[f"i2p_v_w_layer{layer_idx}"] = layer.cross_attn_image_to_token.v_proj.weight.data

        # Final attention layer weights
        final_attn = model.mask_decoder.transformer.final_attn_token_to_image
        weights["final_q_w"] = final_attn.q_proj.weight.data
        weights["final_k_w"] = final_attn.k_proj.weight.data
        weights["final_v_w"] = final_attn.v_proj.weight.data

        TwoWayTransformerObserver.weights = weights
        return weights

    def clear_dict():
        TwoWayTransformerObserver.attention_score = defaultdict(list)
        TwoWayTransformerObserver.weights = {}


class TwoWayAttentionBlockObserver(TwoWayAttentionBlock):
    """
    Observer wrapper for TwoWayAttentionBlock that captures attention metrics.

    Extends TwoWayAttentionBlock to return attention scores and intermediate
    values for debugging and analysis purposes.
    """

    attention_dict = {}

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_pe: torch.Tensor,
        key_pe: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries, p2p_attn, p2p_q, p2p_k, p2p_v, p2p_q_pre, p2p_k_pre, p2p_v_pre = self.self_attn(
                q=queries, k=queries, v=queries
            )
        else:
            q = queries + query_pe
            attn_out, p2p_attn, p2p_q, p2p_k, p2p_v, p2p_q_pre, p2p_k_pre, p2p_v_pre = self.self_attn(
                q=q, k=q, v=queries
            )
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out, p2i_attn, p2i_q, p2i_k, p2i_v, p2i_q_pre, p2i_k_pre, p2i_v_pre = self.cross_attn_token_to_image(
            q=q, k=k, v=keys
        )
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out, i2p_attn, i2p_q, i2p_k, i2p_v, i2p_q_pre, i2p_k_pre, i2p_v_pre = self.cross_attn_image_to_token(
            q=k, k=q, v=queries
        )
        keys = keys + attn_out
        keys = self.norm4(keys)

        return (
            queries,
            keys,
            p2p_attn,
            p2p_q,
            p2p_k,
            p2p_v,
            p2p_q_pre,
            p2p_k_pre,
            p2p_v_pre,
            p2i_attn,
            p2i_q,
            p2i_k,
            p2i_v,
            p2i_q_pre,
            p2i_k_pre,
            p2i_v_pre,
            i2p_attn,
            i2p_q,
            i2p_k,
            i2p_v,
            i2p_q_pre,
            i2p_k_pre,
            i2p_v_pre,
        )


class AttentionObserver(Attention):
    """
    Attention layer with quantization support and activation tracking.

    This class extends the standard Attention layer to support:
    - Quantization of activations
    - Optional processing strategies for Q, K, V
    - Tracking and returning attention scores
    """

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Store pre-projection activations
        q_pre = q
        k_pre = k
        v_pre = v

        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # q =   quantize_activation_per_token_absmax(q, n_bits=4)
        # k =   quantize_activation_per_token_absmax(k, n_bits=8)
        # v =   quantize_activation_per_token_absmax(v, n_bits=8)

        if self.processor is not None and (
            "cross" in self.name or "final" in self.name
        ):
            q = self._separate_heads(q, self.num_heads)
            k = self._separate_heads(k, self.num_heads)
            v = self._separate_heads(v, self.num_heads)
            q, k, v = self.processor.process(q, k, v, self.name, self.n_bits)
            q = self._recombine_heads(q)
            k = self._recombine_heads(k)
            v = self._recombine_heads(v)
            # q = quantize_activation_per_token_absmax(q, n_bits=self.n_bits)
            # k = quantize_activation_per_token_absmax(k, n_bits=self.n_bits)

        v = quantize_activation_per_token_absmax(v, n_bits=self.n_bits)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Apply attention to values and project output
        output = attn @ v
        output = self._recombine_heads(output)
        output = self.out_proj(output)

        return output, attn, q, k, v, q_pre, k_pre, v_pre


# ============================================================================
# Quantization Functions
# ============================================================================


def replace_linear_with_target_and_quantize(
    module,
    config: QuantizationConfig,
    module_name_to_exclude,
    k_preserve=None,
):
    """
    Replace linear layers in attention modules with quantized versions.

    Args:
        module: Module to process
        config: QuantizationConfig object containing all quantization parameters
        module_name_to_exclude: List of module names to skip
        k_preserve: Number of channels to preserve in selective quantization
    """

    def _process_module_recursive(current_module, current_path=""):
        """Recursively process modules and match with parent names in statistics."""

        for name, child in current_module.named_children():
            # Build full path including parent names
            full_path = f"{current_path}.{name}" if current_path else name

            if isinstance(child, Attention) and ('cross' in name or 'final' in name):
                # Get order statistics using full path
                order = None
                topk = None

                has_processor_stat = (
                    hasattr(child, 'processor') and
                    child.processor is not None and
                    hasattr(child.processor, 'stat')
                )

                if has_processor_stat and config.weight_quant == 'selective_channel':
                    # Match statistics using full path
                    stat_data = None
                    for stat_key in child.processor.stat.keys():
                        if stat_key in full_path or full_path in stat_key:
                            stat_data = child.processor.stat[stat_key]
                            print(f"Matched statistics: {stat_key} -> {full_path}")
                            break

                    if stat_data and 'order' in stat_data:
                        order = stat_data['order']
                        if k_preserve is not None and k_preserve > 0:
                            # Select top-k most important channels
                            topk = list(range(min(k_preserve, order.size(-1))))
                        print(f"Found order statistics for {full_path}, order shape: {order.shape}, topk={topk}")

                # Process linear layers within this attention module
                for linear_name, linear_module in child.named_children():
                    if isinstance(linear_module, nn.Linear) and linear_name not in module_name_to_exclude:

                        # Determine weight quantization method
                        actual_weight_quant = config.weight_quant
                        actual_order = None
                        actual_topk = None

                        if config.weight_quant == 'selective_channel' and order is not None:
                            actual_weight_quant = 'selective_channel'
                            # Apply selective quantization to Q/K projections in cross-attention
                            if ('cross' in name or 'final' in name) and ('k_proj' in linear_name or 'q_proj' in linear_name):
                                actual_order = order
                                actual_topk = topk
                                print(f"Applying selective quantization to {full_path}.{linear_name}")

                        print(f"Processing module: {name}.{linear_name}")

                        # Create a modified config for this specific layer
                        layer_config = QuantizationConfig(
                            n_bits_w=config.n_bits_w,
                            n_bits_a=config.n_bits_a,
                            weight_quant=actual_weight_quant,
                            act_quant=config.act_quant,
                            quantize_output=config.quantize_output,
                            group_size=config.group_size,
                            quantize_weight=config.quantize_weight,
                            order=actual_order,
                            topk=actual_topk,
                        )

                        quantized_class = layer_config.get_w8a8linear_class()
                        new_module = quantized_class.from_float(
                            linear_module,
                            **layer_config.to_kwargs()
                        )
                        setattr(child, linear_name, new_module)

            else:
                # Recursively process nested modules
                _process_module_recursive(child, full_path)

    # Start recursive processing
    _process_module_recursive(module)


def mask_decoder_monkey_patch(
    model,
    processor: ProcessStrategy = None,
    n_bits=8,
    weight_quant="per_channel",
    k_preserve=0,
    debug=False,
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
            module.__class__ = AttentionObserver
            module.processor = processor
            module.name = name
            module.n_bits = n_bits
        if isinstance(module, TwoWayAttentionBlock):
            module.__class__ = TwoWayAttentionBlockObserver
            TwoWayAttentionBlockObserver.debug = debug 
        if isinstance(module, TwoWayTransformer):
            module.__class__ = TwoWayTransformerObserver

    modules_to_exclude = [
        "pos_embed",
        "cls_token",
        "patch_embed",
        "neck",
        "fpn",
        "mask_tokens",
        "iou_token",
        "output_upscaling",
        "output_hypernetworks_mlps",
    ]

    config = QuantizationConfig(
        n_bits_w=n_bits,
        n_bits_a=n_bits,
        weight_quant=weight_quant,
        act_quant="per_token",
        quantize_output=False,
    )

    replace_linear_with_target_and_quantize(
        module=model.mask_decoder,
        config=config,
        module_name_to_exclude=modules_to_exclude,
        k_preserve=k_preserve,
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

    # Setup processor with calibration
    processor = AttnBasedProcessor("attn")
    processor.calibrate(
        predictor=predictor,
        modules=(TwoWayTransformer),
        num_samples=num_calib_samples,
    )

    # Apply quantization with monkey-patching
    mask_decoder_monkey_patch(
        predictor.model,
        processor,
        n_bits=4,
        weight_quant="selective_channel",
        k_preserve=4,
    )

    # Extract weights once (before inference)
    weights = TwoWayTransformerObserver.extract_weights(predictor.model)
    print(f"Extracted {len(weights)} weight tensors")

    # Run inference
    results = inference_image(
        predictor,
        image_dir="./input_imgs/",
        example_idx=3,
        show_image=True,
    )
