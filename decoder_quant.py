import math

import torch
import torch.nn as nn

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling.transformer import (
    Attention,
    TwoWayAttentionBlock,
    TwoWayTransformer,
)

from collections import defaultdict

from processors.decoder import DecoderDoNothingProcessor
from processors.decoder_observer import (
    TwoWayTransformerObserverElementLow,
    TwoWayAttentionBlockObserverElementLow,
    AttentionObserverElementLow,
)
from RTN_quantization.utils import replace_linear_with_quantized, QuantizationConfig
from utils.utils import inference_image


# ============================================================================
# Decoder Observer Classes (used by sam_engine.py and notebooks)
# ============================================================================


class TwoWayTransformerObserver(TwoWayTransformer):
    """
    Observer for TwoWayTransformer that tracks Q/K/V activations and attention scores.
    Used for analysis and calibration of the mask decoder.
    """

    attention_score = defaultdict(list)
    weights = {}
    debug = False

    def forward(self, image_embedding, image_pe, point_embedding):
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            TwoWayTransformerObserver.attention_score["pre_p"].append(queries)
            TwoWayTransformerObserver.attention_score["pre_i"].append(keys)
            (
                queries, keys,
                p2p_attn, p2p_q, p2p_k, p2p_v, p2p_q_pre, p2p_k_pre, p2p_v_pre,
                p2i_attn, p2i_q, p2i_k, p2i_v, p2i_q_pre, p2i_k_pre, p2i_v_pre,
                i2p_attn, i2p_q, i2p_k, i2p_v, i2p_q_pre, i2p_k_pre, i2p_v_pre,
            ) = layer(queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe)

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
        """Extract QKV weights from all attention layers."""
        weights = {}
        for layer_idx, layer in enumerate(model.mask_decoder.transformer.layers):
            weights[f"p2p_q_w_layer{layer_idx}"] = layer.self_attn.q_proj.weight.data
            weights[f"p2p_k_w_layer{layer_idx}"] = layer.self_attn.k_proj.weight.data
            weights[f"p2p_v_w_layer{layer_idx}"] = layer.self_attn.v_proj.weight.data
            weights[f"p2i_q_w_layer{layer_idx}"] = layer.cross_attn_token_to_image.q_proj.weight.data
            weights[f"p2i_k_w_layer{layer_idx}"] = layer.cross_attn_token_to_image.k_proj.weight.data
            weights[f"p2i_v_w_layer{layer_idx}"] = layer.cross_attn_token_to_image.v_proj.weight.data
            weights[f"i2p_q_w_layer{layer_idx}"] = layer.cross_attn_image_to_token.q_proj.weight.data
            weights[f"i2p_k_w_layer{layer_idx}"] = layer.cross_attn_image_to_token.k_proj.weight.data
            weights[f"i2p_v_w_layer{layer_idx}"] = layer.cross_attn_image_to_token.v_proj.weight.data
        final_attn = model.mask_decoder.transformer.final_attn_token_to_image
        weights["final_q_w"] = final_attn.q_proj.weight.data
        weights["final_k_w"] = final_attn.k_proj.weight.data
        weights["final_v_w"] = final_attn.v_proj.weight.data
        TwoWayTransformerObserver.weights = weights
        return weights

    @staticmethod
    def clear_dict():
        TwoWayTransformerObserver.attention_score = defaultdict(list)
        TwoWayTransformerObserver.weights = {}


def separate_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, n, c = x.shape
    x = x.reshape(b, n, num_heads, c // num_heads)
    return x.transpose(1, 2)


def recombine_heads(x: torch.Tensor) -> torch.Tensor:
    b, n_heads, n_tokens, c_per_head = x.shape
    x = x.transpose(1, 2)
    return x.reshape(b, n_tokens, n_heads * c_per_head)


def re_cal_attn(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    _, _, _, c_per_head = q.shape
    attn = q @ k.permute(0, 1, 3, 2)
    attn = attn / math.sqrt(c_per_head)
    attn = torch.softmax(attn, dim=-1)
    return attn


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
        config: QuantizationConfig with quantization parameters
        module_name_to_exclude: List of module names to skip
        k_preserve: Number of channels to preserve in selective quantization
    """
    def _process_module_recursive(current_module, current_path=""):
        for name, child in current_module.named_children():
            full_path = f"{current_path}.{name}" if current_path else name

            if isinstance(child, Attention) and ('cross' in name or 'final' in name):
                order = None
                topk = None

                has_processor_stat = (
                    hasattr(child, 'processor') and
                    child.processor is not None and
                    hasattr(child.processor, 'stat')
                )

                if has_processor_stat and config.weight_quant == 'selective_channel':
                    stat_data = None
                    for stat_key in child.processor.stat.keys():
                        if stat_key in full_path or full_path in stat_key:
                            stat_data = child.processor.stat[stat_key]
                            print(f"Matched statistics: {stat_key} -> {full_path}")
                            break

                    if stat_data and 'order' in stat_data:
                        order = stat_data['order']
                        if k_preserve is not None and k_preserve > 0:
                            topk = list(range(min(k_preserve, order.size(-1))))
                        print(f"Found order statistics for {full_path}, order shape: {order.shape}, topk={topk}")

                for linear_name, linear_module in child.named_children():
                    if isinstance(linear_module, nn.Linear) and linear_name not in module_name_to_exclude:
                        actual_weight_quant = config.weight_quant
                        actual_order = None
                        actual_topk = None

                        if config.weight_quant == 'selective_channel' and order is not None:
                            if ('cross' in name or 'final' in name) and ('k_proj' in linear_name or 'q_proj' in linear_name):
                                actual_order = order
                                actual_topk = topk
                                print(f"Applying selective quantization to {full_path}.{linear_name}")

                        print(f"Processing module: {name}.{linear_name}")

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
                _process_module_recursive(child, full_path)

    _process_module_recursive(module)


def mask_decoder_monkey_patch(
    model,
    processor=None,
    n_bits=8,
    weight_quant="per_channel",
    k_preserve=0,
    debug=False,
):
    """
    Apply monkey-patching to SAM mask decoder for quantization.

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
            module.high_element = False
        if isinstance(module, TwoWayAttentionBlock):
            module.__class__ = TwoWayAttentionBlockObserverElementLow
        if isinstance(module, TwoWayTransformer):
            module.__class__ = TwoWayTransformerObserverElementLow
            TwoWayTransformerObserverElementLow.debug = debug

    config = QuantizationConfig(
        n_bits_w=n_bits,
        n_bits_a=n_bits,
        weight_quant=weight_quant,
        act_quant="per_token",
        quantize_output=False,
    )
    replace_linear_with_quantized(
        module=model.mask_decoder,
        config=config,
        module_name_to_exclude=[],
    )


if __name__ == "__main__":
    model_type = "vit_l"
    num_calib_samples = 8
    checkpoint_path = "./pretrained_checkpoint/sam_hq_vit_l.pth"

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to("cuda")
    predictor = SamPredictor(sam)

    processor = DecoderDoNothingProcessor("donothing")
    processor.calibrate(
        predictor=predictor,
        modules=(TwoWayTransformer),
        num_samples=num_calib_samples,
    )
    mask_decoder_monkey_patch(
        predictor.model,
        processor,
        n_bits=4,
        weight_quant="selective_channel",
        k_preserve=4,
    )
    results = inference_image(
        predictor,
        image_dir="./input_imgs/",
        example_idx=3,
        show_image=True,
    )
