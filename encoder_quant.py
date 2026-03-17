import copy
from collections import defaultdict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling.image_encoder import (
    Attention as EncoderAttention,
    Block as EncoderBlock,
    ImageEncoderViT,
    add_decomposed_rel_pos,
    window_partition,
    window_unpartition,
)

from processors.base import AttentionProcessor
from processors.encoder import EncoderRecenterAttentionProcessor
from utils.quant_utils import quantize_activation_per_token_absmax
from quant.quant_utils import replace_linear_with_quantized, QuantizationConfig
from utils.utils import inference_image


@torch.no_grad()
def quantize_weight_per_channel_absmax(w: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Quantize weights per output channel using absolute maximum scaling."""
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


class nnLinear_qkv_hat(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_weight = None
        self.quant_activation = None
        self.single_head_features = self.out_features // 3
        self.n_bits = 8

    def quantize_weight(self, n_bits=8):
        q_weight = self.weight[:, :]
        q_weight_q = quantize_weight_per_channel_absmax(q_weight, n_bits=n_bits)
        with torch.no_grad():
            self.weight[:, :] = q_weight_q

    def forward(self, x, q_scales):
        q_weight = self.weight[:self.single_head_features, :]
        k_weight = self.weight[self.single_head_features:2*self.single_head_features, :]
        v_weight = self.weight[2*self.single_head_features:, :]
        x_scaled = x / q_scales.view(1, 1, -1)

        if self.quant_activation == "per_token":
            x_scaled = quantize_activation_per_token_absmax(x_scaled, n_bits=self.n_bits)
        q_output = torch.nn.functional.linear(x_scaled, q_weight)

        x = quantize_activation_per_token_absmax(x, n_bits=self.n_bits)
        k_output = torch.nn.functional.linear(x, k_weight)
        v_output = torch.nn.functional.linear(x, v_weight)

        output = torch.cat([q_output, k_output, v_output], dim=-1)
        if self.bias is not None:
            output = output + self.bias
        return output


class QuantizedAttention(EncoderAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_processor(self, processor: AttentionProcessor, module_name):
        self.processor = processor
        self.module_name = module_name

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.processor.process(x, self, self.module_name)


class QuantizedAttentionQuarotCenterQ(QuantizedAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def smooth(self, alpha=0.5):
        input_qkv = self.stat['inputqkv']
        output_qkv = self.stat['qkv']
        act_scales = self.stat['act_scales']

        device, dtype = self.qkv_w_hat.weight.device, self.qkv_w_hat.weight.dtype
        act_scales = act_scales.to(device=device, dtype=dtype)

        C = act_scales.numel()
        q_weight = self.qkv_w_hat.weight[:C, :]
        weight_scales = q_weight.abs().max(dim=0)[0].clamp(min=1e-5)

        scales = (
            (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
            .clamp(min=1e-5)
            .to(device)
            .to(dtype)
        )
        self.q_scales = scales

        with torch.no_grad():
            self.qkv_w_hat.weight[:C, :].mul_(scales.view(1, -1))

    def _take_qkv_w_hat(self, args=None):
        device = args.quarot_inf.device
        key = '.'.join(self.module_name.split('.')[1:])
        self.stat = self.processor.stat[key]
        self.qkv_w_hat = nnLinear_qkv_hat(
            self.qkv.in_features,
            self.qkv.out_features,
            bias=self.qkv.bias is not None
        ).to(device)
        self.qkv_w_hat.n_bits = args.rtn_ro_config.n_bits
        self.qkv_w_hat.weight.data = self.qkv.weight.data.clone().to(device)
        self.qkv_w_hat.bias.data = self.qkv.bias.data.clone().to(device)
        self.smooth()
        self.qkv_w_hat.quant_activation = args.quantization.act_quant
        self.qkv_w_hat.quant_weight = args.quantization.weight_quant
        self.qkv_w_hat.quantize_weight(n_bits=args.rtn_ro_config.n_bits)
        self.percent = args.quantization.percent


def image_encoder_monkey_patch(
    model,
    processor=None,
    n_bits=8,
    weight_quant="per_channel",
    act_quant="per_token",
    layers=[0, 5, 17],
    debug=False,
    device="cuda",
    args_yaml=None,
):
    """
    Apply monkey-patching to SAM image encoder for quantization.

    Args:
        model: SAM model to patch
        processor: Processing strategy for activations
        n_bits: Number of bits for quantization
        weight_quant: Weight quantization strategy
        act_quant: Activation quantization strategy
        device: Target device
    """
    for name, module in model.named_modules():
        if isinstance(module, EncoderAttention):
            module.__class__ = QuantizedAttention
            module.set_processor(processor, name)

    if n_bits < 16:
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
            module_name_to_exclude=['decoder'],
        )
    model.to(device)


if __name__ == "__main__":
    model_type = "vit_l"
    num_calib_samples = 1
    checkpoint_path = "./ckts/sam_hq_vit_l.pth"

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to("cuda")
    predictor = SamPredictor(sam)

    processor = EncoderRecenterAttentionProcessor("recenter")
    processor.calibrate(
        predictor=predictor,
        modules=(EncoderAttention,),
        num_samples=num_calib_samples,
    )
    image_encoder_monkey_patch(
        predictor.model,
        processor=processor,
        n_bits=4,
        weight_quant="per_channel",
        act_quant="per_token",
    )
    results = inference_image(
        predictor,
        image_dir="./input_imgs/",
        example_idx=3,
        show_image=True,
    )
