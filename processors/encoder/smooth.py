"""Smooth quantization processors for SAM encoder."""

import torch
import torch.nn as nn
from functools import partial

from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
from ..base import AttentionProcessor
from RTN_quantization.per_tensor_channel_group import (
    quantize_activation_per_token_absmax,
    quantize_weight_per_channel_absmax,
)


class EncoderAttentionProcessorSmoothMeanQ(AttentionProcessor):
    """
    Processor for calibrating and processing image encoder attention layers.

    This processor is designed specifically for the ViT-based image encoder
    in SAM, which has a different architecture than the mask decoder.
    """

    def __init__(self, strategy_name: str = 'smooth_mean_q'):
        super().__init__(strategy_name)
        self.n_bits = 8
        self.stat = {}

    def stat_linear(self, X, Y: torch.Tensor, name, linear_name):
        """Collect statistics for linear layers (QKV projections)."""
        self.stat_tensor(name, X)
        self.stat[name][linear_name] = Y
        self.stat[name]["input" + linear_name] = X

    def stat_attn(self, X, Y: torch.Tensor, name, n_heads):
        pass

    def _separate_heads_encoder(self, qkv: torch.Tensor, num_heads: int):
        """Separate QKV tensor into heads for encoder attention."""
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v

    def _register_hooks(self, predictor, modules):
        """Register hooks for QKV linear layers."""
        def stat_attn_hook(module, X, Y: torch.Tensor, name, n_heads):
            self.stat_attn(X, Y, name, n_heads)

        linear_hooks = []
        attn_hooks = []

        # Register hooks for image encoder blocks
        for name, component in predictor.model.image_encoder.named_modules():
            if isinstance(component, (modules)):
                # Hook the QKV linear layer
                for linear_name, m in component.named_modules():
                    if isinstance(m, nn.Linear) and linear_name == 'qkv':
                        print(f"Registering hook for {name}.{linear_name}")
                        linear_hooks.append(
                            m.register_forward_hook(
                                partial(self.stat_linear, name=name, linear_name=linear_name)
                            )
                        )
                # Hook the attention module
                attn_hooks.append(
                    component.register_forward_hook(
                        partial(stat_attn_hook, name=name, n_heads=component.num_heads)
                    )
                )

        return linear_hooks, attn_hooks

    def quantize_activation_per_token_absmax(self, X, mask):
        scales = X.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2 ** (self.n_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        X.div_(scales).round_().mul_(scales)
        return X

    def quantize_activation_per_channel_absmax(self, X, mask):
        scales = X.abs().max(dim=1, keepdim=True)[0]
        q_max_8 = 2 ** (8 - 1) - 1
        q_max_4 = 2 ** (self.n_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max_8)
        X.div_(scales).round_().mul_(scales)
        return X

    def process(self, x: torch.Tensor, module, module_name: str = None):
        B, H, W, C = x.shape
        energy = self.cal_energy(x, 0.9)
        mask = torch.where(energy < 0.8, 1, 0)
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x)  # (B, H, W, 3*C)
        qkv = qkv.permute(0, 3, 1, 2).contiguous()  # (B, 3*C, H, W)
        qkv = qkv.view(B, 3, module.num_heads, -1, H, W)  # (B, 3, num_heads, C_per_head, H, W)
        qkv = qkv.permute(1, 0, 2, 4, 5, 3).contiguous()  # (3, B, num_heads, H, W, C_per_head)

        # Flatten spatial dimensions: (3, B, num_heads, H*W, C_per_head)
        qkv = qkv.view(3, B, module.num_heads, H * W, -1)
        # Merge batch and heads: (3, B*num_heads, H*W, C_per_head)
        qkv = qkv.view(3, B * module.num_heads, H * W, -1)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        attn = self.quantize_activation_per_token_absmax(attn, self.n_bits, mask)

        out = attn @ v  # (B*num_heads, H*W, C_per_head)
        out = out.view(B, module.num_heads, H, W, -1)
        out = out.permute(0, 2, 3, 1, 4).contiguous()
        out = out.view(B, H, W, -1)
        x = module.proj(out)

        return x


class EncoderAttentionProcessorSmooth(EncoderAttentionProcessorSmoothMeanQ):
    """Processor with smooth quantization of attention and values."""

    def __init__(self, strategy_name: str = 'smooth'):
        super().__init__(strategy_name)

    def _take_Q(self, args=None):
        self.qkT_v = args.quantization.qkT_v
        self.n_bits_act = args.quantization.n_bits

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Process with smooth quantization."""
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))
        if self.qkT_v:
            attn = quantize_activation_per_token_absmax(attn, n_bits=self.n_bits_act)
            v = quantize_weight_per_channel_absmax(v.permute(0, 2, 1), n_bits=self.n_bits_act).permute(0, 2, 1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x


class EncoderAttentionProcessorSmoothLogQ(EncoderAttentionProcessorSmoothMeanQ):
    """Processor with logarithmic quantization."""

    def __init__(self, strategy_name: str = 'smooth_log_q'):
        super().__init__(strategy_name)

    def _take_Q(self, args=None):
        self.qkT_v = args.quantization.qkT_v
        self.n_bits_act = args.quantization.n_bits

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Process with logarithmic quantization."""
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))
        if self.qkT_v:
            from RTN_quantization.per_tensor_channel_group import quantize_activation_log_per_token_absmax, quantize_weight_log_per_channel
            attn = quantize_activation_log_per_token_absmax(attn, n_bits=self.n_bits_act)
            v = quantize_weight_log_per_channel(v.permute(0, 2, 1), n_bits=self.n_bits_act).permute(0, 2, 1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x
