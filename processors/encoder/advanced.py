"""Advanced quantization processors for SAM encoder."""

import torch

from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
from .smooth import EncoderAttentionProcessorSmoothMeanQ
from RTN_quantization.per_tensor_channel_group import (
    quantize_activation_per_token_absmax,
    quantize_weight_per_channel_absmax,
    quantize_activation_low_high_density_activation_index,
)
from utils.utils import quantize_activation_per_highblock_abmax, find_O_qha
# from quarot import rotation_utils


class EncoderAttentionProcessorCompensate(EncoderAttentionProcessorSmoothMeanQ):
    """Processor with compensation for quantization errors."""

    def __init__(self, strategy_name: str = 'compensate'):
        super().__init__(strategy_name)

    def _take_Q(self, args=None):
        pass

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Process with quantization compensation."""
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        attn_qha, indicies = quantize_activation_per_highblock_abmax(attn, n_bits=4, percent=0.5, block_size=1)
        O_qha = find_O_qha(qattn=attn_qha, v=v, indices=indicies, n_bits=4, block_size=1)

        x = (O_qha).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x


class EncoderAttentionProcessorHighLow(EncoderAttentionProcessorSmoothMeanQ):
    """Processor with high-low density activation quantization."""

    def __init__(self, strategy_name: str = 'high_low_attn_v'):
        super().__init__(strategy_name)

    def _take_Q(self, args=None):
        self.n_bits_act = args.quantization.n_bits
        self.percent = args.quantization.percent

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Process with high-low density quantization."""
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

        attn, indices = quantize_activation_low_high_density_activation_index(attn, n_bits=self.n_bits_act, percent=self.percent, quantizehigh=True)
        v = quantize_activation_low_high_density_activation_index(v, n_bits=self.n_bits_act, percent=self.percent, quantizehigh=True, indices=indices)[0]

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x


class EncoderAttentionProcessorQuarot(EncoderAttentionProcessorSmoothMeanQ):
    """Processor with QuaRot rotation-based quantization."""

    def __init__(self, strategy_name: str = 'quarot'):
        super().__init__(strategy_name)
        self.Q = None

    def _take_Q(self, args=None):
        self.Q = rotation_utils.get_orthogonal_matrix(args.quarot_inf.hidden_size_image_en,
                                                       args.quarot_inf.rotate_mode,
                                                       device=args.quarot_inf.device,
                                                       seed=args.quarot_inf.seed)
        self.qkT_v = args.quantization.qkT_v
        self.n_bits_act = args.rtn_ro_config.n_bits

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Process with QuaRot rotation."""
        # Apply Q matrix multiplication if provided
        if self.Q is not None:
            B, H, W, C = x.shape
            self.Q = self.Q.to(dtype=x.dtype)

            x_flat = x.reshape(B, H * W, C)
            x_rotated = torch.matmul(x_flat, self.Q)
            x = x_rotated.reshape(B, H, W, C)

        B, H, W, _ = x.shape

        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
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
