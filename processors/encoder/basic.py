"""Basic encoder attention processors for SAM quantization."""

import torch
from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
from ..base import AttentionProcessor


class EncoderAttentionProcessor(AttentionProcessor):
    """
    Processor for calibrating and processing image encoder attention layers.

    This processor is designed specifically for the ViT-based image encoder
    in SAM, which has a different architecture than the mask decoder.
    """

    def __init__(self, strategy_name: str = 'base'):
        super().__init__(strategy_name)
        self.stat = {}

    def stat_linear(self, X, Y: torch.Tensor, name, linear_name):
        """
        Collect statistics for linear layers (QKV projections).

        Args:
            X: Input tensor
            Y: Output tensor from linear layer
            name: Module name
            linear_name: Linear layer name (e.g., 'qkv', 'proj')
        """
        pass

    def stat_attn(self, X, Y: torch.Tensor, name, n_heads):
        pass

    def _separate_heads_encoder(self, qkv: torch.Tensor, num_heads: int):
        """
        Separate QKV tensor into heads for encoder attention.

        Args:
            qkv: Combined QKV tensor of shape (B, H*W, 3, num_heads, C_per_head)
            num_heads: Number of attention heads

        Returns:
            q, k, v tensors each of shape (B, num_heads, H*W, C_per_head)
        """
        # qkv shape: (B, H*W, 3, num_heads, C_per_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, H*W, C_per_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """
        Process Q, K, V tensors with quantization for encoder attention.

        Args:
            x: Input tensor
            module: Attention module
            module_name: Module name

        Returns:
            Processed output tensor
        """
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x


class EncoderRecenterAttentionProcessor(AttentionProcessor):
    """
    Encoder processor that recenters QKV projections by subtracting spatial means.

    Processing Strategy:
    -------------------
    1. Computes spatial mean: x_mean = x.mean(H).mean(W)
    2. Subtracts mean: x_hat = x - x_mean
    3. Projects: qkv_hat = QKV(x_hat), qkv_mean = QKV(x_mean)
    4. Computes attention with recentered keys
    5. Applies attention to recentered values

    This helps with quantization by centering activations around zero.
    """

    def __init__(self, strategy_name: str = 'recentered'):
        super().__init__(strategy_name)
        self.stat = {}

    def stat_linear(self, X, Y: torch.Tensor, name, linear_name):
        pass

    def stat_attn(self, X, Y: torch.Tensor, name, n_heads):
        pass

    def _separate_heads_encoder(self, qkv: torch.Tensor, num_heads: int):
        """Separate QKV tensor into heads for encoder attention."""
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v

    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Process with recentering for better quantization."""
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        x_mean = x.mean(1, keepdim=True).mean(2, keepdim=True)
        x_hat = x - x_mean
        qkv_hat = module.qkv(x_hat).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv_mean = module.qkv(x_mean).reshape(B, 1, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q_hat, k_hat, v_hat = qkv_hat.reshape(3, B * module.num_heads, H * W, -1).unbind(0)
        q_mean, k_mean, v_mean = qkv_mean.reshape(3, B * module.num_heads, 1, -1).unbind(0)

        attn = (q_hat * module.scale) @ k_hat.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q_hat + q_mean, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = ((attn @ v_hat) + v_mean).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x
