import copy
# Standard library imports
from collections import defaultdict
from typing import Optional, Tuple

import torch
from segment_anything.modeling.image_encoder import (
    Attention as EncoderAttention,
    Block as EncoderBlock,
    ImageEncoderViT,
)

from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'RTN_quantization'))
from segment_anything.modeling.image_encoder import (
    window_partition,
    window_unpartition,
)

def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def calculate_correlation(x: torch.Tensor, y: torch.Tensor, method: str = 'pearson') -> float:
    """
    Calculate correlation between two tensors.

    Args:
        x: First tensor to correlate
        y: Second tensor to correlate (must be same shape as x)
        method: Correlation method - 'pearson', 'spearman', or 'cosine'

    Returns:
        Correlation coefficient as a float
    """
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    # Convert to numpy and flatten
    x_flat = to_numpy(x).flatten()
    y_flat = to_numpy(y).flatten()

    if x_flat.shape != y_flat.shape:
        raise ValueError(f"Tensors must have same shape. Got {x_flat.shape} and {y_flat.shape}")

    if method == 'pearson':
        corr, _ = pearsonr(x_flat, y_flat)
    elif method == 'spearman':
        corr, _ = spearmanr(x_flat, y_flat)
    elif method == 'cosine':
        # Cosine similarity
        x_norm = x_flat / (np.linalg.norm(x_flat) + 1e-8)
        y_norm = y_flat / (np.linalg.norm(y_flat) + 1e-8)
        corr = np.dot(x_norm, y_norm)
    else:
        raise ValueError(f"Unknown correlation method: {method}. Choose 'pearson', 'spearman', or 'cosine'")

    return float(corr)




class AttentionObserver(EncoderAttention):


    attention_score = defaultdict(list)

    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent Attention class."""
        super().__init__(*args, **kwargs)
        # Quantization attributes

  
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  
        B, H, W, C = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        x = x.reshape(B, H*W, C)
        bias = self.qkv.bias[None, None, ...]
        x_mean = x.mean(1, keepdim=True)
        x_hat = x - x_mean

        ImageEncoderViTObserver.attention_score[f"block_x"].append(to_numpy(x))
        ImageEncoderViTObserver.attention_score[f"block_x_hat"].append(to_numpy(x_hat))
        ImageEncoderViTObserver.attention_score[f"block_x_mean"].append(to_numpy(x_mean))

        qkv_hat = self.qkv(x_hat)
        qkv_mean = self.qkv(x_mean) - bias
        qkv   = self.qkv(x)       


        qkv = qkv.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv_hat = qkv_hat.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv_mean = qkv_mean.reshape(B, 1, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        ImageEncoderViTObserver.attention_score[f"block_q"].append(to_numpy(q))
        ImageEncoderViTObserver.attention_score[f"block_k"].append(to_numpy(k))
        ImageEncoderViTObserver.attention_score[f"block_v"].append(to_numpy(v))
        q_hat, k_hat, v_hat = qkv_hat.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        ImageEncoderViTObserver.attention_score[f"block_q_hat"].append(to_numpy(q_hat))
        ImageEncoderViTObserver.attention_score[f"block_k_hat"].append(to_numpy(k_hat))
        ImageEncoderViTObserver.attention_score[f"block_v_hat"].append(to_numpy(v_hat))
        q_mean, _, v_mean = qkv_mean.reshape(3, B * self.num_heads, 1, -1).unbind(0)

        assert torch.allclose(q, q_hat+q_mean, rtol=1e-4, atol=1e-4), "q_ori != q + q_mean"
        assert torch.allclose(v, v_hat+v_mean, rtol=1e-4, atol=1e-4), "v_ori != v + v_mean"
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn_hat = (q_hat * self.scale) @ k_hat.transpose(-2, -1)
        attn_mean = (q_mean * self.scale) @ k_hat.transpose(-2, -1)
        attn_hat = attn_hat + attn_mean

        if self.use_rel_pos:
            attn_hat = add_decomposed_rel_pos(
                attn_hat, q_hat+q_mean, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)
        attn_hat = attn_hat.softmax(dim=-1)
        attn_mean = attn_mean.softmax(dim=-1)

        output = (
            (attn_hat @ (v_hat + v_mean)).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        )
        output = self.proj(output)

        return output, x_hat, attn_hat, attn_mean, attn, q_hat, k_hat, v_hat, q_mean, v_mean

    @staticmethod
    def clear_dict():
        """Clear the attention score dictionary."""
        AttentionObserver.attention_score = defaultdict(list)


class BlockObserver(EncoderBlock):
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

        # x, attn, q, k, v = self.attn(x)
        x, x_hat, attn, attn_mean, attn_ori, q_hat, k_hat, v_hat, q_mean, v_mean  = self.attn(x)
        x_attn = x
        # Reverse window partition
        if self.window_size > 0:
          
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x_normed = self.norm2(x)
        ImageEncoderViTObserver.attention_score[f"block_x_attn"].append(to_numpy(x_normed))
        x = x + self.mlp(x_normed)

        return  x, x_attn, x_hat, attn, attn_mean, attn_ori, q_hat, k_hat, v_hat, q_mean, v_mean 


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
            x, x_attn, x_hat, attn, attn_mean, attn_ori, q_hat, k_hat, v_hat, q_mean, v_mean  = blk(x)


            ImageEncoderViTObserver.attention_score[f"block_attn"].append(to_numpy(attn))
            ImageEncoderViTObserver.attention_score[f"block_attn_mean"].append(to_numpy(attn_mean))
            ImageEncoderViTObserver.attention_score[f"block_attn_ori"].append(to_numpy(attn_ori))

            ImageEncoderViTObserver.attention_score[f"block_q_hat"].append(to_numpy(q_hat))
            ImageEncoderViTObserver.attention_score[f"block_k_hat"].append(to_numpy(k_hat))
            ImageEncoderViTObserver.attention_score[f"block_v_hat"].append(to_numpy(v_hat))

            ImageEncoderViTObserver.attention_score[f"block_q_mean"].append(to_numpy(q_mean))
            ImageEncoderViTObserver.attention_score[f"block_v_mean"].append(to_numpy(v_mean))

            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x, interm_embeddings

    @staticmethod
    def clear_dict():
        """Clear the attention score dictionary."""
        ImageEncoderViTObserver.attention_score = defaultdict(list)


def image_encoder_observer_patch(
    model,
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
        if isinstance(module, (EncoderAttention)):
            module.__class__ = AttentionObserver 
        if isinstance(module, (EncoderBlock)):
            module.__class__ = BlockObserver
        if isinstance(module, (ImageEncoderViT)):
            module.__class__ = ImageEncoderViTObserver


  