"""
SAM3-style RoPE helpers for use with the FA2 Ampere kernel in flash_attn.py.

SAM3's `Sam3ViTRoPEAttention` applies a 2D *axial* rotary position embedding
to Q and K **before** the attention matmul — there is no additive learnable
rel_pos bias (unlike SAM1).  The cos/sin tables are built from x/y indices:

    inv_freq = 1 / theta ** (torch.arange(0, D, 4)[: D//4] / D)        # (D/4,)
    freqs_x  = outer(x_pos, inv_freq)                                  # (Sq, D/4)
    freqs_y  = outer(y_pos, inv_freq)                                  # (Sq, D/4)
    inv      = cat([freqs_x, freqs_y], -1).repeat_interleave(2, -1)    # (Sq, D)
    cos, sin = inv.cos(), inv.sin()                                    # (Sq, D)

RoPE itself is a pairwise rotation on the head-dim:

    q_rot[..., 2i  ] =  q[..., 2i]   * cos_i  -  q[..., 2i+1] * sin_i
    q_rot[..., 2i+1] =  q[..., 2i+1] * cos_i  +  q[..., 2i  ] * sin_i

Because RoPE acts purely on the Q/K projections and produces tensors of the
same shape, the cleanest way to wire it into the existing FA2 kernel
(`FlashAttentionForwardAmpere`) is to **pre-rotate** Q and K once in
PyTorch and feed zero rel_h/rel_w bias tensors.  The relative cost of the
pre-rotation is O(B·Sq·H·D), negligible compared with the O(B·Sq²·H·D)
attention itself.

If you ever want a fully-fused kernel: load cos/sin into SMEM alongside Q
and apply `rotate_pairwise(...) * sin + (...) * cos` right after the SMEM→RMEM
load of Q (and per-tile for K).  That work is left as future hardening.
"""

import torch


# ---------------------------------------------------------------------------
# 2D axial RoPE table builder — matches Sam3ViTRotaryEmbedding exactly
# ---------------------------------------------------------------------------

def build_rope_2d(
    end_x: int,
    end_y: int,
    head_dim: int,
    theta: float = 10000.0,
    scale: float = 1.0,
    device="cuda",
    dtype=torch.float32,
):
    """
    Build the 2D axial cos/sin tables used by SAM3's ViT.

    Returns
    -------
    cos, sin : (end_x*end_y, head_dim) float tensors
    """
    if head_dim % 4 != 0:
        raise ValueError("head_dim must be divisible by 4 for axial RoPE")

    freqs = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 4, device=device).float()[: head_dim // 4] / head_dim)
    )

    flat = torch.arange(end_x * end_y, device=device, dtype=torch.long)
    x_pos = (flat % end_x).float() * scale
    y_pos = torch.div(flat, end_x, rounding_mode="floor").float() * scale

    fx = torch.outer(x_pos, freqs)              # (Sq, D/4)
    fy = torch.outer(y_pos, freqs)              # (Sq, D/4)
    inv = torch.cat([fx, fy], dim=-1)           # (Sq, D/2)
    inv = inv.repeat_interleave(2, dim=-1)      # (Sq, D)

    return inv.cos().to(dtype), inv.sin().to(dtype)


# ---------------------------------------------------------------------------
# Pairwise rotation (matches Sam3 modeling)
# ---------------------------------------------------------------------------

def rotate_pairwise(x: torch.Tensor) -> torch.Tensor:
    """[a0, a1, a2, a3, ...] → [-a1, a0, -a3, a2, ...]   (last dim only)."""
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


# ---------------------------------------------------------------------------
# Apply RoPE to Q/K in (B, S, H, D) layout — the layout used by FA2 kernel
# ---------------------------------------------------------------------------

def apply_rope_bshd(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    """
    Apply 2D RoPE to query/key tensors of shape (B, S, H, D).

    `cos`, `sin` have shape (S, D) and broadcast over batch and heads.
    Computation is done in fp32 then cast back to the input dtype, matching
    `apply_rotary_pos_emb_2d` in transformers.models.sam3.modeling_sam3.
    """
    cos_b = cos.view(1, cos.shape[0], 1, cos.shape[1])
    sin_b = sin.view(1, sin.shape[0], 1, sin.shape[1])

    q_f = q.float()
    k_f = k.float()
    q_rot = q_f * cos_b + rotate_pairwise(q_f) * sin_b
    k_rot = k_f * cos_b + rotate_pairwise(k_f) * sin_b
    return q_rot.type_as(q), k_rot.type_as(k)
