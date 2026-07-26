"""Configurable saliency + grid-size ablation for the SparseSAM token ordering.

SparseSAM ranks space-filling-curve token groups by a saliency score and uses
that ranking as the permutation fed to the block-sparse attention kernel
(`PiToMe/algo/sparsesam/sam.py::ToMeSAMAttention.forward`). The shipped
implementation hard-codes one scorer and `group_size=4`.

This module re-implements that permutation with two knobs exposed:

    * `group_size`  — tokens per curve group (the "grid size"): 2, 4, 8, 16, ...
    * `saliency`    — how a group is scored:
                        sobel3 / sobel5 / sobel7   Sobel gradient magnitude
                        scharr3 / scharr5          Scharr gradient magnitude
                        laplacian3 / laplacian5    |Laplacian| response
                        feature_norm               token L2 feature norm
                        attention                  attention-derived mass
                        variance_dissim            shipped sam.py scorer
                        random                     ranking sanity baseline

Nothing under `PiToMe/` is modified: `install()` swaps the module-level
`tile_stride_matching` symbol inside `PiToMe.algo.sparsesam.sam` at runtime and
`restore()` puts the original back.

Usage
-----
    import sparsesam_saliency as sal

    sal.install(sal.SaliencyConfig(saliency="sobel5", group_size=8))
    ...                                  # run the SparseSAM-patched encoder
    sal.restore()

or, equivalently, `with sal.saliency_permutation(cfg): ...`.
"""

from __future__ import annotations

import contextlib
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "sam-hq"), os.path.join(_HERE, "PiToMe")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PiToMe.algo.sparsesam.hilbert_utils import get_hilbert_order  # noqa: E402
from PiToMe.algo.sparsesam.z_utils import get_z_order  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Separable derivative kernels (OpenCV `getDerivKernels` construction)
# ─────────────────────────────────────────────────────────────────────────────

def _binomial_recursion(base: List[float], ksize: int) -> List[float]:
    """Grow a length-3 kernel to length `ksize` by repeated [1,2,1] convolution."""
    k = list(base)
    while len(k) < ksize:
        out = [0.0] * (len(k) + 2)
        for i, v in enumerate(k):
            out[i] += 1.0 * v
            out[i + 1] += 2.0 * v
            out[i + 2] += 1.0 * v
        k = out
    return k


def _deriv_1d(ksize: int, order: int, flavor: str) -> Tuple[List[float], List[float]]:
    """Return (derivative_1d, smoothing_1d) of length `ksize`.

    order=1 → first derivative (Sobel/Scharr), order=2 → second (Laplacian).
    """
    if ksize % 2 == 0 or ksize < 3:
        raise ValueError(f"kernel size must be odd and >= 3, got {ksize}")

    if flavor == "scharr":
        deriv, smooth = [-1.0, 0.0, 1.0], [3.0, 10.0, 3.0]
    elif order == 1:
        deriv, smooth = [-1.0, 0.0, 1.0], [1.0, 2.0, 1.0]
    elif order == 2:
        deriv, smooth = [1.0, -2.0, 1.0], [1.0, 2.0, 1.0]
    else:
        raise ValueError(f"unsupported derivative order {order}")

    return _binomial_recursion(deriv, ksize), _binomial_recursion(smooth, ksize)


def _kernel_2d(deriv: List[float], smooth: List[float], axis: str,
               device, dtype) -> torch.Tensor:
    """Outer-product a separable pair into a (1, 1, k, k) conv kernel.

    `axis='x'` differentiates along W and smooths along H, `axis='y'` the
    transpose. Kernels are L1-normalised so responses stay comparable across
    kernel sizes (ranking is scale-invariant, but the magnitudes are logged).
    """
    d = torch.tensor(deriv, device=device, dtype=dtype)
    s = torch.tensor(smooth, device=device, dtype=dtype)
    k2d = torch.outer(s, d) if axis == "x" else torch.outer(d, s)
    k2d = k2d / k2d.abs().sum().clamp_min(1e-8)
    return k2d.view(1, 1, *k2d.shape)


_KERNEL_CACHE: Dict[tuple, torch.Tensor] = {}


def _get_kernel(spec: str, axis: str, ksize: int, order: int,
                device, dtype) -> torch.Tensor:
    key = (spec, axis, ksize, order, str(device), str(dtype))
    if key not in _KERNEL_CACHE:
        deriv, smooth = _deriv_1d(ksize, order, spec)
        _KERNEL_CACHE[key] = _kernel_2d(deriv, smooth, axis, device, dtype)
    return _KERNEL_CACHE[key]


def _depthwise(x_bchw: torch.Tensor, k2d: torch.Tensor) -> torch.Tensor:
    """Apply one (1,1,k,k) kernel to every channel independently."""
    B, C, H, W = x_bchw.shape
    pad = k2d.shape[-1] // 2
    return F.conv2d(x_bchw.reshape(B * C, 1, H, W), k2d, padding=pad).reshape(B, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Token-level saliency estimators:  (B, N, C) → (B, N)
# ─────────────────────────────────────────────────────────────────────────────

def _gradient_magnitude(x: torch.Tensor, H: int, W: int,
                        ksize: int, spec: str) -> torch.Tensor:
    """RMS-over-channels gradient magnitude, matching the shipped Sobel score."""
    B, N, C = x.shape
    x_bchw = x.reshape(B, H, W, C).permute(0, 3, 1, 2).float()
    kx = _get_kernel(spec, "x", ksize, 1, x.device, torch.float32)
    ky = _get_kernel(spec, "y", ksize, 1, x.device, torch.float32)
    gx = _depthwise(x_bchw, kx)
    gy = _depthwise(x_bchw, ky)
    return torch.sqrt((gx ** 2 + gy ** 2).mean(dim=1)).reshape(B, N)


def _laplacian_magnitude(x: torch.Tensor, H: int, W: int, ksize: int) -> torch.Tensor:
    B, N, C = x.shape
    x_bchw = x.reshape(B, H, W, C).permute(0, 3, 1, 2).float()
    kxx = _get_kernel("sobel", "x", ksize, 2, x.device, torch.float32)
    kyy = _get_kernel("sobel", "y", ksize, 2, x.device, torch.float32)
    lap = _depthwise(x_bchw, kxx) + _depthwise(x_bchw, kyy)
    return torch.sqrt((lap ** 2).mean(dim=1)).reshape(B, N)


def _feature_norm(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    return x.float().pow(2).mean(dim=-1).sqrt()


TOKEN_SCORERS = {
    "sobel3":     lambda x, H, W: _gradient_magnitude(x, H, W, 3, "sobel"),
    "sobel5":     lambda x, H, W: _gradient_magnitude(x, H, W, 5, "sobel"),
    "sobel7":     lambda x, H, W: _gradient_magnitude(x, H, W, 7, "sobel"),
    "sobel9":     lambda x, H, W: _gradient_magnitude(x, H, W, 9, "sobel"),
    "scharr3":    lambda x, H, W: _gradient_magnitude(x, H, W, 3, "scharr"),
    "scharr5":    lambda x, H, W: _gradient_magnitude(x, H, W, 5, "scharr"),
    "laplacian3": lambda x, H, W: _laplacian_magnitude(x, H, W, 3),
    "laplacian5": lambda x, H, W: _laplacian_magnitude(x, H, W, 5),
    "feature_norm": _feature_norm,
}


# ─────────────────────────────────────────────────────────────────────────────
# Group-level saliency estimators:  (B, G, gs, C) + validity mask → (B, G)
# ─────────────────────────────────────────────────────────────────────────────

def _score_variance_dissim(x_grp: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """The scorer shipped in `sam.py`: z-scored intra-group std + inter-group
    dissimilarity. Kept here so the ablation has the current method as a row."""
    B, G, gs, C = x_grp.shape
    x_grp = x_grp.float()
    grp_std = x_grp.std(dim=2).mean(dim=-1)
    grp_mean = x_grp.mean(dim=2)
    gm_norm = F.normalize(grp_mean, dim=-1)
    sim = gm_norm @ gm_norm.transpose(1, 2)
    avg_sim = (sim.sum(dim=-1) - 1.0) / max(G - 1, 1)
    dissim = -avg_sim

    eps = 1e-6
    std_n = (grp_std - grp_std.mean(-1, keepdim=True)) / (grp_std.std(-1, keepdim=True) + eps)
    dissim_n = (dissim - dissim.mean(-1, keepdim=True)) / (dissim.std(-1, keepdim=True) + eps)
    return std_n + dissim_n


def _score_attention(x_grp: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Attention-derived importance: incoming softmax mass a group receives
    from all other group representatives (a cheap G×G stand-in for the full
    N×N attention map)."""
    B, G, gs, C = x_grp.shape
    grp_mean = x_grp.float().mean(dim=2)
    logits = (grp_mean @ grp_mean.transpose(1, 2)) / math.sqrt(C)
    return torch.softmax(logits, dim=-1).sum(dim=1)


def _score_random(x_grp: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    B, G = x_grp.shape[0], x_grp.shape[1]
    return torch.rand(B, G, device=x_grp.device, dtype=torch.float32)


GROUP_SCORERS = {
    "variance_dissim": _score_variance_dissim,
    "attention": _score_attention,
    "random": _score_random,
}

SALIENCY_CHOICES = sorted(TOKEN_SCORERS) + sorted(GROUP_SCORERS)


# ─────────────────────────────────────────────────────────────────────────────
# Permutation construction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SaliencyConfig:
    """One point in the ablation grid.

    saliency   : key into TOKEN_SCORERS / GROUP_SCORERS.
    group_size : tokens per curve group ("grid size"). N need not be divisible
                 by it — the trailing partial group is scored on its valid
                 members only.
    curve      : space-filling curve used to make groups spatially compact.
    layout     : 'grouped'      — ranked groups laid out back-to-back, so the
                                  dense keep-prefix is exactly the top-scoring
                                  groups (ranking drives the token ordering);
                 'interleaved'  — stripe layout of the shipped `sam.py`
                                  (token j of every group, groups in rank
                                  order). Reproduces current behaviour.
    descending : True keeps high-saliency groups first (Sobel: high gradient
                 = detail = keep dense).
    """
    saliency: str = "sobel3"
    group_size: int = 4
    curve: str = "z"
    layout: str = "grouped"
    descending: bool = True

    def tag(self) -> str:
        return f"{self.saliency}_g{self.group_size}_{self.curve}_{self.layout}"


_CURVES = {
    "z": get_z_order,
    "hilbert": get_hilbert_order,
    "raster": lambda H, W, device=None: torch.arange(H * W, device=device),
}

_GROUP_IDX_CACHE: Dict[tuple, torch.Tensor] = {}


def _group_raster(H: int, W: int, group_size: int, curve: str, device) -> torch.Tensor:
    """(G, gs) raster indices per curve group; -1 pads the trailing group."""
    key = (H, W, group_size, curve, str(device))
    if key not in _GROUP_IDX_CACHE:
        order = _CURVES[curve](H, W, device=device).to(torch.int64)
        N = order.numel()
        n_groups = math.ceil(N / group_size)
        pad = n_groups * group_size - N
        if pad:
            order = torch.cat(
                [order, torch.full((pad,), -1, dtype=torch.int64, device=device)]
            )
        _GROUP_IDX_CACHE[key] = order.view(n_groups, group_size)
    return _GROUP_IDX_CACHE[key]


def tile_stride_matching_saliency(
    x: torch.Tensor,
    H: int,
    W: int,
    ratio: float = 0.0,
    group_size: int = 4,
    n_block: int = 64,
    cfg: Optional[SaliencyConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for `sparsesam.sam.tile_stride_matching`.

    `x` is the attention key tensor (B*heads, N, D) in raster order; returns
    (perm, inv_perm), both int64 (B*heads, N).
    """
    cfg = cfg or _ACTIVE_CFG or SaliencyConfig()
    gs = int(cfg.group_size if cfg.group_size else group_size)
    N = H * W
    assert x.shape[1] == N, f"x has {x.shape[1]} tokens, expected H*W={N}"

    with torch.no_grad():
        B, _, C = x.shape
        device = x.device

        group_raster = _group_raster(H, W, gs, cfg.curve, device)   # (G, gs)
        G = group_raster.shape[0]
        idx = group_raster.clamp_min(0)
        valid = (group_raster >= 0).to(torch.float32)               # (G, gs)

        if cfg.saliency in TOKEN_SCORERS:
            tok_score = TOKEN_SCORERS[cfg.saliency](x, H, W)        # (B, N)
            s = tok_score.gather(1, idx.reshape(1, -1).expand(B, -1)).view(B, G, gs)
            grp_score = (s * valid).sum(-1) / valid.sum(-1).clamp_min(1.0)
        else:
            gathered = x.gather(
                1, idx.reshape(1, -1, 1).expand(B, G * gs, C)
            ).view(B, G, gs, C)
            gathered = gathered * valid.view(1, G, gs, 1)
            grp_score = GROUP_SCORERS[cfg.saliency](gathered, valid)

        grp_rank = grp_score.argsort(dim=-1, descending=cfg.descending)   # (B, G)
        ranked = group_raster[grp_rank.reshape(-1)].reshape(B, G, gs)

        if cfg.layout == "interleaved":
            flat = ranked.permute(0, 2, 1).reshape(B, G * gs)
        elif cfg.layout == "grouped":
            flat = ranked.reshape(B, G * gs)
        else:
            raise ValueError(f"unknown layout {cfg.layout!r}")

        # Every row drops exactly the same number of pad slots, so the masked
        # select reshapes cleanly back to (B, N).
        perm = flat[flat >= 0].view(B, N).contiguous()
        inv_perm = torch.argsort(perm, dim=1)

    return perm, inv_perm


# ─────────────────────────────────────────────────────────────────────────────
# Runtime patching of PiToMe.algo.sparsesam.sam (no file under PiToMe/ changes)
# ─────────────────────────────────────────────────────────────────────────────

_ACTIVE_CFG: Optional[SaliencyConfig] = None
_ORIGINAL_FN = None


def _sam_module():
    import importlib
    return importlib.import_module("PiToMe.algo.sparsesam.sam")


def install(cfg: SaliencyConfig):
    """Point SparseSAM's attention at the configurable permutation."""
    global _ACTIVE_CFG, _ORIGINAL_FN
    mod = _sam_module()
    if _ORIGINAL_FN is None:
        _ORIGINAL_FN = mod.tile_stride_matching
    _ACTIVE_CFG = cfg
    mod.tile_stride_matching = tile_stride_matching_saliency
    return cfg


def set_config(cfg: SaliencyConfig):
    """Switch configs without re-installing (perm caches must be cleared)."""
    global _ACTIVE_CFG
    _ACTIVE_CFG = cfg
    return cfg


def active_config() -> Optional[SaliencyConfig]:
    return _ACTIVE_CFG


def restore():
    """Put the original `tile_stride_matching` back."""
    global _ACTIVE_CFG, _ORIGINAL_FN
    if _ORIGINAL_FN is not None:
        _sam_module().tile_stride_matching = _ORIGINAL_FN
        _ORIGINAL_FN = None
    _ACTIVE_CFG = None


@contextlib.contextmanager
def saliency_permutation(cfg: SaliencyConfig):
    install(cfg)
    try:
        yield cfg
    finally:
        restore()


def clear_perm_caches(encoder) -> None:
    """Drop cached permutations so a new config takes effect immediately.

    `sparsesam.sam.apply_patch` resets `perm_cache` on every encoder forward,
    but the cache key is (win, ratio, n_block) — it does not include the
    saliency config, so stale entries must be dropped when sweeping.
    """
    info = getattr(encoder, "tome_info", None)
    if isinstance(info, dict):
        info["perm_cache"] = {}
    for module in encoder.modules():
        minfo = getattr(module, "_tome_info", None)
        if isinstance(minfo, dict):
            minfo["perm_cache"] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Self-check: every (saliency, group_size, layout) must yield a valid permutation
# ─────────────────────────────────────────────────────────────────────────────

def self_test(device: str = "cpu", windows=(14, 64),
              group_sizes=(2, 4, 8, 16), saliencies=None,
              layouts=("grouped", "interleaved"), curves=("z", "hilbert")) -> int:
    """Validate permutations on random tensors. Returns the number of checks."""
    saliencies = saliencies or SALIENCY_CHOICES
    dev = torch.device(device)
    checks = 0
    for win in windows:
        N = win * win
        x = torch.randn(2, N, 32, device=dev)
        for gs in group_sizes:
            for curve in curves:
                for layout in layouts:
                    for sal in saliencies:
                        cfg = SaliencyConfig(saliency=sal, group_size=gs,
                                             curve=curve, layout=layout)
                        perm, inv = tile_stride_matching_saliency(
                            x, win, win, ratio=0.5, cfg=cfg
                        )
                        assert perm.shape == (2, N), (cfg, perm.shape)
                        for b in range(perm.shape[0]):
                            got = torch.sort(perm[b]).values
                            assert torch.equal(
                                got, torch.arange(N, device=dev)
                            ), f"{cfg.tag()} win={win}: not a permutation"
                            assert torch.equal(
                                perm[b][inv[b]], torch.arange(N, device=dev)
                            ), f"{cfg.tag()} win={win}: bad inverse"
                        checks += 1
    return checks


if __name__ == "__main__":
    n = self_test(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"OK — {n} (saliency × group_size × curve × layout × window) configs "
          f"produce valid permutations")
