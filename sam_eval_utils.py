"""Shared SAM evaluation helpers.

Compat shims around `PiToMe.algo.registry`'s SAM side, plus a couple of
small utilities (`reset_memory`, the sparsesam-family set) used by both
`tasks/sam_hq44k/eval_hq44k.py` and `tasks/sam_coco/eval_coco.py`. Keeping
these here means each task script can stand alone without cross-folder
imports.

The algorithm registry itself lives in `PiToMe/algo/registry.py` — see
`docs/ADDING_ALGORITHMS.md` for the walkthrough on adding new patches.
"""

import os
import sys
import gc

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'PiToMe'))

from algo.registry import (  # noqa: E402  (sys.path mutation above)
    SAM_REGISTRY, sam_algo_choices,
    apply_sam, remove_all_sam, update_sam_ratio,
)


# Back-compat aliases. New code should call `apply_sam` / `remove_all_sam`
# directly from `algo.registry`.
VALID_ALGOS = sam_algo_choices()
ALGO_REGISTRY = {name: (spec.apply, name) for name, spec in SAM_REGISTRY.items()}

# Sparsesam-family algos use a banded-diagonal sparse-attn mask whose width
# is controlled by `--diagonal-width`. Derived from the registry so it stays
# in sync as algorithms are added/removed.
SPARSESAM_ALGOS = {n for n in SAM_REGISTRY if n.startswith("sparsesam")}


def apply_tome(encoder, algo: str, ratio: float, margin: float = 0.5,
               mlp_merge: bool = True):
    """Patch `encoder` via the central SAM registry. `algo='none'` or
    `ratio>=1.0` are no-ops.

    `mlp_merge` is consumed by sparsesam-family patches that have a
    partial-MLP path (`PiToMe/algo/sparsesam/sam.py::ToMeSAMBlock`); other
    patches silently ignore it via the registry's signature-aware kwargs
    filter."""
    if algo == "none" or ratio >= 1.0:
        return
    return apply_sam(encoder, algo, ratio=ratio, margin=margin,
                     mlp_merge=mlp_merge)


def update_ratio(encoder, ratio: float):
    """Update merge ratio without re-patching (works if already patched)."""
    update_sam_ratio(encoder, ratio)


def remove_tome(encoder, mask_decoder=None):
    """Undo any token-merging patch on `encoder`."""
    return remove_all_sam(encoder, mask_decoder=mask_decoder)


def reset_memory():
    """Release CUDA caches + reset peak-memory counter between sweep configs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
