#!/usr/bin/env python3
"""Stage-wise SparseSAM ablation on HQ44K.

Runs attention-only, MLP-only, and attention+MLP routing on early/middle/late
encoder blocks, plus a dense baseline and per-stage sparsity profiles.

MLP keep-token routing is selectable with `--mlp-route`:
    rank      the shipped behaviour — top-k by attention-permutation rank
              (sam.py:400-411), running the FULL MLP when no global
              permutation is available yet, exactly as sam.py:412-413 does.
              With --prime-perm (default) the permutation is also computed on
              global blocks whose attention runs dense, so `mlp_only` uses the
              same selection rule as the shipped method instead of a surrogate.
    saliency  top-k by Sobel gradient magnitude of the block's own tokens,
              which needs no attention information and works at any depth
    uniform   fixed uniform stride over the Z-order (importance-free lower
              bound)

The attention mask itself is left exactly as the paper's pipeline builds it —
`make_A_mask` is untouched, so local (windowed) blocks keep band_width=0 and
their block-granularity density. `mean_attn_density` reports what that comes
out to, since it is not equal to the nominal keep ratio.

Every run reports mIoU, boundary IoU, per-image encoder latency and peak
memory, so accuracy and latency can be read off the same table. Always run
with `--include-baseline`: the dense row is what the sparse rows must be
compared against, and comparing across scripts is not valid (different
`hq_token_only`, different image counts).
"""

import argparse
import csv
import gc
import math
import sys
import time
import types
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SAM_HQ_ROOT = ROOT / "sam-hq"
PITOME_ROOT = ROOT / "PiToMe"
for path in (SAM_HQ_ROOT, ROOT, PITOME_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from segment_anything import SamPredictor, sam_model_registry  # noqa: E402
from segment_anything.modeling.image_encoder import (  # noqa: E402
    Attention,
    Block,
    window_partition,
    window_unpartition,
)
from PiToMe.algo.sparsesam.sam import (  # noqa: E402
    ToMeSAMAttention,
    _DIAG_BAND_WIDTH,
    _FA2_M_BLOCK_GLOBAL,
    _FA2_N_BLOCK_GLOBAL,
    _FA2_THREADS_GLOBAL,
    _FA2_N_BLOCK_LOCAL,
    _KEEP_BAR_SCALE,
    tile_stride_matching,
    tile_stride_matching_uniform,
)
import sparsesam_saliency as sal  # noqa: E402
from data_utils import OnlineDataset  # noqa: E402
from train.train import compute_boundary_iou, compute_iou  # noqa: E402
from train.utils.dataloader import Resize, get_im_gt_name_dict  # noqa: E402
import train.utils.misc as misc  # noqa: E402


DATASETS = [
    {
        "name": "DIS5K-VD",
        "im_dir": "./data/DIS5K/DIS-VD/im",
        "gt_dir": "./data/DIS5K/DIS-VD/gt",
        "im_ext": ".jpg",
        "gt_ext": ".png",
    },
    {
        "name": "COIFT",
        "im_dir": "./data/thin_object_detection/COIFT/images",
        "gt_dir": "./data/thin_object_detection/COIFT/masks",
        "im_ext": ".jpg",
        "gt_ext": ".png",
    },
    {
        "name": "ThinObject5k-TE",
        "im_dir": "./data/thin_object_detection/ThinObject5K/images_test",
        "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_test",
        "im_ext": ".jpg",
        "gt_ext": ".png",
    },
    {
        "name": "HRSOD",
        "im_dir": "./data/thin_object_detection/HRSOD/images",
        "gt_dir": "./data/thin_object_detection/HRSOD/masks",
        "im_ext": ".jpg",
        "gt_ext": ".png",
    },
]

CHECKPOINTS = {
    "vit_b": ROOT / "ckts/sam_hq_vit_b.pth",
    "vit_l": ROOT / "ckts/sam_hq_vit_l.pth",
    "vit_h": ROOT / "ckts/sam_hq_vit_h.pth",
}

SETTING_NAMES = {
    "attention": "attention_only",
    "mlp": "mlp_only",
    "attn_mlp": "attention_mlp",
}


def block_mask_density(n_tokens: int, ratio: float, n_block: int, band_width: int) -> float:
    """Fraction of column-blocks the cute kernel actually attends.

    Exact replica of `sam.py::make_A_mask` arithmetic. The nominal `ratio` is
    NOT the realised density: a 14x14 window is only 4 column-blocks wide, so
    block-granularity rounding pushes local attention well above `ratio`.
    """
    num_m = math.ceil(n_tokens / n_block)
    num_n = math.ceil(n_tokens / n_block)
    t = np.zeros((num_m, num_n), dtype=np.float32)

    half = band_width // 2
    for k in range(-half, band_width - half):
        if k == 0:
            i = np.arange(min(num_m, num_n))
            t[i, i] = 1.0
        else:
            i = np.arange(max(0, -k), min(num_m, num_n - k))
            t[i, i + k] = 1.0

    n_keep_cols = int(ratio * num_n * _KEEP_BAR_SCALE)
    if n_keep_cols > 0:
        t[:, : (n_keep_cols - band_width + 1)] = 1.0
    return float(t.mean())




def enable_half_image_encoder_for_predictor(predictor: SamPredictor):
    predictor.model.image_encoder.half()
    encoder_dtype = next(predictor.model.image_encoder.parameters()).dtype
    decoder_dtype = next(predictor.model.mask_decoder.parameters()).dtype

    @torch.no_grad()
    def set_torch_image_half_encoder(transformed_image: torch.Tensor, original_image_size):
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == predictor.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {predictor.model.image_encoder.img_size}."
        predictor.reset_image()
        predictor.original_size = original_image_size
        predictor.input_size = tuple(transformed_image.shape[-2:])
        input_image = predictor.model.preprocess(transformed_image).to(dtype=encoder_dtype)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        features, interm_features = predictor.model.image_encoder(input_image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        predictor._last_encoder_ms = (time.perf_counter() - t0) * 1000
        predictor.features = features.to(dtype=decoder_dtype)
        predictor.interm_features = [
            feat.to(dtype=decoder_dtype) if torch.is_floating_point(feat) else feat
            for feat in interm_features
        ]
        predictor.is_image_set = True

    predictor.set_torch_image = set_torch_image_half_encoder

def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        if key == "ori_im":
            collated[key] = [item[key] for item in batch]
        elif key in ("ori_im_path", "ori_gt_path"):
            collated[key] = [item[key] for item in batch]
        else:
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except Exception:
                collated[key] = [item[key] for item in batch]
    return collated


def build_dataloader(dataset_config: Dict, num_workers: int):
    valid_im_gt_list = get_im_gt_name_dict([dataset_config], flag="valid")
    dataset = OnlineDataset(
        [valid_im_gt_list[0]],
        transform=transforms.Compose([Resize([1024, 1024])]),
        eval_ori_resolution=True,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn,
    )


def stage_block_indices(num_blocks: int, stage: str) -> Set[int]:
    parts = np.array_split(np.arange(num_blocks), 3)
    names = {"early": 0, "middle": 1, "late": 2}
    return {int(i) for i in parts[names[stage]].tolist()}


def _keep_token_count(n_tokens: int, ratio: float) -> int:
    """Same keep-count for every routing mode, so `--mlp-route` compares
    selection *quality* rather than selection *size*."""
    return max(1, min(int(round(ratio * n_tokens)), n_tokens))


def position_keep_indices(x_seq: torch.Tensor, h: int, w: int, ratio: float, group_size: int):
    """Importance-free baseline: uniform stride over the Z-order."""
    perm, _ = tile_stride_matching_uniform(
        x_seq, h, w, ratio=ratio, group_size=group_size, n_block=_FA2_N_BLOCK_GLOBAL,
    )
    return perm[:, :_keep_token_count(x_seq.shape[1], ratio)]


def saliency_keep_indices(x_seq: torch.Tensor, h: int, w: int, ratio: float,
                          group_size: int, saliency: str = "sobel3"):
    """Top-k Z-order groups by Sobel gradient magnitude of the block's own
    tokens. Needs no attention information, so it is available at any depth —
    which is why the uniform fallback is not actually necessary in early
    blocks."""
    cfg = sal.SaliencyConfig(saliency=saliency, group_size=group_size,
                             curve="z", layout="grouped")
    perm, _ = sal.tile_stride_matching_saliency(x_seq, h, w, ratio=ratio, cfg=cfg)
    return perm[:, :_keep_token_count(x_seq.shape[1], ratio)]


def rank_keep_indices(x_seq: torch.Tensor, ratio: float, attn_info: Dict,
                      h: int, num_heads: int) -> Optional[torch.Tensor]:
    """Shipped routing (sam.py:400-411): top-k by mean attention-permutation
    rank. Returns None when no global permutation has been cached yet."""
    cached = attn_info.get("perm_cache", {}).get((h, ratio, _FA2_N_BLOCK_GLOBAL))
    if cached is None:
        return None
    _, g_inv_perm = cached
    b = x_seq.shape[0]
    avg_rank = g_inv_perm.view(b, num_heads, -1).float().mean(dim=1)
    keep_n = _keep_token_count(x_seq.shape[1], ratio)
    return avg_rank.topk(keep_n, dim=1, largest=False).indices


class StageAblationSparseSAMBlock(Block):
    def _run_attention(self, x: torch.Tensor, use_sparse: bool, ratio: float) -> torch.Tensor:
        x_norm = self.norm1(x)
        if self.window_size > 0:
            h, w = x_norm.shape[1], x_norm.shape[2]
            x_win, pad_hw = window_partition(x_norm, self.window_size)
            if use_sparse:
                x_attn = self.attn(x_win, ratio=ratio)
            else:
                x_attn = Attention.forward(self.attn, x_win)
            return window_unpartition(x_attn, self.window_size, pad_hw, (h, w))
        if use_sparse:
            return self.attn(
                x_norm,
                ratio=ratio,
                m_block=_FA2_M_BLOCK_GLOBAL,
                n_block=_FA2_N_BLOCK_GLOBAL,
                threads=_FA2_THREADS_GLOBAL,
                is_global=True,
            )
        return Attention.forward(self.attn, x_norm)

    def _prime_global_perm(self, x_norm: torch.Tensor, ratio: float) -> None:
        """Populate the global permutation cache on a block whose attention is
        running dense.

        The shipped pipeline gets this perm as a by-product of sparse global
        attention (sam.py:328). In `mlp_only` the attention is dense, so the
        cache would stay empty and MLP routing would silently degrade to a
        full MLP. Computing it here keeps the *selection rule* identical to the
        paper. Costs one extra qkv projection on global blocks only, which
        inflates the measured encoder latency of those rows slightly.
        """
        info = self.attn._tome_info
        cache = info.setdefault("perm_cache", {})
        b, h, w, c = x_norm.shape
        key = (h, ratio, _FA2_N_BLOCK_GLOBAL)
        if key in cache:
            return
        nh = self.attn.num_heads
        d = c // nh
        qkv = self.attn.qkv(x_norm.reshape(b, h * w, c))
        k = (qkv.view(b, h * w, 3, nh, d)
                .permute(2, 0, 3, 1, 4)
                .reshape(3, b * nh, h * w, d))[1]
        cache[key] = tile_stride_matching(k, h, w, ratio=ratio,
                                          n_block=_FA2_N_BLOCK_GLOBAL)

    def _mlp_keep_indices(self, x_seq: torch.Tensor, h: int, w: int,
                          ratio: float, info: Dict) -> Optional[torch.Tensor]:
        """None → run the full MLP, exactly as sam.py:412-413 does when no
        global permutation is available."""
        route = info["mlp_route"]
        if route == "rank":
            return rank_keep_indices(x_seq, ratio, self.attn._tome_info, h,
                                     self.attn.num_heads)
        if route == "saliency":
            return saliency_keep_indices(x_seq, h, w, ratio,
                                         info["group_size"], info["saliency"])
        return position_keep_indices(x_seq, h, w, ratio, info["group_size"])

    def _run_routed_mlp(self, x: torch.Tensor, ratio: float, info: Dict) -> Optional[torch.Tensor]:
        b, h, w, c = x.shape
        x_seq = x.reshape(b, h * w, c)
        keep_idx = self._mlp_keep_indices(x_seq, h, w, ratio, info)
        if keep_idx is None:
            return None
        idx_e = keep_idx.unsqueeze(-1).expand(-1, -1, c)
        x_kept = x_seq.gather(1, idx_e)
        x_kept = x_kept + self.mlp(self.norm2(x_kept))
        x_seq = x_seq.scatter(1, idx_e, x_kept)
        return x_seq.reshape(b, h, w, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        info = self._stage_ablation_info
        block_idx = self._stage_ablation_idx
        selected = block_idx in info["stage_blocks"]
        setting = info["setting"]
        ratio = info["ratio_per_block"].get(block_idx, info["ratio"])

        use_sparse_attn = selected and ratio < 1.0 and setting in ("attention", "attn_mlp")
        use_sparse_mlp = selected and ratio < 1.0 and setting in ("mlp", "attn_mlp")
        # The shipped method never routes the MLP on global-attention blocks
        # (sam.py:399 gates on window_size > 0); mirror that unless asked not to.
        if use_sparse_mlp and self.window_size == 0 and not info["mlp_on_global"]:
            use_sparse_mlp = False

        # Keep the perm cache in the same state the shipped pipeline would be
        # in: every global block contributes it, whether or not this stage runs
        # sparse attention. Only the first one actually computes it.
        needs_perm = (info["mlp_route"] == "rank"
                      and setting in ("mlp", "attn_mlp")
                      and info["prime_perm"])
        if needs_perm and self.window_size == 0 and not use_sparse_attn and ratio < 1.0:
            self._prime_global_perm(self.norm1(x), ratio)

        x = x + self._run_attention(x, use_sparse_attn, ratio)
        if use_sparse_mlp:
            routed = self._run_routed_mlp(x, ratio, info)
            x = routed if routed is not None else x + self.mlp(self.norm2(x))
        else:
            x = x + self.mlp(self.norm2(x))
        return x


def apply_stage_ablation_patch(encoder, setting: str, stage: str, ratio: float,
                               group_size: int, mlp_route: str = "rank",
                               saliency: str = "sobel3",
                               mlp_on_global: bool = False,
                               prime_perm: bool = True,
                               ratio_per_block: Optional[Dict[int, float]] = None):
    n_blocks = len(encoder.blocks)
    selected = (set(range(n_blocks)) if stage == "mixed"
                else stage_block_indices(n_blocks, stage))
    # One shared attn-state dict, exactly as sam.py::apply_patch does — the
    # `rank` MLP route reads the global permutation out of this cache.
    attn_info = {"ratio": ratio, "perm_cache": {}}
    info = {
        "setting": setting,
        "stage": stage,
        "stage_blocks": selected,
        "ratio": float(ratio),
        "ratio_per_block": dict(ratio_per_block or {}),
        "group_size": int(group_size),
        "mlp_route": mlp_route,
        "saliency": saliency,
        "mlp_on_global": bool(mlp_on_global),
        "prime_perm": bool(prime_perm),
        "attn_info": attn_info,
    }
    encoder._stage_ablation_info = info

    for idx, block in enumerate(encoder.blocks):
        if not isinstance(block, StageAblationSparseSAMBlock):
            block.__class__ = StageAblationSparseSAMBlock
        block._stage_ablation_info = info
        block._stage_ablation_idx = idx
        if not isinstance(block.attn, ToMeSAMAttention):
            block.attn.__class__ = ToMeSAMAttention
        block.attn._tome_info = attn_info

    # Reset the permutation cache on every encoder forward. Without this the
    # ordering computed for the first image is silently reused for the whole
    # split (sam.py::apply_patch installs the same reset at line 513).
    if "forward" not in encoder.__dict__:
        _orig_forward = encoder.__class__.forward

        def _patched_forward(self, x):
            self._stage_ablation_info["attn_info"]["perm_cache"] = {}
            return _orig_forward(self, x)

        encoder.forward = types.MethodType(_patched_forward, encoder)
    return selected


def remove_stage_ablation_patch(encoder):
    for block in encoder.blocks:
        if isinstance(block, StageAblationSparseSAMBlock):
            block.__class__ = Block
        block.__dict__.pop("_stage_ablation_info", None)
        block.__dict__.pop("_stage_ablation_idx", None)
        if isinstance(block.attn, ToMeSAMAttention):
            block.attn.__class__ = Attention
        block.attn.__dict__.pop("_tome_info", None)
    encoder.__dict__.pop("_stage_ablation_info", None)
    encoder.__dict__.pop("forward", None)


@torch.no_grad()
def evaluate_dataset(predictor: SamPredictor, dataset_config: Dict, num_samples: int,
                     num_workers: int, warmup_batches: int = 3) -> Dict:
    """Returns accuracy + latency for one (config, dataset).

    `hq_token_only=True` scores the HQ branch alone, matching sam-hq's own
    `evaluate()` (train/train.py:583) and tasks/sam_hq44k/eval_hq44k.py. The
    default (False) would score `masks_sam + masks_hq` — a different quantity,
    not comparable with either.
    """
    dataloader = build_dataloader(dataset_config, num_workers)
    device = predictor.device

    # Warm up: the first sparse config otherwise pays cute JIT-compile cost
    # inside the timed loop.
    warm_iter = iter(dataloader)
    for _ in range(warmup_batches):
        try:
            data = next(warm_iter)
        except StopIteration:
            break
        image = data["image"].to(device)
        predictor.set_torch_image(image, (image.shape[2], image.shape[3]))
        predictor.reset_image()
    del warm_iter
    reset_memory()

    ious, b_ious, enc_ms = [], [], []
    processed = 0
    t_overall = time.perf_counter()
    for data in tqdm(dataloader, total=min(len(dataloader), num_samples), desc=dataset_config["name"], leave=False):
        if processed >= num_samples:
            break
        image = data["image"].to(device)
        label = data["label"].to(device)
        label_ori = data["ori_label"].to(device)
        if label.dim() == 4:
            boxes = misc.masks_to_boxes(label[:, 0, :, :]).to(device)
        else:
            boxes = misc.masks_to_boxes(label[0:1, :, :]).to(device)

        predictor.set_torch_image(image, (image.shape[2], image.shape[3]))
        enc_ms.append(predictor._last_encoder_ms)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes,
            multimask_output=False,
            hq_token_only=True,
        )
        ious.append(compute_iou(masks.float(), label_ori).item())
        b_ious.append(compute_boundary_iou(masks.float(), label_ori).item())
        predictor.reset_image()
        processed += image.shape[0]

    overall_sec = time.perf_counter() - t_overall
    mem = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem = {
            "peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
            "peak_memory_reserved_mb": torch.cuda.max_memory_reserved() / 1024 ** 2,
        }

    return {
        "miou": float(np.mean(ious)) if ious else 0.0,
        "miou_std": float(np.std(ious)) if ious else 0.0,
        "boundary_iou": float(np.mean(b_ious)) if b_ious else 0.0,
        "boundary_iou_std": float(np.std(b_ious)) if b_ious else 0.0,
        "encoder_per_image_mean_ms": float(np.mean(enc_ms)) if enc_ms else 0.0,
        "encoder_per_image_std_ms": float(np.std(enc_ms)) if enc_ms else 0.0,
        "throughput_imgs_per_sec": processed / overall_sec if overall_sec else 0.0,
        "num_images": processed,
        "elapsed_sec": overall_sec,
        **mem,
    }


def save_rows(rows: List[Dict], output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_stage_ratios(spec: str) -> Dict[str, float]:
    """'early=0.75,middle=0.5,late=0.25' → per-stage keep ratios."""
    out = {}
    for part in spec.split(","):
        name, _, value = part.partition("=")
        name = name.strip()
        if name not in ("early", "middle", "late"):
            raise argparse.ArgumentTypeError(f"unknown stage {name!r} in {spec!r}")
        out[name] = float(value)
    missing = {"early", "middle", "late"} - set(out)
    if missing:
        raise argparse.ArgumentTypeError(f"missing stages {sorted(missing)} in {spec!r}")
    return out


def stage_density_columns(encoder, stage_blocks: Set[int],
                          ratio_per_block: Dict[int, float], ratio: float) -> Dict:
    """Realised attention density and global-block count for the selected set.

    The global-block count matters: equal thirds do NOT split global-attention
    blocks evenly (vit_l → early [5], middle [11], late [17, 23]), so the late
    stage receives twice the sparsification of the others.
    """
    n_global = sum(1 for i in stage_blocks if encoder.blocks[i].window_size == 0)
    densities = []
    for i in sorted(stage_blocks):
        blk = encoder.blocks[i]
        r = ratio_per_block.get(i, ratio)
        if r >= 1.0:
            densities.append(1.0)
            continue
        if blk.window_size > 0:
            n_tok, n_blk, band = blk.window_size ** 2, _FA2_N_BLOCK_LOCAL, 0
        else:
            n_tok = encoder.img_size // encoder.patch_embed.proj.stride[0]
            n_tok, n_blk, band = n_tok ** 2, _FA2_N_BLOCK_GLOBAL, _DIAG_BAND_WIDTH
        densities.append(block_mask_density(n_tok, r, n_blk, band))
    return {
        "n_global_blocks_in_stage": n_global,
        "n_local_blocks_in_stage": len(stage_blocks) - n_global,
        "mean_attn_density": float(np.mean(densities)) if densities else 1.0,
    }


def run_model(args, model_type: str):
    checkpoint = Path(args.checkpoint) if args.checkpoint else CHECKPOINTS[model_type]
    print(f"Loading {model_type}: {checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint)).to(args.device)
    sam.eval()
    predictor = SamPredictor(sam)
    enable_half_image_encoder_for_predictor(predictor)

    encoder = predictor.model.image_encoder
    n_blocks = len(encoder.blocks)
    output_csv = Path(args.output_dir) / f"sparsesam_stage_ablation_hq44k_{model_type}.csv"
    rows: List[Dict] = []

    def record(setting_name, stage, selected, ratio_kw, ratio_per_block, dataset_config, res):
        rows.append({
            "model_type": model_type,
            "checkpoint": str(checkpoint),
            "setting": setting_name,
            "stage": stage,
            "mlp_route": args.mlp_route if setting_name != "dense_baseline" else "none",
            "saliency": args.saliency if setting_name != "dense_baseline" else "none",
            "mlp_on_global": args.mlp_on_global,
            "prime_perm": args.prime_perm,
            "hq_token_only": True,
            "dataset": dataset_config["name"],
            "ratio_keep": ratio_kw,
            "sparsity": 1.0 - ratio_kw if isinstance(ratio_kw, float) else "",
            "ratio_per_stage": (";".join(f"{k}={v}" for k, v in ratio_per_block.items())
                                if isinstance(ratio_per_block, dict) and ratio_per_block else ""),
            "num_blocks": n_blocks,
            "stage_blocks": " ".join(str(i) for i in sorted(selected)),
            "group_size": args.group_size,
            **stage_density_columns(encoder, selected,
                                    ratio_per_block if isinstance(ratio_per_block, dict) else {},
                                    ratio_kw if isinstance(ratio_kw, float) else 1.0),
            **res,
        })
        save_rows(rows, output_csv)

    # ── dense baseline: the only valid reference for the sparse rows ────────
    if args.include_baseline:
        remove_stage_ablation_patch(encoder)
        print(f"{model_type} dense_baseline")
        for dataset_config in DATASETS:
            reset_memory()
            res = evaluate_dataset(predictor, dataset_config,
                                   num_samples=args.num_samples,
                                   num_workers=args.num_workers)
            record("dense_baseline", "none", set(), 1.0, {}, dataset_config, res)
            print(f"  [{dataset_config['name']}] mIoU={res['miou']:.4f} "
                  f"B-IoU={res['boundary_iou']:.4f} enc/img={res['encoder_per_image_mean_ms']:.1f}ms")

    # ── uniform-sparsity stage runs ─────────────────────────────────────────
    for setting in (args.settings if not args.no_stage_sweep else []):
        for stage in args.stages:
            selected = apply_stage_ablation_patch(
                encoder, setting=setting, stage=stage, ratio=args.ratio,
                group_size=args.group_size, mlp_route=args.mlp_route,
                saliency=args.saliency, mlp_on_global=args.mlp_on_global,
                prime_perm=args.prime_perm,
            )
            print(f"{model_type} {SETTING_NAMES[setting]} {stage}: blocks={sorted(selected)}")
            for dataset_config in DATASETS:
                reset_memory()
                res = evaluate_dataset(predictor, dataset_config,
                                       num_samples=args.num_samples,
                                       num_workers=args.num_workers)
                record(SETTING_NAMES[setting], stage, selected, args.ratio, {},
                       dataset_config, res)
                print(f"  [{dataset_config['name']}] mIoU={res['miou']:.4f} "
                      f"B-IoU={res['boundary_iou']:.4f} enc/img={res['encoder_per_image_mean_ms']:.1f}ms")
            remove_stage_ablation_patch(encoder)

    # ── layer-specific density: one ratio per stage, all blocks active ──────
    for spec in (args.stage_ratios or []):
        profile = parse_stage_ratios(spec)
        per_block = {}
        for stage_name, r in profile.items():
            for i in stage_block_indices(n_blocks, stage_name):
                per_block[i] = r
        for setting in args.settings:
            selected = apply_stage_ablation_patch(
                encoder, setting=setting, stage="mixed", ratio=args.ratio,
                group_size=args.group_size, mlp_route=args.mlp_route,
                saliency=args.saliency, mlp_on_global=args.mlp_on_global,
                prime_perm=args.prime_perm, ratio_per_block=per_block,
            )
            print(f"{model_type} {SETTING_NAMES[setting]} mixed[{spec}]")
            for dataset_config in DATASETS:
                reset_memory()
                res = evaluate_dataset(predictor, dataset_config,
                                       num_samples=args.num_samples,
                                       num_workers=args.num_workers)
                record(SETTING_NAMES[setting], f"mixed[{spec}]", selected, "",
                       profile, dataset_config, res)
                print(f"  [{dataset_config['name']}] mIoU={res['miou']:.4f} "
                      f"B-IoU={res['boundary_iou']:.4f} enc/img={res['encoder_per_image_mean_ms']:.1f}ms")
            remove_stage_ablation_patch(encoder)

    save_rows(rows, output_csv)
    print(f"Saved {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="SparseSAM stage ablation on HQ44K.")
    parser.add_argument("--model-types", nargs="+", default=["vit_b"], choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint for single-model runs.")
    parser.add_argument("--settings", nargs="+", default=["attention", "mlp", "attn_mlp"], choices=["attention", "mlp", "attn_mlp"])
    parser.add_argument("--stages", nargs="+", default=["early", "middle", "late"], choices=["early", "middle", "late"])
    parser.add_argument("--ratio", type=float, default=0.5, help="Token keep ratio. 0.5 means 50%% sparsity.")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--mlp-route", default="rank", choices=["rank", "saliency", "uniform"],
                        help="MLP keep-token selection. 'rank' = shipped behaviour\n"
                             "(falls back to 'saliency' before the first global block).")
    parser.add_argument("--saliency", default="sobel3", choices=sal.SALIENCY_CHOICES,
                        help="Score used by --mlp-route saliency.")
    parser.add_argument("--mlp-on-global", action=argparse.BooleanOptionalAction, default=False,
                        help="Route the MLP on global-attention blocks too. The shipped\n"
                             "method does not (sam.py:399), so default is off.")
    parser.add_argument("--prime-perm", action=argparse.BooleanOptionalAction, default=True,
                        help="With --mlp-route rank: compute the global permutation on\n"
                             "global blocks whose attention runs dense, so MLP-only uses\n"
                             "the same selection rule as the shipped method.")
    parser.add_argument("--include-baseline", action="store_true",
                        help="Evaluate the unpatched dense encoder in the same run.")
    parser.add_argument("--no-stage-sweep", action="store_true",
                        help="Skip the early/middle/late loop (use with --stage-ratios).")
    parser.add_argument("--stage-ratios", nargs="+", default=None,
                        help="Layer-specific density, e.g. early=0.75,middle=0.5,late=0.25 .\n"
                             "Repeatable; each spec adds a stage='mixed[...]' row set.")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(ROOT / "benchmark_results/stage_ablation_hq44k"))
    return parser.parse_args()


def main():
    args = parse_args()
    for model_type in args.model_types:
        run_model(args, model_type)


if __name__ == "__main__":
    main()
