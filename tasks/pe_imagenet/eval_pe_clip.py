#!/usr/bin/env python3
"""Evaluate Perception Encoder (PE) Core models on CLIP zero-shot benchmarks.

Wraps the existing perception_models/apps/pe/clip_benchmark machinery:
  - Loads a PE-Core CLIP model via core.vision_encoder.pe.CLIP.from_config
  - Builds datasets via clip_benchmark.datasets.builder.build_dataset
  - Runs zero-shot classification (or retrieval) and reports metrics

Examples:
  # Single dataset, default PE-Core-L14-336
  python eval_pe_clip.py --model PE-Core-L14-336 --dataset cifar10 --dataset-root ./data/cifar10

  # Several datasets in one run
  python eval_pe_clip.py --model PE-Core-B16-224 \
      --dataset cifar10 cifar100 imagenetv2 \
      --dataset-root './data/{dataset}'

  # Retrieval (COCO captions)
  python eval_pe_clip.py --model PE-Core-L14-336 --dataset mscoco_captions \
      --dataset-root ./data/coco --task zeroshot_retrieval

List supported PE configs:
  python eval_pe_clip.py --list-models
"""

import os
import sys
import csv
import json
import time
import argparse
import datetime
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
PM_ROOT = os.path.join(_REPO, "perception_models")
PM_PE   = os.path.join(PM_ROOT, "apps", "pe")
sys.path.insert(0, _REPO)                              # shared utils at repo root
sys.path.insert(0, os.path.join(_REPO, "PiToMe"))      # PE algo registry
sys.path.insert(0, PM_ROOT)                            # for `core.*`
sys.path.insert(0, PM_PE)                              # for `clip_benchmark.*`

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transforms

from clip_benchmark.datasets.builder import (
    build_dataset, get_dataset_collate_fn, get_dataset_default_task,
)
from contextlib import suppress
from tqdm import tqdm
import torch.nn.functional as F
from clip_benchmark.metrics import zeroshot_classification
# zeroshot_retrieval imports audio-visual code (needs xformers); import lazily when used.

# Patch upstream accuracy(): float(np.ndarray) breaks on NumPy >=2 / >=1.25 for 1-elem arrays.
def _accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [correct[:k].reshape(-1).float().sum().item() / n for k in topk]
zeroshot_classification.accuracy = _accuracy


def _run_classification_with_acc_bar(model, classifier, dataloader, device,
                                     video_dataset=False, amp=True, args=None,
                                     desc="zeroshot"):
    """Drop-in replacement for `zeroshot_classification.run_classification` that
    updates a tqdm postfix with running top-1 (top-5 when ≥5 classes)."""
    autocast = torch.cuda.amp.autocast if amp else suppress
    pred_chunks, true_chunks = [], []
    n_classes = classifier.shape[1] if classifier.ndim == 2 else 0
    track_top5 = n_classes >= 5
    seen, c1, c5 = 0, 0, 0

    bar = tqdm(dataloader, desc=desc)
    with torch.no_grad():
        for images, target in bar:
            if isinstance(images, torch.Tensor):
                images = images.to(device, torch.float32)
            elif isinstance(images, (list, tuple)):  # video frames
                images = [x.to(device, torch.float32) for x in images]
                images = torch.stack(images, dim=0).permute(1, 0, 2, 3, 4).contiguous()
            else:
                raise NotImplementedError
            target = target.to(device)

            with autocast():
                if video_dataset:
                    image_features = model.encode_video(images)
                else:
                    image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100.0 * image_features @ classifier

            # Running accuracy (skip if multi-label — shape (B, C) — since argmax/top-k aren't meaningful).
            if target.ndim == 1:
                k = 5 if track_top5 else 1
                top = logits.topk(k, dim=1).indices
                correct = top.eq(target.view(-1, 1))
                c1 += correct[:, 0].sum().item()
                if track_top5:
                    c5 += correct.any(dim=1).sum().item()
                seen += target.numel()
                postfix = {"acc1": f"{c1/seen*100:.2f}"}
                if track_top5:
                    postfix["acc5"] = f"{c5/seen*100:.2f}"
                bar.set_postfix(postfix)

            true_chunks.append(target.cpu())
            pred_chunks.append(logits.float().cpu())

    return torch.cat(pred_chunks), torch.cat(true_chunks)


# ──────────────────────────────────────────────────────────────────────────────
# PE adapter — clip_benchmark expects a tokenizer callable mapping list[str] -> Tensor
# ──────────────────────────────────────────────────────────────────────────────

class PETokenizerAdapter:
    """Wraps PE SimpleTokenizer so it is callable: list[str] -> Tensor[B, L]."""
    def __init__(self, context_length: int):
        self.tok = pe_transforms.get_text_tokenizer(context_length)

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.tok(texts)


def build_pe_model(name: str, checkpoint_path: str, device: str, dtype: torch.dtype):
    print(f"Loading PE model: {name} (ckpt={checkpoint_path or 'HF default'})")
    model = pe.CLIP.from_config(name, pretrained=True, checkpoint_path=checkpoint_path)
    model = model.to(device=device, dtype=dtype).eval()
    img_size = model.image_size
    ctx_len  = model.context_length if hasattr(model, "context_length") else 32
    transform = pe_transforms.get_image_transform(img_size)
    tokenizer = PETokenizerAdapter(ctx_len)
    return model, transform, tokenizer, img_size, ctx_len


# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(
    model, transform, tokenizer,
    dataset_name: str, dataset_root: str, split: str, task: str,
    batch_size: int, num_workers: int, device: str, amp: bool,
) -> Dict[str, float]:
    root = dataset_root.format(dataset=dataset_name) if "{dataset}" in dataset_root else dataset_root

    if task == "auto":
        task = get_dataset_default_task(dataset_name)
        print(f"  [auto] task = {task}")

    print(f"  Building dataset {dataset_name}  (root={root}, split={split}, task={task})")
    dataset = build_dataset(
        dataset_name=dataset_name, root=root, transform=transform,
        split=split, download=True, task=task,
    )
    collate_fn = get_dataset_collate_fn(dataset_name)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=collate_fn,
    )

    if task == "zeroshot_classification":
        # clip_benchmark's `evaluate` reads classnames/templates off the dataset object
        classnames = dataset.classes if hasattr(dataset, "classes") else None
        templates  = dataset.templates if hasattr(dataset, "templates") else None
        if classnames is None or templates is None:
            raise RuntimeError(f"Dataset {dataset_name} did not expose classnames/templates")
        # Swap upstream `run_classification` for one that surfaces running top-1/5
        # via tqdm postfix. Restored after the call so other code paths are untouched.
        from functools import partial as _partial
        _orig_run = zeroshot_classification.run_classification
        zeroshot_classification.run_classification = _partial(
            _run_classification_with_acc_bar, desc=f"{dataset_name}",
        )
        try:
            metrics = zeroshot_classification.evaluate(
                model, loader, tokenizer, classnames, templates, device, amp=amp, verbose=False,
            )
        finally:
            zeroshot_classification.run_classification = _orig_run
    elif task == "zeroshot_retrieval":
        # zeroshot_retrieval imports core.audio_visual_encoder.PEAudioVisual just for
        # an `isinstance` check. That chain pulls in xformers (often ABI-broken vs torch).
        # Stub the module so the import succeeds; PE-Core CLIP isn't a PEAudioVisual anyway.
        import types
        stub = types.ModuleType("core.audio_visual_encoder")
        class _PEAudioVisualStub: ...
        stub.PEAudioVisual = _PEAudioVisualStub
        stub.PEAudioFrame = type("PEAudioFrame", (), {})
        stub.PEAudioVisualOutput = type("PEAudioVisualOutput", (), {})
        sys.modules.setdefault("core.audio_visual_encoder", stub)
        from clip_benchmark.metrics import zeroshot_retrieval
        from types import SimpleNamespace
        from functools import partial
        retrieval_args = SimpleNamespace(reweight_retrieval=False)
        # The upstream `tqdm(dataloader_with_indices(loader))` has no total because
        # the wrapper is a generator. Inject total=len(loader) so the bar reflects batches.
        _orig_tqdm = zeroshot_retrieval.tqdm
        zeroshot_retrieval.tqdm = partial(
            _orig_tqdm, total=len(loader), desc=f"{dataset_name} retrieval"
        )
        try:
            metrics = zeroshot_retrieval.evaluate(
                model, loader, tokenizer, device=device, amp=amp,
                recall_k_list=[1, 5, 10], args=retrieval_args,
            )
        finally:
            zeroshot_retrieval.tqdm = _orig_tqdm
    else:
        raise ValueError(f"Unsupported task: {task}")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# PE algorithm dispatch — delegates to the central registry
# (`PiToMe/algo/registry.py`). To add a new algorithm, register it there;
# this module needs no changes.
# ──────────────────────────────────────────────────────────────────────────────

from PiToMe.algo.registry import (
    PE_REGISTRY, algo_choices as _pe_choices,
    apply_pe as _registry_apply_pe, remove_all_pe as _registry_remove_all,
)


def _warmup_rope(model, dummy):
    """rope.freq is built lazily inside VisionTransformer.forward_features.
    Run one dummy forward so cute kernels can pull cos/sin during patching."""
    with torch.no_grad():
        model.encode_image(dummy)


def apply_pe_patch(model, algo, ratio, args, dummy):
    """Apply a PE patch via the registry. Cute-kernel patches need rope.freq
    populated upfront; we warm it up unconditionally before applying."""
    if algo == "none":
        return
    if algo not in PE_REGISTRY:
        raise ValueError(f"Unknown algo: {algo}. Choices: {_pe_choices()}")
    _warmup_rope(model, dummy)
    _registry_apply_pe(model, algo, args=args, ratio=ratio)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate PE on CLIP benchmark datasets")
    p.add_argument("--model", default="PE-Core-L14-336",
                   help="PE config name (see --list-models)")
    p.add_argument("--checkpoint", default=None, help="Local checkpoint; if omitted, HF weights are downloaded")
    p.add_argument("--dataset", nargs="+", default=["cifar10"],
                   help="One or more dataset names (e.g. cifar10 cifar100 imagenetv2 mscoco_captions)")
    p.add_argument("--dataset-root", default="./data/{dataset}",
                   help="Root for datasets; supports '{dataset}' template")
    p.add_argument("--split", default="test")
    p.add_argument("--task", default="auto",
                   choices=["auto", "zeroshot_classification", "zeroshot_retrieval"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--no-amp", action="store_true",
                   help="Disable autocast inside clip_benchmark (still uses --dtype for weights)")
    p.add_argument("--algorithm", nargs="+", default=["none"],
                   choices=_pe_choices(),
                   help="One or more PE patches to evaluate (sweep). "
                        "Registered algorithms are listed automatically — see "
                        "`PiToMe/algo/registry.py` and `docs/ADDING_PE.md`.")
    p.add_argument("--ratio", nargs="+", type=float, default=[0.5],
                   help="Keep-bar / keep-fraction in (0, 1] to sweep; lower = more compression")
    # Stage-compression knobs (consumed by `_kw_compress`)
    p.add_argument("--num-stages", type=int, default=4,
                   help="(stage-compress) Number of stages; one merge step at each boundary.")
    p.add_argument("--group-size", type=int, default=4,
                   help="(sparsesam) Z-group size for tile-stride permute")
    p.add_argument("--use-flash-rope", action="store_true",
                   help="(stage-compress) Route attention through fused FA2+RoPE cute kernel.")
    p.add_argument("--compress-at-blocks", nargs="+", type=int, default=None,
                   help="(stage-compress) Explicit 0-indexed block indices after which "
                        "compression fires; overrides --num-stages.")
    # Partial knobs (consumed by `_kw_partial_basic` / `_kw_partial_sparsesam`)
    p.add_argument("--partial-start-block", type=int, default=0,
                   help="(partial) First block index at which the patch kicks in. "
                        "Earlier blocks run stock SDPA + full MLP.")
    p.add_argument("--sparse-ratio", type=float, default=None,
                   help="(sparsesam_partial) keep-bar width inside the cute mask; "
                        "defaults to --ratio.")
    p.add_argument("--mlp-merge", dest="mlp_merge", action="store_true", default=True,
                   help="(partial) Merge → MLP → unmerge sandwich (default ON).")
    p.add_argument("--no-mlp-merge", dest="mlp_merge", action="store_false",
                   help="(partial) Disable MLP merge — attention-only compression.")
    p.add_argument("--output-dir", default="./benchmark_results")
    p.add_argument("--list-models", action="store_true")
    args = p.parse_args()

    if args.list_models:
        print("Available PE configs:")
        for cfg in pe.CLIP.available_configs():
            print(f"  {cfg}")
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    model, transform, tokenizer, img_size, ctx_len = build_pe_model(
        args.model, args.checkpoint, args.device, dtype
    )
    print(f"  image_size={img_size}, context_length={ctx_len}, dtype={args.dtype}")

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build (algo, ratio) configs to sweep. Algos with `accepts_ratio=False`
    # (e.g. flash_rope) run once; the rest run once per --ratio value.
    configs: List[tuple] = []
    for algo in args.algorithm:
        spec = PE_REGISTRY.get(algo)
        if algo == "none" or (spec is not None and not spec.accepts_ratio):
            configs.append((algo, None))
        else:
            for r in args.ratio:
                configs.append((algo, float(r)))

    dummy = torch.zeros(1, 3, img_size, img_size, device=args.device, dtype=dtype)

    all_results: List[Dict] = []
    for algo, ratio in configs:
        tag = f"{algo}" + (f"@r={ratio}" if ratio is not None else "")
        print(f"\n############ config: {tag} ############")

        # Reset to baseline before applying the next patch.
        _registry_remove_all(model)

        if algo != "none":
            print(f"Applying PE algorithm: {algo}" + (f" (ratio={ratio})" if ratio is not None else ""))
            apply_pe_patch(model, algo=algo, ratio=ratio, args=args, dummy=dummy)

        for ds_name in args.dataset:
            print(f"\n=== [{tag}] {ds_name} ===")
            t0 = time.perf_counter()
            try:
                metrics = evaluate_dataset(
                    model, transform, tokenizer,
                    dataset_name=ds_name, dataset_root=args.dataset_root,
                    split=args.split, task=args.task,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    device=args.device, amp=not args.no_amp,
                )
            except Exception as e:
                print(f"  ERROR on {ds_name}: {repr(e)}")
                import traceback; traceback.print_exc()
                metrics = {"error": repr(e)}
            elapsed = time.perf_counter() - t0
            print(f"  took {elapsed:.1f}s  →  {json.dumps(metrics, default=str)}")
            all_results.append({
                "model": args.model, "algorithm": algo, "ratio": ratio,
                "dataset": ds_name, "split": args.split,
                "elapsed_s": elapsed, **metrics,
            })

    # CSV dump
    csv_path = os.path.join(args.output_dir, f"pe_clip_{args.model}_{ts}.csv")
    keys = sorted({k for r in all_results for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_results:
            w.writerow(r)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
