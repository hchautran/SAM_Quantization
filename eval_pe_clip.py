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

ROOT = os.path.dirname(os.path.abspath(__file__))
PM_ROOT = os.path.join(ROOT, "perception_models")
PM_PE   = os.path.join(PM_ROOT, "apps", "pe")
sys.path.insert(0, PM_ROOT)   # for `core.*`
sys.path.insert(0, PM_PE)     # for `clip_benchmark.*`

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transforms

from clip_benchmark.datasets.builder import (
    build_dataset, get_dataset_collate_fn, get_dataset_default_task,
)
from clip_benchmark.metrics import zeroshot_classification
# zeroshot_retrieval imports audio-visual code (needs xformers); import lazily when used.

# Patch upstream accuracy(): float(np.ndarray) breaks on NumPy >=2 / >=1.25 for 1-elem arrays.
def _accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = len(target)
    return [correct[:k].reshape(-1).float().sum().item() / n for k in topk]
zeroshot_classification.accuracy = _accuracy


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
        metrics = zeroshot_classification.evaluate(
            model, loader, tokenizer, classnames, templates, device, amp=amp, verbose=False,
        )
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
# PE algorithm registry — add entries here to plug in new patches
# ──────────────────────────────────────────────────────────────────────────────

def _apply_flash_rope(model, args, dtype, img_size):
    if args.dtype == "fp32" or args.device != "cuda":
        print("[algorithm=flash_rope] requires CUDA + fp16/bf16; skipping patch.")
        return
    from PiToMe.algo.sparsesam.patch.pe import apply_pe_flash_rope_patch
    # Warm up rope.freq once so the patched forward has cos/sin ready.
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size, device=args.device, dtype=dtype)
        model.encode_image(dummy)
    apply_pe_flash_rope_patch(model)


PE_ALGORITHMS = {
    "none":       lambda model, args, dtype, img_size: None,
    "flash_rope": _apply_flash_rope,
    # add more here, e.g. "pitome": _apply_pitome, "tome": _apply_tome, ...
}


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
    p.add_argument("--algorithm", default="none",
                   choices=["none", "flash_rope"],
                   help="Optional PE patch to apply before evaluation. "
                        "'flash_rope' = fused FlashAttention+RoPE cute kernel "
                        "(needs CUDA + fp16/bf16). Add new entries to PE_ALGORITHMS to extend.")
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

    if args.algorithm != "none":
        print(f"Applying PE algorithm: {args.algorithm}")
        PE_ALGORITHMS[args.algorithm](model, args, dtype, img_size)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results: List[Dict] = []
    for ds_name in args.dataset:
        print(f"\n=== {ds_name} ===")
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
            "model": args.model, "dataset": ds_name, "split": args.split,
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
