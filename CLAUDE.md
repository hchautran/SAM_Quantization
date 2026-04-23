# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repository for quantization and optimization of Segment Anything Models (SAM-1, SAM-HQ, SAM-2, SAM-2.1). The main techniques explored are token merging (ToMe/PiToMe/SparseSAM/GradToMe) and quantization to reduce compute and memory while maintaining segmentation quality.

## Common Commands

### Run token merging benchmark
```bash
python benchmark_tome.py \
    --algos none tome pitome sparsesam \
    --ratios 0.9 0.8 0.7 \
    --batch-sizes 1 2 4 \
    --model-ckt ./ckts/sam_hq_vit_l.pth \
    --model-type vit_l \
    --num-samples 100 --no-wandb
# or use the wrapper:
sh eval_tome.sh
```

### Run batch inference benchmark
```bash
python benchmark_batch_inference.py --batch-sizes 1 2 4 8 16 --num-samples 100
```

### Profile encoder (SAM-HQ baseline vs. patched)
```bash
python profile_encoder.py --version sam1 --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l
python profile_encoder.py --version sam1 --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l --tome-algo pitome --tome-ratio 0.8
```

### Evaluate SAM2 on HQ44k
```bash
python eval_sam2_hq44k.py \
    --model-cfg ./sam2_configs/sam2.1/sam2.1_hiera_b+.yaml \
    --checkpoint ./sam2_ckts/sam2.1_hiera_base_plus.pt \
    --num-samples 100
# or use the wrapper:
sh eval_sam2.sh
```

## Architecture

### Submodules
- **`sam-hq/`** — SAM-HQ model with training support; provides `SamPredictor`, checkpoint loading, and mask decoder
- **`PiToMe/`** — Token merging algorithms (ToMe, PiToMe, SparseSAM, GradToMe); patched into SAM encoder at runtime
- **`Block-Sparse-Attention/`** — Custom CUDA sparse attention kernels

### Main Repository Files
- **`benchmark_tome.py`** — Primary benchmark: sweeps algorithms × compression ratios × batch sizes; reports throughput, latency, GPU memory, mIoU
- **`benchmark_batch_inference.py`** — Batch-mode throughput benchmark (true multi-image batching)
- **`eval_sam2_hq44k.py`** — SAM2 evaluation with quantization and entropy-based pruning support
- **`profile_encoder.py`** — Per-component profiling of SAM/SAM2 encoder with optional ToMe patch
- **`sam_engine.py`** — `Evaluator` class wrapping SAM inference; `get_default_datasets()` returns DIS5K/ThinObject5K configs
- **`data_utils.py`** — `OnlineDataset` — loads image+mask pairs with augmentation for multiple datasets
- **`utils/`** — Shared helpers: benchmarking utilities, SAV/VOS inference utilities

### Token Merging Pattern (PiToMe submodule)
All algorithms follow the same three-step flow injected around every transformer block:
1. **Merge** — reduce token count via bipartite matching (metric varies by algorithm)
2. **MLP forward** — run on reduced token set (main speedup)
3. **Unmerge** — expand tokens back to full count

Key parameters:
- `--tome-ratio` (0–1): fraction of tokens to keep; lower = more compression
- `--tome-algo`: `none | tome | pitome | sparsesam | gradtome`
- `--tome-margin`: energy threshold for PiToMe

### Data Pipeline
```
Dataset (DIS5K / ThinObject5K / CascadePSP)
  → OnlineDataset (data_utils.py)
  → DataLoader
  → SAM model (optionally patched with ToMe)
  → SamPredictor / SAM2ImagePredictor
  → compute_iou / compute_boundary_iou
  → wandb + benchmark_results/
```

### Datasets (expected under `/data/`)
- `/data/DIS5K/` — high-detail segmentation (train + validation)
- `/data/thin_object_detection/` — COIFT, HRSOD, ThinObject5K
- `/data/cascade_psp/` — salient object detection benchmarks

### Checkpoints
- SAM-HQ checkpoints: `./ckts/sam_hq_vit_{t,b,l,h}.pth`
- SAM2/2.1 checkpoints: `./sam2_ckts/`
- SAM2 model configs: `./sam2_configs/sam2.1/`

## Environment

- Python 3.10 (`.python-version`)
- Key deps: `torch`, `torchvision`, `transformers>=4.55.4`, `timm>=1.0.19`, `accelerate`, `flash-attn2` (via `kernels`), `wandb`, `pycocotools`
- Install: `pip install -e .` (uses `pyproject.toml`)

## Metrics

Benchmarks report: **mIoU**, **Boundary IoU**, **throughput (img/s)**, **latency (ms)**, **GPU memory (MB)**. Results saved to `benchmark_results/` as CSV and optionally logged to wandb (disable with `--no-wandb`).
