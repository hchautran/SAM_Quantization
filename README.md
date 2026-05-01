---
license: apache-2.0
---

# SAM Quantization

Research repository for quantization and token-compression of Vision
Transformers — currently **Segment Anything (SAM-HQ)** and Meta's
**Perception Encoder (PE)**. Algorithms include token merging
(ToMe / PiToMe / SparseSAM / GradToMe), block-sparse attention, and
optional fused FA2+RoPE kernels.

All compression algorithms are runtime patches: they monkey-patch the
encoder's transformer blocks at apply time and revert cleanly, so the
original checkpoints stay unchanged and a single eval run can sweep
several `(algo, ratio)` configs back-to-back.

---

## Installation

The maintainer's working environment is a conda env with **Python 3.12**
and **PyTorch 2.5.1 + CUDA 12.1**. `.python-version` pins 3.10 but the
codebase runs on 3.10–3.12; pick whichever matches your CUDA toolchain.

```bash
# 1. Clone with submodules (PiToMe, sam-hq, perception_models, SpargeAttn, …)
git clone --recurse-submodules <repo-url> SAM_Quantization
cd SAM_Quantization
# (or, if already cloned)
git submodule update --init --recursive

# 2. Create env
conda create -n sam python=3.12 -y
conda activate sam

# 3. PyTorch (must match your CUDA — these are the versions in the working env)
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 4. Repo + extra pinned deps
pip install -e .
pip install -r requirements.txt

# 5. Submodules that ship a Python package — install editable so local
#    changes are picked up. (Each has its own setup.py.)
pip install -e sam-hq
pip install -e PiToMe
pip install -e perception_models
# only if you actually use SAM3 / SAM2:
pip install sam2 sam3
```

### Optional / kernel deps

Install only the ones you need:

```bash
# Flash-Attention 2 (used by some PE patches and the FA2+RoPE fused kernel)
pip install flash-attn==2.8.3 --no-build-isolation

# xFormers (memory-efficient attention; required by perception_models)
pip install xformers==0.0.35

# Cute block-sparse attention path (sparsesam_partial). Pulls in
# nvidia-cutlass-dsl >=4.4.
pip install -e SpargeAttn
```

`Block-Sparse-Attention` is listed in `.gitmodules` but is **not used**
by the current eval/profile scripts — its directory is not populated in
the working tree. Skip it unless you need a path that explicitly imports
it.

### Verified versions

The exact pins from the working conda env (useful when reproducing
benchmark numbers):

| Package | Version |
|---|---|
| python | 3.12.13 |
| torch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 |
| triton | 3.1.0 |
| flash-attn | 2.8.3 |
| xformers | 0.0.35 |
| transformers | 5.6.2 |
| timm | 1.0.26 |
| accelerate | 1.13.0 |
| kernels | 0.13.0 |
| nvidia-cutlass-dsl | 4.4.2 |
| pycocotools | 2.0.11 |
| wandb | 0.25.0 |

Expected layout for data and checkpoints:

```
ckts/                       # SAM-HQ: sam_hq_vit_{t,b,l,h}.pth
sam2_ckts/                  # SAM2 / SAM2.1
sam2_configs/sam2.1/        # SAM2 model configs
data/DIS5K/                 # high-detail segmentation
data/thin_object_detection/ # COIFT, HRSOD, ThinObject5K
data/cascade_psp/           # salient object detection
data/coco/                  # COCO val2017 + annotations
data/imagenet/              # ImageNet1k for PE zero-shot eval
benchmark_results/          # CSV outputs (created on first run)
```

---

## Repo layout

```
PiToMe/algo/        # token-compression algorithms + central registry
tasks/              # eval / profile entry points, grouped by task
  pe_imagenet/        PE zero-shot CLIP eval + per-block profiler
  sam_hq44k/          SAM-HQ on HQ44K-style segmentation benchmarks
  sam_coco/           SAM-HQ on COCO val2017 with GT-box prompts
  sam_profile/        SAM per-component / per-attn-layer profilers
sam-hq/             # SAM-HQ submodule (model + predictor)
perception_models/  # PE submodule (Perception Encoder)
docs/               # contributor docs — start here when adding an algo
```

Every task ships both a `*.py` entry point and a `*.sh` wrapper. The
wrapper `cd`s to the repo root before invoking Python so relative paths
(`./data/`, `./ckts/`, `./benchmark_results/`) resolve consistently.
Most knobs (model, batch size, algos, ratios) are env-overridable in
the wrappers.

---

## Running evaluations

| Task | Entry point | Wrapper |
|---|---|---|
| SAM-HQ on HQ44K | [tasks/sam_hq44k/eval_hq44k.py](tasks/sam_hq44k/eval_hq44k.py) | [tasks/sam_hq44k/eval_hq44k.sh](tasks/sam_hq44k/eval_hq44k.sh) |
| SAM-HQ on COCO val2017 | [tasks/sam_coco/eval_coco.py](tasks/sam_coco/eval_coco.py) | [tasks/sam_coco/eval_coco.sh](tasks/sam_coco/eval_coco.sh) |
| PE zero-shot CLIP (ImageNet etc.) | [tasks/pe_imagenet/eval_pe_clip.py](tasks/pe_imagenet/eval_pe_clip.py) | [tasks/pe_imagenet/eval_pe.sh](tasks/pe_imagenet/eval_pe.sh) |

```bash
# SAM-HQ HQ44K mIoU/throughput sweep
python tasks/sam_hq44k/eval_hq44k.py --algos tome pitome --ratios 0.9 0.7 \
    --batch-sizes 1 2 4 --num-samples 100 \
    --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l

# SAM-HQ COCO val2017 with GT-box prompts (+ optional COCO segm AP)
python tasks/sam_coco/eval_coco.py --algos pitome --ratios 0.7 \
    --coco-root ./data/coco --num-images 200 --ap \
    --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l

# PE zero-shot CLIP
ALGOS_SWEEP="tome_partial gradtome_partial" RATIOS_SWEEP="0.9 0.7 0.5" \
    sh tasks/pe_imagenet/eval_pe.sh
```

Each script picks any newly-registered algorithm up automatically — see
the contributor docs below.

Reported metrics: **mIoU**, **Boundary IoU**, **throughput (img/s)**,
**latency (ms)**, **GPU memory (MB)**; PE eval reports top-1 / top-5.
Results are written to `benchmark_results/` as CSV and optionally logged
to wandb (disable with `--no-wandb`).

---

## Profiling

| Target | Entry point | Wrapper |
|---|---|---|
| SAM encoder, per-component | [tasks/sam_profile/profile_encoder.py](tasks/sam_profile/profile_encoder.py) | [tasks/sam_profile/profile.sh](tasks/sam_profile/profile.sh) |
| SAM, per-attention-layer | [tasks/sam_profile/profile_attn_layers.py](tasks/sam_profile/profile_attn_layers.py) | — |
| PE, per-block latency | [tasks/pe_imagenet/profile_pe.py](tasks/pe_imagenet/profile_pe.py) | [tasks/pe_imagenet/profile_pe.sh](tasks/pe_imagenet/profile_pe.sh) |

```bash
# SAM-HQ encoder, baseline vs. patched
python tasks/sam_profile/profile_encoder.py --version sam1 \
    --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l \
    --tome-algo pitome --tome-ratio 0.8

# PE per-block
python tasks/pe_imagenet/profile_pe.py --tome-algo tome_partial --tome-ratio 0.5
```

Note: [tasks/sam_profile/profile_encoder.py](tasks/sam_profile/profile_encoder.py)
imports `apply_patch` directly from a specific algo module rather than
going through the registry — see
[ADDING_SAM.md §5](docs/ADDING_SAM.md#5-common-gotchas) for what to edit
when profiling a new algorithm.

---

## Adding a new algorithm

The contributor docs in [docs/](docs/) cover this end-to-end:

  * **[docs/ADDING_ALGORITHMS.md](docs/ADDING_ALGORITHMS.md)** — overview:
    how the shared PE+SAM registry works, file layout under
    `PiToMe/algo/`, naming conventions (`pe_compress.py` /
    `pe_partial.py` / `sam.py` / `merge.py`), and which doc to read for
    each backbone.
  * **[docs/ADDING_SAM.md](docs/ADDING_SAM.md)** — SAM-HQ patches: the
    subclass-and-swap template (`Block` / `Attention`), three-step
    patch → register → run example, smoke test, and gotchas (windowed
    vs. global blocks, per-image cache reset, profiler bypassing the
    registry).
  * **[docs/ADDING_PE.md](docs/ADDING_PE.md)** — PE patches: both
    flavors (stage-compression and partial / full-token-count), the
    `_pe_stage.py` base classes, `kwargs_from_args` builders, sweep
    + plot, and gotchas (lazy imports, autocast/LayerNorm, RoPE under
    compression, cute kernel availability).

Once registered, your algorithm appears as a choice in `--algos` /
`--algorithm` for every eval and profile script automatically — no
changes to the entry-point scripts needed.
