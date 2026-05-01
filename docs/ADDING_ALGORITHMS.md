# Adding a new token-compression algorithm

This repo evaluates token-compression and sparse-attention patches on two
Vision Transformers:

  * **PE (Perception Encoder)** — Meta's CLIP-style image+text encoder.
    Evaluated zero-shot on ImageNet, CIFAR, COCO captions, etc. via
    [tasks/pe_imagenet/eval_pe_clip.py](../tasks/pe_imagenet/eval_pe_clip.py).
  * **SAM-HQ** — high-quality Segment Anything backbone. Evaluated on
    HQ44K-style segmentation benchmarks and COCO via
    [tasks/sam_hq44k/eval_hq44k.py](../tasks/sam_hq44k/eval_hq44k.py) and
    [tasks/sam_coco/eval_coco.py](../tasks/sam_coco/eval_coco.py).

A "patch" is a piece of code that monkey-patches the model's transformer
blocks at runtime to change how they process tokens — typically dropping or
merging tokens between blocks (compression) or substituting a faster
attention kernel. The original checkpoint stays unchanged; you just install
new `forward()` methods on the existing modules.

## How the registry works

Both backbones share a single registry at
[PiToMe/algo/registry.py](../PiToMe/algo/registry.py). Each registered
algorithm has a string name (the one you pass on the CLI), an
`apply(model, …)` function that installs the patch, and a `remove(model)`
function that puts the model back to baseline.

This means **adding a new algorithm is two steps**:

  1. Write the patch (an `apply` and a `remove` function — and optionally
     `Block`/`Attention` subclasses for SAM).
  2. Register it with one `register_pe(...)` or `register_sam(...)` call.

You don't touch any eval script. Once registered, the new name appears in
`--algorithm` / `--algos` choices automatically, the eval loop sweeps it
alongside the others, and the per-config `ratio` value is forwarded through
`kwargs_from_args` (see the per-backbone docs for which CLI flags get
plumbed through).

PE and SAM patches share the **same shape**: subclass the encoder's
`Block` / `Attention` classes (or for PE: `ResidualAttentionBlock` /
`SelfAttention`), override `forward()`, then in `apply_patch` walk the
encoder's modules and reassign `module.__class__` to your subclass.

## Which doc do I read?

  * **PE** → [ADDING_PE.md](ADDING_PE.md). Two patch flavors
    (stage-compression vs. partial-MLP) with their respective base classes,
    walkthroughs for both, smoke test, sweep+plot, gotchas.
  * **SAM-HQ** → [ADDING_SAM.md](ADDING_SAM.md). Single subclass-and-swap
    template, three-step patch→register→run example, smoke test, gotchas.

The PE side has a small amount of extra plumbing for stage boundaries
(RoPE re-indexing onto surviving tokens, optional fused FA2+RoPE kernel)
in `_pe_stage.py` / `_pe_stage_sparse.py`, exposed as base classes
(`FlashRopePEAttention`, `StageCompressPEBlock`) that you subclass.

---

## File map

```
PiToMe/algo/
├── registry.py                 # ← central PE+SAM registry; register here
├── _pe_stage.py                # PE stage-compression plumbing
├── _pe_stage_sparse.py         # PE block-sparse cute-kernel plumbing
├── tome/                       # bipartite ToMe / PiToMe
│   ├── merge.py                #   bipartite_soft_matching primitive
│   ├── pe_compress.py          #   PE: drop tokens at stage boundaries
│   ├── pe_partial.py           #   PE: full S; merged-K/V SDPA + merge/MLP/unmerge
│   └── sam.py                  #   SAM-HQ patch
├── gradtome/                   # gradient-aware matching variant
│   ├── merge.py
│   ├── pe_compress.py
│   ├── pe_partial.py
│   ├── sam.py
│   └── sam_hilbert.py          #   Hilbert-order variant
└── sparsesam/                  # Z-group / Hilbert sparsesam
    ├── pe_compress.py
    ├── pe_partial.py
    ├── sam.py
    └── sam_random.py           #   random-keep ablation baseline

tasks/                          # ← eval / profile scripts grouped by task
├── pe_imagenet/                # PE zero-shot CLIP (ImageNet, etc.)
├── sam_hq44k/                  # SAM-HQ throughput + mIoU on HQ44K-style sets
├── sam_coco/                   # SAM-HQ on COCO val2017 with GT-box prompts
└── sam_profile/                # SAM per-component profilers

# Repo root (shared by all tasks)
sam_eval_utils.py               # SAM apply/remove shims (used by both SAM tasks)
sam_engine.py                   # SAM-HQ default-datasets helper
data_utils.py                   # OnlineDataset + augmentations
```

### Naming conventions

The filename inside a `PiToMe/algo/<algo>/` folder tells you what the
patch does:

  * **`pe_compress.py`** — PE stage-compression (token count drops at the
    boundaries between stages). Built on `_pe_stage::apply_stage_compress`
    so all you write is a `compress_fn(x, active_idx, info)`.
  * **`pe_partial.py`** — full token count throughout (no compression).
    Reduces work by merging K/V before attention and/or running the MLP on
    a merged token set with a merge → MLP → unmerge sandwich.
  * **`sam.py`** / **`sam_hilbert.py`** / **`sam_random.py`** — SAM-HQ
    patches. Each defines `class ToMeSAMBlock(Block)`,
    `class ToMeSAMAttention(Attention)`, and `apply_patch(encoder, ...)`.
  * **`merge.py`** — the matcher primitive a patch imports
    (`bipartite_soft_matching`, `grad_bipartite_soft_matching`, …).
    Returns `(merge_fn, unmerge_fn)` callables.

### Running scripts

Each task has both a `*.py` entry point and a `*.sh` wrapper. The wrapper
`cd`s to the repo root before running Python so relative paths
(`./data/`, `./ckts/`, `./benchmark_results/`) resolve consistently no
matter where you call it from. Most knobs (model name, batch size,
algorithms to sweep, ratios) are env-overridable in the shell wrapper.

You can also call the `.py` directly — each one resolves
`_REPO = ../..` from `__file__` and adds `PiToMe/`, `sam-hq/`, and
`perception_models/` to `sys.path` automatically, so it works from any
directory.

```bash
# via wrapper (with optional env-var overrides)
sh tasks/pe_imagenet/eval_pe.sh
sh tasks/sam_coco/eval_coco.sh
MODEL=PE-Core-G14-448 BATCH=32 sh tasks/pe_imagenet/eval_pe.sh

# or call the .py directly
python tasks/sam_hq44k/eval_hq44k.py --algos tome --ratios 0.5 ...
```
