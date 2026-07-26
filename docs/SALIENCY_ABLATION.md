# Reproducing the SparseSAM token-ordering ablation

This ablation answers two reviewer questions about the SparseSAM permutation:

  * *"An ablation over different Sobel filter sizes"* — does the **kernel size**
    of the gradient operator matter (3×3 vs 5×5 vs 7×7 vs 9×9)?
  * *"comparisons with alternative saliency estimators, such as Scharr filters,
    Laplacian filters, feature-norm-based scores, or attention-derived
    importance scores"* — does the **choice of operator** matter?

Plus the knob we wanted anyway: the **grid size** (tokens per space-filling-curve
group), currently fixed at 4.

> Grid size and Sobel kernel size are independent parameters. Grid size is
> `group_size` ∈ {2, 4, 8, 16}; Sobel kernel size is the conv kernel, odd only,
> ∈ {3, 5, 7, 9}.

## Files

| File | Role |
|---|---|
| [`sparsesam_saliency.py`](../sparsesam_saliency.py) | The configurable permutation: saliency estimators + grid size + curve + layout, plus a self-test |
| [`eval_hq44k_saliency_ablation.py`](../eval_hq44k_saliency_ablation.py) | HQ44K evaluation sweep (mIoU / boundary IoU / encoder latency / peak memory → CSV) |

Nothing under `PiToMe/` is modified. `sparsesam_saliency.install(cfg)` swaps the
module-level `tile_stride_matching` symbol inside
[`PiToMe/algo/sparsesam/sam.py`](../PiToMe/algo/sparsesam/sam.py) at runtime and
`restore()` puts the original back. The SparseSAM patch, the block-sparse cute
kernel, and the SAM-HQ decoder are untouched.

## Prerequisites

  * `./ckts/sam_hq_vit_l.pth` (or `_vit_b` / `_vit_h`)
  * HQ44K-style data under `./data/` — the four validation splits used here are
    `DIS5K-VD`, `COIFT`, `HRSOD`, `ThinObject5k-TE`
  * A CUDA GPU. The image encoder runs in fp16 because the cute block-sparse
    kernel is fp16-only; the prompt encoder / mask decoder stay fp32 (sam-hq's
    `PositionEmbeddingRandom` hard-casts prompt coords to fp32, so halving the
    **whole** model breaks box prompts).

## Quick check — no model, no data

Validates that every (saliency × grid size × curve × layout) combination
produces a valid permutation and inverse:

```bash
python sparsesam_saliency.py                      # all 384 combinations
python eval_hq44k_saliency_ablation.py --self-test-only \
    --group-sizes 2 4 8 16 --saliency sobel3 sobel5 sobel7 sobel9
```

The eval script runs this same self-test automatically before loading the model,
so a bad sweep spec fails in seconds instead of after the checkpoint load.

## The two sweeps

Both at keep-ratio 0.5 (= 50 % sparsity), SAM-HQ ViT-L, all four splits.
Run them on separate GPUs in parallel; they are independent.

**Axis 1 — grid size** (Sobel 3×3 held fixed), plus the dense baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_hq44k_saliency_ablation.py \
    --model-type vit_l --ratio 0.5 \
    --group-sizes 2 4 8 16 --saliency sobel3 \
    --num-samples 500 --include-baseline \
    --output-csv benchmark_results/saliency_ablation_hq44k/gridsize_sweep_vit_l_500.csv
```

**Axis 2 — saliency estimator** (grid size 4 held fixed):

```bash
CUDA_VISIBLE_DEVICES=1 python eval_hq44k_saliency_ablation.py \
    --model-type vit_l --ratio 0.5 --group-sizes 4 \
    --saliency sobel3 sobel5 sobel7 sobel9 scharr3 laplacian3 \
               feature_norm attention variance_dissim random \
    --num-samples 500 \
    --output-csv benchmark_results/saliency_ablation_hq44k/saliency_sweep_vit_l_500.csv
```

`--num-samples` is a cap, not a fixed count: COIFT has 280 images, so it
evaluates the whole split. Results are written to CSV incrementally after every
(config, dataset) pair, so an interrupted sweep keeps its completed rows.

Expect ~2 h per sweep at 500 images on an A100; ~25 min at 100 images.

## Summarising the CSVs

```bash
python - <<'EOF'
import pandas as pd
for f in ["benchmark_results/saliency_ablation_hq44k/gridsize_sweep_vit_l_500.csv",
          "benchmark_results/saliency_ablation_hq44k/saliency_sweep_vit_l_500.csv"]:
    d = pd.read_csv(f)
    piv = d.pivot_table(index=["saliency", "group_size"], columns="dataset", values="miou")
    piv["MEAN"] = piv.mean(axis=1)
    print(f, "\n", piv.round(4).to_string(), "\n")
EOF
```

Per-row columns: `saliency`, `group_size`, `curve`, `layout`, `ratio_keep`,
`sparsity`, `dataset`, `miou`, `miou_std`, `boundary_iou`,
`encoder_per_image_mean_ms`, `throughput_imgs_per_sec`,
`peak_memory_allocated_mb`.

## Available knobs

| Flag | Values | Notes |
|---|---|---|
| `--saliency` | `sobel3/5/7/9`, `scharr3/5`, `laplacian3/5`, `feature_norm`, `attention`, `variance_dissim`, `random` | `variance_dissim` is the scorer currently shipped in `sam.py`; `random` is the sanity floor |
| `--group-sizes` | any ints | `N` need not divide the grid size — local 14×14 windows (196 tokens) leave a ragged tail group, scored on its valid members only |
| `--curve` | `z`, `hilbert`, `raster` | grouping curve; `z` matches `sam.py` |
| `--layout` | `grouped` (default), `interleaved` | see below |
| `--ratio` | float in (0, 1] | keep ratio; 0.5 = 50 % sparsity |
| `--include-baseline` | flag | also evaluates the unpatched dense encoder |
| `--dataset-idx` | `0..3` | `0` DIS5K-VD, `1` COIFT, `2` HRSOD, `3` ThinObject5k-TE |

### What each saliency estimator computes

Each reduces a token's D-dim feature to one scalar; group score is the mean over
the group; groups are then ranked descending (high score = keep dense).

  * `sobel{3,5,7,9}` — `sqrt(mean_c(gx² + gy²))`, kernels built with the OpenCV
    `getDerivKernels` binomial recursion and L1-normalised
  * `scharr{3,5}` — same, with Scharr's rotationally-symmetric weights
  * `laplacian{3,5}` — `sqrt(mean_c(∇²x²))`, second derivative instead of first
  * `feature_norm` — `‖x‖₂` of the token
  * `attention` — incoming softmax mass among group representatives (a cheap
    G×G stand-in for the full N×N attention map)
  * `variance_dissim` — z-scored intra-group std + inter-group dissimilarity;
    the scorer in `sam.py::tile_stride_matching` today
  * `random` — random ranking; the floor that makes any positive claim meaningful

## Important: the `--layout` flag

The permutation in `sam.py::tile_stride_matching` ends with
`all_raster.permute(0, 2, 1)` — a stride/interleave layout. Under it the dense
keep-prefix is **identical regardless of the saliency score**; group ranking only
reorders tokens *within* each stripe:

```
interleaved  keep-prefix overlap, sobel3 vs random: 2048/2048   (N=4096, gs=4, r=0.5)
grouped      keep-prefix overlap, sobel3 vs random:  948/2048
```

So the ablation is only meaningful under `--layout grouped`, where ranked groups
are laid out back-to-back and the keep-prefix really is the top-scoring groups.
That matches the docstring on that same function ("Keep groups … followed by
merge groups") and the sibling implementations in
`PiToMe/algo/sparsesam/merge.py` and `PiToMe/algo/gradtome/sam_hilbert.py`.

`--layout interleaved` reproduces today's shipped behaviour and is the
null-result control: every estimator returns the same numbers.

## Results at 100 images (ViT-L, ratio 0.5, mean mIoU over the four splits)

Grid size, Sobel 3×3 — dense baseline `.8928`:

| grid | 2 | 4 | 8 | 16 |
|---|---|---|---|---|
| mIoU | .8869 | .8853 | .8855 | .8847 |

Saliency at grid 4:

| scharr3 | sobel5 | variance_dissim | sobel3 | sobel7 | laplacian3 | **random** | feature_norm | attention |
|---|---|---|---|---|---|---|---|---|
| .8862 | .8860 | .8859 | .8853 | .8849 | .8793 | **.8755** | .8750 | .8544 |

Reading: the gradient family (any Sobel size, Scharr) sits in a 0.13 pp band and
beats random ordering by ~1 pp mIoU; grid size 2→16 spans 0.22 pp. Both are
robustness results — the method does not hinge on the specific operator or grid
size, but it does hinge on *using a gradient*: Laplacian is measurably worse,
feature-norm is at the random floor, and attention-derived importance is 2 pp
*below* random.

Differences under ~0.3 pp are within run-to-run noise at 100 images. Rerun at
`--num-samples 500` for the numbers to quote, and use several draws of `random`
if you want an error bar on the floor.

## Extending the sweep

  * **Other sparsity levels** — `--ratio 0.25`; the ordering should matter more
    the fewer tokens you keep.
  * **Full cross product** — pass several `--group-sizes` *and* several
    `--saliency` values; the script sweeps `group_size × curve × saliency`.
  * **A new estimator** — add one entry to `TOKEN_SCORERS` (returns `(B, N)`
    token scores) or `GROUP_SCORERS` (returns `(B, G)` group scores) in
    `sparsesam_saliency.py`. It becomes a valid `--saliency` value automatically
    and is covered by the self-test.
