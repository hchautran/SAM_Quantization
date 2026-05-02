# SAM + PE Compression Sweep — Agent Playbook

This is the master playbook that an agent follows to **run every experiment**
in the SAM (HQ44K) × PE (ImageNet) compression study, including profiling, and
to **persist results + trade-off plots** under `.outputs/`.

The agent must execute every section in order, never skip a config, and
**not stop** until all results are saved and all plots are generated.

---

## 0. Layout

Eval / profile scripts (do not move them):

```
tasks/
├── tasks.md                       ← this file
├── sam_hq44k/
│   ├── eval_hq44k.py              ← SAM-HQ × HQ44K sweep (mIoU, B-IoU, throughput)
│   └── eval_hq44k.sh              ← env-var driven wrapper
├── pe_imagenet/
│   ├── eval_pe_clip.py            ← PE × ImageNet zero-shot CLS
│   ├── eval_pe.sh                 ← wrapper
│   ├── plot_pe_partial.py         ← PE trade-off curve plotter
│   ├── profile_pe.py / profile_pe.sh
│   └── data/imagenet/             ← imagenet1k root (val split)
└── sam_profile/
    ├── profile_encoder.py         ← per-component encoder profiler
    └── profile.sh
```

Algorithm implementations (modify here if a registry / patch needs fixing —
**not in the eval scripts**):

```
PiToMe/algo/
├── registry.py                    ← SAM_REGISTRY + PE_REGISTRY (single source of truth)
├── tome/         {sam.py, pe_compress.py, pe_partial.py, …}
├── gradtome/     {sam.py, pe_compress.py, pe_partial.py, …}
└── sparsesam/    {sam.py, pe_compress.py, pe_partial.py, …}
```

All outputs (CSV + plots) go to `.outputs/` under the repo root:

```
.outputs/
├── sam_hq44k/
│   ├── attn_only/        ← --no-mlp-merge  (sparse-attn only, full MLP)
│   ├── attn_plus_mlp/    ← --mlp-merge     (compressed MLP path)
│   ├── baseline/         ← algo=none reference
│   └── plots/            ← mIoU / B-IoU vs ratio, vs throughput
├── pe_imagenet/
│   ├── attn_only/        ← --no-mlp-merge
│   ├── attn_plus_mlp/    ← --mlp-merge
│   ├── baseline/
│   └── plots/            ← acc1/acc5 vs ratio, vs runtime
├── profile/
│   ├── sam/              ← per-block timings
│   └── pe/
└── run.log               ← consolidated stdout (append per-step)
```

Create the tree once at startup:

```bash
mkdir -p .outputs/{sam_hq44k/{attn_only,attn_plus_mlp,baseline,plots},pe_imagenet/{attn_only,attn_plus_mlp,baseline,plots},profile/{sam,pe}}
```

---

## 1. Pre-flight checks

Run **all** of these before the first experiment. Stop and surface to the
human if any check fails — do not paper over a missing dataset / checkpoint.

1. **GPU free**: `nvidia-smi` — confirm at least one GPU is idle and has
   ≥ 24 GB free (vit_l + bs=8 needs ~18 GB; PE-Core-L14-336 + bs=128 needs ~16 GB).
2. **SAM checkpoint**: `ls ./ckts/sam_hq_vit_l.pth`. If missing, abort.
3. **HQ44K datasets**: `python -c "from sam_engine import get_default_datasets; print(get_default_datasets())"`
   and verify the listed dirs exist under `/data/`.
4. **PE ImageNet**: `ls ./tasks/pe_imagenet/data/imagenet/val | head` — must
   contain class subdirs (one per WordNet ID).
5. **Registry sanity**: `python -c "from PiToMe.algo.registry import sam_algo_choices, algo_choices; print('SAM:', sam_algo_choices()); print('PE:', algo_choices())"`
   — must list `sparsesam, tome, gradtome` (SAM) and
   `sparsesam_partial, tome_partial, gradtome_partial` (PE).
6. **Branch**: create `experiments/<YYYYMMDD>` from current branch and run
   the whole sweep on it. Never push.

---

## 2. Sweep matrix

Fixed across every config:

| field        | value                             |
|--------------|-----------------------------------|
| ratios       | `0.75 0.6 0.5 0.4 0.3 0.25`                   |
| SAM model    | `vit_l` (`./ckts/sam_hq_vit_l.pth`) |
| SAM batch    | `8`                               |
| SAM samples  | `470` (full HQ44K val per dataset) |
| PE model     | `PE-Core-L14-336`                 |
| PE batch     | `128`                             |
| PE dtype     | `fp16`                            |
| PE dataset   | `imagenet1k` (val split)          |
| margin       | `0.5` (PiToMe energy)             |

The **algorithm × MLP-mode** matrix:

|                        | algorithms                                              | flag             |
|------------------------|---------------------------------------------------------|------------------|
| SAM baseline           | `none`                                                  | n/a              |
| SAM attn-only          | `sparsesam tome gradtome`                               | `--no-mlp-merge` |
| SAM attn + MLP merge   | `sparsesam tome gradtome`                               | `--mlp-merge`    |
| PE  baseline           | `none`                                                  | n/a              |
| PE  attn-only          | `sparsesam_partial tome_partial gradtome_partial`       | `--no-mlp-merge` |
| PE  attn + MLP merge   | `sparsesam_partial tome_partial gradtome_partial`       | `--mlp-merge`    |

**Why both MLP modes?** The diff between `--mlp-merge` and `--no-mlp-merge`
isolates the contribution of MLP-path compression. Sparse attention without
MLP merge is the conservative quality-preserving setting; MLP merge gives
extra speed at some quality cost.

---

## 3. Run experiments

All steps below redirect stdout/stderr to `.outputs/run.log` and **never
use `tee`** — log flooding kills agent context. After each step, verify the
expected CSV exists before moving on.

### 3.1 SAM — baseline (`algo=none`)

```bash
python tasks/sam_hq44k/eval_hq44k.py \
    --algos none \
    --batch-sizes 8 \
    --num-samples 470 \
    --model-ckt ./ckts/sam_hq_vit_l.pth \
    --model-type vit_l \
    --output-dir .outputs/sam_hq44k/baseline \
    --no-wandb \
    >> .outputs/run.log 2>&1
```

### 3.2 SAM — attn-only (`--no-mlp-merge`)

```bash
python tasks/sam_hq44k/eval_hq44k.py \
    --algos sparsesam tome gradtome \
    --ratios 0.75 0.6 0.5 0.4 0.3 0.25 \
    --batch-sizes 8 \
    --num-samples 470 \
    --model-ckt ./ckts/sam_hq_vit_l.pth \
    --model-type vit_l \
    --no-mlp-merge \
    --output-dir .outputs/sam_hq44k/attn_only \
    --no-wandb \
    >> .outputs/run.log 2>&1
```

### 3.3 SAM — attn + MLP merge (`--mlp-merge`)

```bash
python tasks/sam_hq44k/eval_hq44k.py \
    --algos sparsesam tome gradtome \
    --ratios 0.75 0.6 0.5 0.4 0.3 0.25 \
    --batch-sizes 8 \
    --num-samples 470 \
    --model-ckt ./ckts/sam_hq_vit_l.pth \
    --model-type vit_l \
    --mlp-merge \
    --output-dir .outputs/sam_hq44k/attn_plus_mlp \
    --no-wandb \
    >> .outputs/run.log 2>&1
```

### 3.4 PE — baseline (`algorithm=none`)

```bash
python tasks/pe_imagenet/eval_pe_clip.py \
    --model PE-Core-L14-336 \
    --dataset imagenet1k \
    --dataset-root ./tasks/pe_imagenet/data/imagenet \
    --batch-size 128 --dtype fp16 \
    --algorithm none \
    --output-dir .outputs/pe_imagenet/baseline \
    >> .outputs/run.log 2>&1
```

### 3.5 PE — attn-only (`--no-mlp-merge`)

```bash
python tasks/pe_imagenet/eval_pe_clip.py \
    --model PE-Core-L14-336 \
    --dataset imagenet1k \
    --dataset-root ./tasks/pe_imagenet/data/imagenet \
    --batch-size 128 --dtype fp16 \
    --algorithm sparsesam_partial tome_partial gradtome_partial \
    --ratio 0.75 0.6 0.5 0.4 0.3 0.25 \
    --no-mlp-merge \
    --output-dir .outputs/pe_imagenet/attn_only \
    >> .outputs/run.log 2>&1
```

### 3.6 PE — attn + MLP merge (`--mlp-merge`)

```bash
python tasks/pe_imagenet/eval_pe_clip.py \
    --model PE-Core-L14-336 \
    --dataset imagenet1k \
    --dataset-root ./tasks/pe_imagenet/data/imagenet \
    --batch-size 128 --dtype fp16 \
    --algorithm sparsesam_partial tome_partial gradtome_partial \
    --ratio 0.75 0.6 0.5 0.4 0.3 0.25 \
    --mlp-merge \
    --output-dir .outputs/pe_imagenet/attn_plus_mlp \
    >> .outputs/run.log 2>&1
```

After 3.1–3.6 finish, you should have **6 CSVs** under `.outputs/`. Verify:

```bash
find .outputs -name '*.csv' | sort
# expected: 1 SAM baseline + 2 SAM sweeps + 1 PE baseline + 2 PE sweeps = 6
```

---

## 4. Profiling

Profilers print per-block timings to stdout — capture them as text reports
(no CSV). Filenames embed the algo + ratio so reports don't overwrite.

### 4.1 SAM encoder

```bash
# Baseline (no compression)
python tasks/sam_profile/profile_encoder.py \
    --version sam1 --batch-size 1 \
    --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l \
    > .outputs/profile/sam/baseline.txt 2>&1

# One profile per (algo, ratio).  Algos: sparsesam, tome, gradtome.
# Ratios: 0.75 0.6 0.5 0.4 0.3 0.25.
for algo in sparsesam tome gradtome; do
  for r in 0.75 0.6 0.5 0.4 0.3 0.25; do
    python tasks/sam_profile/profile_encoder.py \
        --version sam1 --batch-size 1 \
        --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l \
        --tome-algo "$algo" --tome-ratio "$r" \
        > ".outputs/profile/sam/${algo}_r${r}.txt" 2>&1
  done
done
```

> If `profile_encoder.py --tome-algo gradtome` errors with "invalid choice",
> it means the script's argparse `choices=` is out of date. Fix it in
> `tasks/sam_profile/profile_encoder.py` (search for `--tome-algo`, add
> `gradtome` to the `choices` list) — do **not** wrap-around with a try/except.

### 4.2 PE encoder

```bash
# Baseline
MODEL=PE-Core-L14-336 ALGO=none BATCH=128 DTYPE=fp16 \
    sh tasks/pe_imagenet/profile_pe.sh \
    > .outputs/profile/pe/baseline.txt 2>&1

# Sweep
for algo in sparsesam_partial tome_partial gradtome_partial; do
  for r in 0.75 0.6 0.5 0.4 0.3 0.25; do
    MODEL=PE-Core-L14-336 ALGO=$algo RATIO=$r BATCH=128 DTYPE=fp16 \
        sh tasks/pe_imagenet/profile_pe.sh \
        > ".outputs/profile/pe/${algo}_r${r}.txt" 2>&1
  done
done
```

---

## 5. Plots — trade-off curves

### 5.1 SAM (mIoU + Boundary IoU vs keep-ratio)

`eval_hq44k.py` already drops a per-dataset PNG next to the CSV via its
`plot_results()` (mIoU and Boundary IoU vs ratio, one line per algo, dashed
baseline reference). Move them into the canonical plot dir and regenerate
a **combined** trade-off plot that puts attn-only and attn+MLP curves on the
same axes:

```bash
python - <<'PY' >> .outputs/run.log 2>&1
"""Combine SAM CSVs from attn_only/ + attn_plus_mlp/ + baseline/ and plot
mIoU and Boundary IoU vs ratio. One color per algo, solid=attn+mlp,
dashed=attn-only, dotted=baseline."""
import glob, os, pandas as pd, matplotlib.pyplot as plt

OUT = ".outputs/sam_hq44k/plots"
os.makedirs(OUT, exist_ok=True)

frames = []
for tag, sub in [("attn_only", ".outputs/sam_hq44k/attn_only"),
                 ("attn_plus_mlp", ".outputs/sam_hq44k/attn_plus_mlp"),
                 ("baseline", ".outputs/sam_hq44k/baseline")]:
    for csv in glob.glob(f"{sub}/*.csv"):
        df = pd.read_csv(csv); df["mode"] = tag
        frames.append(df)
df = pd.concat(frames, ignore_index=True)

datasets = sorted(df["dataset"].unique())
for metric, ylabel in [("miou", "mIoU"), ("boundary_iou", "Boundary IoU")]:
    n = len(datasets); ncols = min(3, n); nrows = (n + ncols - 1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 4.2*nrows),
                              squeeze=False)
    cmap = plt.get_cmap("tab10")
    algos = sorted(df[df.algo != "none"].algo.unique())
    color = {a: cmap(i % 10) for i, a in enumerate(algos)}

    for ax, ds in zip(axes.flat, datasets):
        d = df[df.dataset == ds]
        base = d[d.algo == "none"][metric].mean() if not d[d.algo == "none"].empty else None
        for algo in algos:
            for mode, ls, mk in [("attn_only", "--", "s"),
                                  ("attn_plus_mlp", "-", "o")]:
                row = (d[(d.algo == algo) & (d["mode"] == mode)]
                       .groupby("ratio", as_index=False)[metric].mean()
                       .sort_values("ratio"))
                if row.empty: continue
                ax.plot(row.ratio, row[metric], color=color[algo],
                        linestyle=ls, marker=mk, lw=1.6, ms=6,
                        label=f"{algo} ({mode})")
        if base is not None:
            ax.axhline(base, ls=":", color="black", alpha=0.6,
                       label=f"baseline = {base:.4f}")
        ax.set(title=ds, xlabel="keep ratio", ylabel=ylabel)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    for ax in axes.flat[len(datasets):]: ax.axis("off")
    fig.suptitle(f"SAM-HQ HQ44K — {ylabel} vs keep-ratio")
    fig.tight_layout()
    out = f"{OUT}/sam_tradeoff_{metric}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)
PY
```

### 5.2 PE (acc1/acc5 vs keep-ratio + vs runtime)

Use the existing plotter, pointing at the merged CSV set:

```bash
python tasks/pe_imagenet/plot_pe_partial.py \
    .outputs/pe_imagenet/attn_only/*.csv \
    .outputs/pe_imagenet/attn_plus_mlp/*.csv \
    .outputs/pe_imagenet/baseline/*.csv \
    --metric acc1 --x ratio \
    --out .outputs/pe_imagenet/plots/pe_tradeoff_acc1_vs_ratio.png \
    --model PE-Core-L14-336 \
    >> .outputs/run.log 2>&1

python tasks/pe_imagenet/plot_pe_partial.py \
    .outputs/pe_imagenet/attn_only/*.csv \
    .outputs/pe_imagenet/attn_plus_mlp/*.csv \
    .outputs/pe_imagenet/baseline/*.csv \
    --metric acc5 --x ratio \
    --out .outputs/pe_imagenet/plots/pe_tradeoff_acc5_vs_ratio.png \
    --model PE-Core-L14-336 \
    >> .outputs/run.log 2>&1

python tasks/pe_imagenet/plot_pe_partial.py \
    .outputs/pe_imagenet/attn_only/*.csv \
    .outputs/pe_imagenet/attn_plus_mlp/*.csv \
    .outputs/pe_imagenet/baseline/*.csv \
    --metric acc1 --x elapsed_s \
    --out .outputs/pe_imagenet/plots/pe_tradeoff_acc1_vs_runtime.png \
    --model PE-Core-L14-336 \
    >> .outputs/run.log 2>&1
```

> If `plot_pe_partial.py` doesn't already distinguish baseline rows when
> mlp_merge is missing, that's fine — the plotter falls back to a dotted
> "no-mlp_merge-info" line, which we want for the `algo=none` row.

---

## 6. LaTeX export — segmentation results table

After §3 finishes, emit a LaTeX table that mirrors the layout of Table 3
in the SparseSAM paper (Method × Density × per-dataset {mIoU, time}). The
table covers SAM-HQ on the four HQ44K val datasets:
**DIS5K-VD, COIFT, ThinObject5K-TE, HRSOD**.

Row → run mapping (read from the CSVs in `.outputs/sam_hq44k/`):

| LaTeX row name             | source CSV(s)                                      | algo filter        | density (%)        |
|----------------------------|----------------------------------------------------|--------------------|--------------------|
| Base Model                 | `.outputs/sam_hq44k/baseline/*.csv`                | `algo == "none"`   | 100                |
| Flash Attention 2          | (skip if not run; print `–` placeholders)          | n/a                | 100                |
| Sparse Attention           | `.outputs/sam_hq44k/attn_only/*.csv`               | `algo == "tome"`   | 25, 50, 75         |
| PieceWise Attention        | `.outputs/sam_hq44k/attn_plus_mlp/*.csv`           | `algo == "tome"`   | 25, 50, 75         |
| GradToMe (attn-only)       | `.outputs/sam_hq44k/attn_only/*.csv`               | `algo == "gradtome"` | 25, 50, 75       |
| GradToMe (attn + MLP)      | `.outputs/sam_hq44k/attn_plus_mlp/*.csv`           | `algo == "gradtome"` | 25, 50, 75       |
| SparseSAM (Ours, attn)     | `.outputs/sam_hq44k/attn_only/*.csv`               | `algo == "sparsesam"` | 25, 50, 75      |
| SparseSAM (Ours, attn+MLP) | `.outputs/sam_hq44k/attn_plus_mlp/*.csv`           | `algo == "sparsesam"` | 25, 50, 75      |

`density = round(ratio * 100)` — so ratio 0.25 → 25%. The matrix sweeps
six ratios; the LaTeX table only renders **{0.25, 0.50, 0.75}** to match
the paper layout. Other ratios are still in the trade-off plots (§5).

`time` = encoder forward latency in ms per image. Read
`encoder_per_image_mean_ms` from the CSVs (for SAM); for missing
configs print `--` so the row stays visible.

Run this script to generate `.outputs/sam_hq44k/seg_table.tex`:

```bash
python - <<'PY' >> .outputs/run.log 2>&1
"""Build a SparseSAM-style LaTeX segmentation table from the SAM CSVs."""
import glob, os, math, pandas as pd

OUT = ".outputs/sam_hq44k/seg_table.tex"
DATASETS = ["DIS5K-VD", "COIFT", "ThinObject5K-TE", "HRSOD"]
DENSITIES = [25, 50, 75]
EPS = 1e-3  # ratio fuzzy-match tolerance

def load(sub):
    rows = []
    for csv in glob.glob(f".outputs/sam_hq44k/{sub}/*.csv"):
        rows.append(pd.read_csv(csv))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

baseline = load("baseline")
attn     = load("attn_only")
mlp      = load("attn_plus_mlp")

def cell(df, dataset, ratio=None, algo=None, agg_metric="miou", agg_time="encoder_per_image_mean_ms"):
    """Return formatted (miou, time) strings or ('--','--') if missing."""
    d = df.copy()
    if d.empty:
        return "--", "--"
    d = d[d["dataset"] == dataset]
    if algo is not None:
        d = d[d["algo"] == algo]
    if ratio is not None:
        d = d[(d["ratio"] - ratio).abs() < EPS]
    if d.empty:
        return "--", "--"
    miou = d[agg_metric].mean()
    t    = d[agg_time].mean()
    return f"{miou:.4f}", f"{t:.2f}"

def fmt_row(label, density_str, df, ratio, algo):
    cells = []
    for ds in DATASETS:
        miou, t = cell(df, ds, ratio=ratio, algo=algo)
        cells.append(f"{miou} & {t}")
    body = " & ".join(cells)
    return f"{label} & {density_str} & {body} \\\\"

def fmt_block(label, df, algo):
    """Three rows (one per density), with multirow on the label column."""
    out = [f"\\multirow{{3}}{{*}}{{{label}}}"]
    parts = []
    for i, dens in enumerate(DENSITIES):
        ratio = dens / 100.0
        prefix = "" if i == 0 else " "
        cells = []
        for ds in DATASETS:
            miou, t = cell(df, ds, ratio=ratio, algo=algo)
            cells.append(f"{miou} & {t}")
        body = " & ".join(cells)
        parts.append(f"{prefix}& {dens}\\% & {body} \\\\")
    parts[0] = out[0] + parts[0]   # attach multirow to first density row
    return "\n".join(parts)

# Header
ds_header = " & ".join([f"\\multicolumn{{2}}{{c}}{{{ds}}}" for ds in DATASETS])
metric_header = " & ".join(["mIoU $\\uparrow$ & time $\\downarrow$"] * len(DATASETS))
col_spec = "l|c|" + "|".join(["cc"] * len(DATASETS))

lines = []
lines.append("\\begin{table}[t]")
lines.append("\\centering")
lines.append("\\caption{Segmentation results}")
lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
lines.append("\\hline")
lines.append(f"\\multirow{{2}}{{*}}{{Method}} & \\multirow{{2}}{{*}}{{Density}} & {ds_header} \\\\")
lines.append(f" &  & {metric_header} \\\\")
lines.append("\\hline")

# Base Model — read with algo=none, density=100%
base_cells = []
for ds in DATASETS:
    miou, t = cell(baseline, ds, ratio=None, algo="none")
    base_cells.append(f"{miou} & {t}")
lines.append(f"Base Model & 100\\% & {' & '.join(base_cells)} \\\\")

# Flash Attention 2 — placeholder unless we add a flash-rope run later
lines.append("Flash Attention 2 & 100\\% & " +
             " & ".join(["-- & --"] * len(DATASETS)) + " \\\\")
lines.append("\\hline")

# Token-merging baselines: ToMe with both MLP modes
lines.append(fmt_block("Sparse Attention (ToMe, attn-only)", attn, "tome"))
lines.append("\\hline")
lines.append(fmt_block("PieceWise Attention (ToMe, attn+MLP)", mlp, "tome"))
lines.append("\\hline")

# GradToMe rows (both modes)
lines.append(fmt_block("GradToMe (attn-only)", attn, "gradtome"))
lines.append("\\hline")
lines.append(fmt_block("GradToMe (attn+MLP)", mlp, "gradtome"))
lines.append("\\hline")

# SparseSAM — our two variants
lines.append(fmt_block("SparseSAM (Ours, attn-only)", attn, "sparsesam"))
lines.append("\\hline")
lines.append(fmt_block("SparseSAM (Ours, attn+MLP)", mlp, "sparsesam"))
lines.append("\\hline")

lines.append("\\end{tabular}")
lines.append("\\end{table}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("wrote", OUT)
PY
```

For PE on ImageNet, emit a parallel table at
`.outputs/pe_imagenet/cls_table.tex` with the same row structure but
columns = {Method, Density, ImageNet-1k {acc1, time}}. Time = `elapsed_s`
divided by ImageNet-val image count (50000) × 1000 → ms/image. Use the
same load → cell → fmt_block helpers, swapping `miou`→`acc1`,
`encoder_per_image_mean_ms`→`elapsed_s` (with the divisor), and the
algo names (`tome`→`tome_partial`, `gradtome`→`gradtome_partial`,
`sparsesam`→`sparsesam_partial`). Drop the GradToMe rows if you prefer
a tighter table — the trade-off plot already shows them.

---

## 7. Final report

When sections 3 → 6 all complete, write a single human-readable summary
to `.outputs/SUMMARY.md` with:

1. Table of all 6 runs: rows = (model, mode, algo, ratio), columns = key
   metric (mIoU / acc1), throughput, peak memory.
2. Top 3 most-efficient configs per model (highest metric at lowest ratio).
3. Inline links to the trade-off plots in `.outputs/{sam_hq44k,pe_imagenet}/plots/`.
4. Inline links to the LaTeX tables (`.outputs/sam_hq44k/seg_table.tex`
   and `.outputs/pe_imagenet/cls_table.tex`).
5. Any configs that crashed and the one-line root cause from `run.log`.

Build the table by reading the CSVs with pandas — do not retype numbers.

---

## 8. Hard rules for the agent

- **Never skip a config** in the matrix. If one crashes, log it in
  SUMMARY.md and continue with the rest.
- **Never pause to ask the human** mid-sweep. The sweep is the unit of work.
- **Do not modify eval scripts** unless a script is genuinely broken
  (missing argparse choice, import error, etc.). If you must change one,
  commit the fix on the experiment branch with a one-line message before
  re-running.
- **Do modify `PiToMe/algo/`** if a registered algorithm fails to apply or
  produces NaN/inf — that's where the patches live. Always re-run the
  affected config after a fix.
- **Output dir is `.outputs/`** — every CSV / plot / log lives under it.
  Never write to `./benchmark_results/` (legacy).
- **Redirect, don't tee.** Use `>> .outputs/run.log 2>&1` — flooding the
  agent's context with multi-MB logs makes the loop fail.
- **Verify after each step**: `ls` the expected CSV and grep for the
  algorithm names in it before moving on.
