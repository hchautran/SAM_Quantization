#!/usr/bin/env python3
"""Plot accuracy/ratio (or accuracy/runtime) trade-off curves from
eval_pe_clip.py CSV outputs; one curve per (algo, mlp_merge)."""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def _str_to_bool_or_none(s: str) -> Optional[bool]:
    """Map CSV string to bool/None. Empty/'None' → None; 'True'/'False' → bool."""
    if s is None or s == "" or s == "None":
        return None
    s = s.strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _to_float(s: str) -> Optional[float]:
    if s is None or s == "" or s == "None":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_rows(paths: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    for path in paths:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return rows


def group_curves(rows: List[Dict], metric: str, x_key: str
                 ) -> Tuple[Dict[Tuple[str, Optional[bool]], List[Tuple[float, float]]],
                            Optional[float]]:
    """Group rows into (algorithm, mlp_merge) → sorted list of (x, y).

    Skips rows with missing/error metric. Returns (curves, baseline_y) where
    baseline_y is the algorithm='none' accuracy if found."""
    curves: Dict[Tuple[str, Optional[bool]], List[Tuple[float, float]]] = defaultdict(list)
    baseline_y: Optional[float] = None

    for r in rows:
        algo = r.get("algorithm", "")
        if not algo:
            continue
        if "error" in r and r.get("error"):
            continue
        y = _to_float(r.get(metric, ""))
        if y is None:
            continue
        if algo == "none":
            baseline_y = y
            continue
        x = _to_float(r.get(x_key, ""))
        if x is None:
            continue
        mlp_merge = _str_to_bool_or_none(r.get("mlp_merge", ""))
        curves[(algo, mlp_merge)].append((x, y))

    for k in curves:
        curves[k].sort(key=lambda p: p[0])
    return curves, baseline_y


def plot(curves, baseline_y, metric, x_key, out_path, model_name=""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Stable colour-per-algo so {True, False} mlp_merge curves of the same
    # algo share a colour, distinguished by linestyle.
    algos = sorted({a for (a, _) in curves.keys()})
    cmap = plt.get_cmap("tab10")
    colors = {a: cmap(i % 10) for i, a in enumerate(algos)}

    for (algo, mm), pts in sorted(curves.items()):
        if not pts:
            continue
        xs, ys = zip(*pts)
        if mm is True:
            label = f"{algo}  (mlp_merge=True)"
            linestyle = "-"
            marker = "o"
        elif mm is False:
            label = f"{algo}  (mlp_merge=False)"
            linestyle = "--"
            marker = "s"
        else:
            label = algo
            linestyle = ":"
            marker = "x"
        ax.plot(xs, ys, label=label, color=colors[algo],
                linestyle=linestyle, marker=marker, linewidth=1.6,
                markersize=6)

    if baseline_y is not None:
        ax.axhline(baseline_y, color="black", linestyle=":", linewidth=1.2,
                   alpha=0.6, label=f"baseline (none) = {baseline_y:.4f}")

    pretty_x = {
        "ratio":     "kept fraction (ratio)",
        "elapsed_s": "eval runtime (s)",
    }.get(x_key, x_key)
    pretty_y = {
        "acc1": "ImageNet zero-shot top-1",
        "acc5": "ImageNet zero-shot top-5",
        "mean_per_class_recall": "mean per-class recall",
    }.get(metric, metric)

    title = "PE *_partial accuracy / compression trade-off"
    if model_name:
        title += f"  ({model_name})"
    ax.set_title(title)
    ax.set_xlabel(pretty_x)
    ax.set_ylabel(pretty_y)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best", framealpha=0.9)

    if x_key == "ratio":
        # Higher kept-fraction → less compression → typically higher acc.
        # Plot in left-to-right order of compression severity.
        ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+",
                   help="CSV file(s) or glob pattern(s) from eval_pe_clip.py.")
    p.add_argument("--metric", default="acc1",
                   choices=["acc1", "acc5", "mean_per_class_recall"],
                   help="Y-axis metric (default acc1).")
    p.add_argument("--x", default="ratio", dest="x_key",
                   help="X-axis column from the CSV (default ratio; try "
                        "elapsed_s for accuracy vs runtime).")
    p.add_argument("--out", default="./benchmark_results/pe_partial_tradeoff.png",
                   help="Output PNG path.")
    p.add_argument("--model", default="",
                   help="Optional model name to put in the plot title.")
    args = p.parse_args()

    # Expand globs (shell may not have done it).
    paths: List[str] = []
    for pat in args.paths:
        matched = sorted(glob.glob(pat))
        paths.extend(matched if matched else [pat])
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(2)
    print(f"Loading {len(paths)} CSV file(s):")
    for p_ in paths:
        print(f"  {p_}")

    rows = load_rows(paths)
    print(f"Loaded {len(rows)} rows.")

    curves, baseline = group_curves(rows, metric=args.metric, x_key=args.x_key)
    print(f"Groups: {len(curves)}  baseline({args.metric}): {baseline}")
    for (algo, mm), pts in sorted(curves.items()):
        tag = f"{algo}/{mm}"
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        print(f"  {tag:35s}  n={len(pts):2d}  "
              f"x∈[{min(xs):.3g},{max(xs):.3g}]  "
              f"y∈[{min(ys):.4f},{max(ys):.4f}]")

    if not curves:
        print("No curves to plot.", file=sys.stderr)
        sys.exit(2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot(curves, baseline, args.metric, args.x_key, args.out, model_name=args.model)


if __name__ == "__main__":
    main()
