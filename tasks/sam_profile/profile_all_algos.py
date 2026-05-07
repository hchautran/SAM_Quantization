#!/usr/bin/env python3
import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import torch

import profile_encoder
from segment_anything import sam_model_registry


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = (
    "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/"
    "SAM_Quantization/ckts/sam_hq_vit_l.pth"
)
DEFAULT_PYTHON_BIN = (
    "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/"
    "SAM_Quantization/.venv/bin/python"
)
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16]
CSV_COLUMNS = [
    "batch_size",
    "base",
    "tome",
    "gradtome",
    "sparsesam_merge_mlp",
    "sparsesam_no_merge_mlp",
    "piecewise",
    "sparge",
]
ALGO_SPECS = [
    ("base", None, True),
    ("tome", "tome", True),
    ("gradtome", "gradtome", True),
    ("sparsesam_merge_mlp", "sparsesam", True),
    ("sparsesam_no_merge_mlp", "sparsesam", False),
    ("piecewise", "piecewise", True),
    ("sparge", "sparge", True),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile SAM image encoder totals across algorithms and batch sizes."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to profile.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Shared ratio/density used for all non-base algorithms.",
    )
    parser.add_argument(
        "--global-ratio",
        type=float,
        default=0.5,
        help="Global-attention density used for piecewise/sparge.",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=5,
        help="Warmup iterations per configuration.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=20,
        help="Measured iterations per configuration.",
    )
    parser.add_argument(
        "--model-ckt",
        type=str,
        default=DEFAULT_CKPT,
        help="Checkpoint passed to the SAM1 loader.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_l",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM1 model type.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=1024,
        help="Input image size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base seed used to generate identical synthetic inputs across algorithms.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV path. Defaults to profile_table_ratio<R>.csv in this folder.",
    )
    parser.add_argument(
        "--stop-after-batch-size",
        type=int,
        default=None,
        help="Stop cleanly after finishing this batch size.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=DEFAULT_PYTHON_BIN,
        help="Python interpreter used for isolated child runs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Record NaN for failed runs and continue instead of aborting.",
    )
    parser.add_argument(
        "--strict",
        action="store_false",
        dest="continue_on_error",
        help="Abort the sweep on the first failed run.",
    )
    parser.add_argument(
        "--single-column",
        type=str,
        default=None,
        choices=[name for name, _, _ in ALGO_SPECS],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--single-batch-size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def build_output_path(args):
    if args.output_csv:
        return Path(args.output_csv).resolve()
    ratio_tag = str(args.ratio).replace(".", "p")
    return SCRIPT_DIR / f"profile_table_ratio{ratio_tag}.csv"


def write_csv(rows, output_csv):
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_sam(args):
    sam = sam_model_registry[args.model_type](checkpoint=args.model_ckt).to("cuda").eval()
    sam.image_encoder.half()
    return sam


def make_preprocessed_input(sam, batch_size, img_size, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformed_image = torch.randint(
        0,
        256,
        (batch_size, 3, img_size, img_size),
        device="cuda",
        dtype=torch.float32,
    )
    encoder_dtype = next(sam.image_encoder.parameters()).dtype
    input_image = sam.preprocess(transformed_image).to(dtype=encoder_dtype)
    return input_image


def time_encoder(encoder, input_image, n_warmup, n_runs):
    times_ms = []
    with torch.no_grad():
        for _ in range(n_warmup):
            encoder(input_image)
        torch.cuda.synchronize()

        for _ in range(n_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            encoder(input_image)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))

    return sum(times_ms) / len(times_ms)


def benchmark_one(args, batch_size, algo_name, merge_mlp):
    sam = load_sam(args)
    encoder = sam.image_encoder
    input_image = make_preprocessed_input(
        sam, batch_size, args.img_size, seed=args.seed + int(batch_size)
    )
    try:
        if algo_name is not None:
            profile_encoder.apply_profile_patch(
                encoder,
                algo=algo_name,
                ratio=args.ratio,
                margin=0.5,
                mlp_merge=merge_mlp,
                global_ratio=args.global_ratio,
            )
        return time_encoder(encoder, input_image, args.n_warmup, args.n_runs)
    finally:
        if algo_name is not None:
            try:
                profile_encoder.remove_profile_patch(encoder, algo_name)
            except Exception:
                pass
        del sam
        del encoder
        del input_image
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def run_single_child(args, batch_size, column_name):
    cmd = [
        args.python_bin,
        str(Path(__file__).resolve()),
        "--single-column",
        column_name,
        "--single-batch-size",
        str(batch_size),
        "--ratio",
        str(args.ratio),
        "--global-ratio",
        str(args.global_ratio),
        "--n-warmup",
        str(args.n_warmup),
        "--n-runs",
        str(args.n_runs),
        "--model-ckt",
        args.model_ckt,
        "--model-type",
        args.model_type,
        "--img-size",
        str(args.img_size),
        "--seed",
        str(args.seed),
    ]
    proc = subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
        raise RuntimeError(f"Child run failed with exit code {proc.returncode}")

    result = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT "):
            result = float(line.split()[1])
            break
    if result is None:
        raise RuntimeError("Child run did not emit a RESULT line.")
    return result


def main():
    args = parse_args()

    if args.single_column is not None:
        spec = next(spec for spec in ALGO_SPECS if spec[0] == args.single_column)
        total_ms = benchmark_one(args, args.single_batch_size, spec[1], spec[2])
        print(f"RESULT {total_ms}")
        return

    output_csv = build_output_path(args)
    rows = []

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for profiling.")

    for batch_size in args.batch_sizes:
        row = {"batch_size": batch_size}
        print(f"\n=== Batch Size {batch_size} ===", flush=True)

        for column_name, algo_name, merge_mlp in ALGO_SPECS:
            label = algo_name or "base"
            if algo_name == "sparsesam":
                label = f"{label} (merge_mlp={merge_mlp})"
            print(f"[run] bs={batch_size} algo={label}", flush=True)

            try:
                total_ms = run_single_child(args, batch_size, column_name)
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                total_ms = math.nan
                print(f"[fail] bs={batch_size} {column_name}: {exc}", flush=True)
            else:
                print(f"[done] bs={batch_size} {column_name}={total_ms:.3f} ms", flush=True)

            row[column_name] = total_ms
            write_csv(rows + [row], output_csv)

        rows.append(row)
        write_csv(rows, output_csv)

        if args.stop_after_batch_size is not None and batch_size == args.stop_after_batch_size:
            print(f"\nStopping after batch size {batch_size} as requested.", flush=True)
            break

    print(f"\nSaved results to {output_csv}", flush=True)


if __name__ == "__main__":
    main()
