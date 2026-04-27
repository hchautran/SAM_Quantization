#!/usr/bin/env bash
set -euo pipefail

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization"
source "$ROOT/.venv/bin/activate"
cd "$ROOT"
r_value=(128 256 512 1024 1536 2048 2560 3072)
for r in "${r_value[@]}"; do
    python token_merging/benchmark_batch_merge.py --r "$r" --batch-sizes 1
done
