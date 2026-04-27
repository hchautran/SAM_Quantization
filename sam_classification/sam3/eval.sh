#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization"
cd "$SCRIPT_DIR"

source "$ROOT/sam3_env/bin/activate"

export HF_HOME="${HF_HOME:-/tmp/huggingface_pe_eval}"
export HF_HUB_DISABLE_XET=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"
mkdir -p "$HF_HOME" "$SCRIPT_DIR/results" "$TRITON_CACHE_DIR"

DATA_PATH="${DATA_PATH:-$ROOT/data/imagenet}"
GT_FILE="${GT_FILE:-$ROOT/data/imagenet/labels/val_labels_hf_0based.txt}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
MODE="${MODE:-clip_zeroshot}"
MEASURE_SPEED="${MEASURE_SPEED:-1}"
PROCESSOR="${PROCESSOR:-PIECEWISE}"
# Set PROCESSOR to PIECEWISE or SPARSEATTN (alias: SPARGEATTN) to enable PE attention monkey-patching.
PERCENT="${PERCENT:-0.25}"
NUM_CALIB_SAMPLES="${NUM_CALIB_SAMPLES:-16}"
SUMMARY_PATH="${SUMMARY_PATH:-$SCRIPT_DIR/results/speed_and_Acc.txt}"

MODELS=(
   "facebook/PE-Core-S16-384"
)
# "facebook/PE-Core-B16-224"
#   "facebook/PE-Core-S16-384"
mkdir -p "$(dirname "$SUMMARY_PATH")"

for model in "${MODELS[@]}"; do
  safe_model="${model//\//_}"
  output_path="$SCRIPT_DIR/results/results_${safe_model}.json"
  cmd=(
    python evaluate_imagenet.py
    --data_path "$DATA_PATH"
    --gt-file "$GT_FILE"
    --model "$model"
    --mode "$MODE"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --num_images "$NUM_IMAGES"
    --output "$output_path"
    --summary-path "$SUMMARY_PATH"
    --percent "$PERCENT"
    --num-calib-samples "$NUM_CALIB_SAMPLES"
  )

  if [[ -n "$PROCESSOR" ]]; then
    cmd+=(--processor "$PROCESSOR")
  fi
  if [[ "$MEASURE_SPEED" == "1" ]]; then
    cmd+=(--measure-speed)
  fi

  echo "Running model: $model (mode=$MODE)"
  "${cmd[@]}"
done
