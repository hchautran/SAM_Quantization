#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/.venv/bin/activate

CUDA_DEVICE="${CUDA_DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_IMAGES="${NUM_IMAGES:-50000}"
DATA_PATH="${DATA_PATH:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/data/imagenet}"
GT_FILE="${GT_FILE:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/data/imagenet/labels/val_labels_hf_0based.txt}"
MEASURE_SPEED="${MEASURE_SPEED:-1}"
PROCESSOR="${PROCESSOR:-MVITV2_PIECEWISE_ATTN}"
LOCAL_PERCENT="${LOCAL_PERCENT:-0.2}"
GLOBAL_PERCENT="${GLOBAL_PERCENT:-0.3}"
PRUNE_GLOBAL="${PRUNE_GLOBAL:-0}"
NUM_CALIB_SAMPLES="${NUM_CALIB_SAMPLES:-16}"
HIGH_ENTROPY="${HIGH_ENTROPY:-0}"
N_BITS="${N_BITS:-16}"
SUMMARY_PATH="${SUMMARY_PATH:-$SCRIPT_DIR/results/speed_and_Acc.txt}"

export HF_HOME="${HF_HOME:-/tmp/huggingface_sam1_eval}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$SCRIPT_DIR/results"

models=("vit_b" "vit_l" "vit_h")

if [[ "$MEASURE_SPEED" == "1" ]]; then
    : > "$SUMMARY_PATH"
fi

for model_name in "${models[@]}"; do
    output_path="$SCRIPT_DIR/results/results_${model_name}.json"

    cmd=(
        python evaluate_imagenet.py
        --model_type "$model_name"
        --data_path "$DATA_PATH"
        --gt_file "$GT_FILE"
        --batch_size "$BATCH_SIZE"
        --num_workers "$NUM_WORKERS"
        --num_images "$NUM_IMAGES"
        --output "$output_path"
        --summary-path "$SUMMARY_PATH"
        --percent "$LOCAL_PERCENT"
        --percent-global "$GLOBAL_PERCENT"
        --n-bits "$N_BITS"
        --num-calib-samples "$NUM_CALIB_SAMPLES"
    )

    if [[ -n "$PROCESSOR" ]]; then
        cmd+=(--processor "$PROCESSOR")
    fi
    if [[ "$PRUNE_GLOBAL" == "1" ]]; then
        cmd+=(--prune-global)
    fi
    if [[ "$HIGH_ENTROPY" == "1" ]]; then
        cmd+=(--high-entropy)
    fi
    if [[ "$MEASURE_SPEED" == "1" ]]; then
        cmd+=(--measure-speed)
    fi

    echo "Running model: $model_name"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}"
done
