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
PROCESSOR="${PROCESSOR:-SPARGEATTN}"
PERCENT="${PERCENT:-0.25}"
HIGH_ENTROPY="${HIGH_ENTROPY:-0}"
NUM_CALIB_SAMPLES="${NUM_CALIB_SAMPLES:-16}"
SUMMARY_PATH="${SUMMARY_PATH:-$SCRIPT_DIR/results/speed_and_Acc.txt}"

export HF_HOME="${HF_HOME:-/tmp/huggingface_sam2_eval}"
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$SCRIPT_DIR/results"

models=(
    
    "hiera_huge_224.mae_in1k_ft_in1k"
)
# Note that SPARSEATTN just support "hiera_huge_224.mae_in1k_ft_in1k" since it's head_dim =64 the remain models
# "hiera_base_plus_224.mae_in1k_ft_in1k" "hiera_large_224.mae_in1k_ft_in1k"  
mkdir -p "$(dirname "$SUMMARY_PATH")"

for model_name in "${models[@]}"; do
    safe_model_name="${model_name//\//_}"
    output_path="$SCRIPT_DIR/results/${safe_model_name}.json"

    # Define the command arguments ONLY (without CUDA_VISIBLE_DEVICES)
    cmd=(
        python evaluate_imagenet.py
        --model_name "$model_name"
        --data_path "$DATA_PATH"
        --gt_file "$GT_FILE"
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

    if [[ "$HIGH_ENTROPY" == "1" ]]; then
        cmd+=(--high-entropy)
    fi

    echo "Running model: $model_name"
    
    # Correct way: Place env var BEFORE the array expansion
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}"
done