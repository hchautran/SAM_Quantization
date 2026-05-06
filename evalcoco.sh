#!/bin/bash
export PYTHONPATH="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/PTQ4SAM/projects/instance_segment_anything/ops:${PYTHONPATH:-}"
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/.venv/bin/activate
# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29515 benchmark_batch_inference_coco.py \
#     --config-file quant/config/coco/rtn.yaml \
#     --quantize-encoder \
#     --n-bits 16 \
#     --num-calib-samples 16 \
#     --processor PRUNE_RATE_DUO \
#     --detector dino \
#     --checkpoint-path "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/ckts/prune_rate/duo_sam_hq_epoch_torchnograd_distill10_vit_h_reg-weight_0.5_lr0.01_lr_drop2.pth"
#     # --checkpoint-path ""/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/ckts/prune_rate/duo_sam_hq_epoch_torchnograd_distill_balance10_vit_l_reg-weight_0.5_lr0.01_lr_drop2.pth""
#     # --num-samples 100 \




# Configuration
YAML_FILE="./quant/config/coco/rtn.yaml"
PYTHON_SCRIPT="benchmark_batch_inference_coco.py"
CUDA_DEVICE=0
MASTER_PORT=29507

MODEL_TYPES=(  "vit_h"  )
HQ_CHECKPOINTS=(  "./ckts/sam_hq_vit_h.pth"  )
DETECTORS=(  "dino")
PROCESSORS=(  "GRAD_TOME")
# "dino" "hdetr" "TOME_PARTIAL" "GRAD_TOME" "SPARSE_PARTIAL"
PERCENT=(0.75)
TRAINING_METHOD="duo"

update_yaml_model() {
    local model_type="$1"
    local hq_checkpoint="$2"
    local training_method="$3"
    local percent_entropy="$4"

    local escaped_model_type
    local escaped_hq_checkpoint

    escaped_model_type=$(printf '%s\n' "$model_type" | sed 's/[&/\\]/\\&/g')
    escaped_hq_checkpoint=$(printf '%s\n' "$hq_checkpoint" | sed 's/[&/\\]/\\&/g')

    sed -i -E "/^[[:space:]]*model:[[:space:]]*$/,/^[^[:space:]]/ \
s/^([[:space:]]*model_type:[[:space:]]*).*/\1${escaped_model_type}/" "$YAML_FILE"

    sed -i -E "/^[[:space:]]*model:[[:space:]]*$/,/^[^[:space:]]/ \
s/^([[:space:]]*hq_checkpoint:[[:space:]]*).*/\1${escaped_hq_checkpoint}/" "$YAML_FILE"

    sed -i -E "/^[[:space:]]*train_prune_rate:[[:space:]]*$/,/^[^[:space:]]/ \
s/^([[:space:]]*training_method:[[:space:]]*).*/\1${training_method}/" "$YAML_FILE"

    sed -i -E "/^[[:space:]]*quantization:[[:space:]]*$/,/^[^[:space:]]/ \
s/^([[:space:]]*percent_entropy:[[:space:]]*).*/\1${percent_entropy}/" "$YAML_FILE"

    sed -i -E "/^[[:space:]]*quantization:[[:space:]]*$/,/^[^[:space:]]/ \
s/^([[:space:]]*percent_entropy_global:[[:space:]]*).*/\1${percent_entropy}/" "$YAML_FILE"

    echo "[YAML updated] model_type=${model_type}, hq_checkpoint=${hq_checkpoint}, training_method=${training_method}, percent=${percent_entropy}"
}

for k in "${!PERCENT[@]}"; do
    PERCENT="${PERCENT[$k]}"
    for i in "${!MODEL_TYPES[@]}"; do
        for j in "${!DETECTORS[@]}"; do
            for processor in "${PROCESSORS[@]}"; do
                model_type="${MODEL_TYPES[$i]}"
                hq_checkpoint="${HQ_CHECKPOINTS[$i]}"
                detector="${DETECTORS[$j]}"

                update_yaml_model "$model_type" "$hq_checkpoint" "$TRAINING_METHOD" "$PERCENT"

                echo "Running evaluation with model_type=${model_type}, hq_checkpoint=${hq_checkpoint}, detector=${detector}, processor=${processor}"
                CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=1 --master_port=$MASTER_PORT $PYTHON_SCRIPT \
                    --config-file $YAML_FILE \
                    --quantize-encoder \
                    --n-bits 16 \
                    --num-calib-samples 1 \
                    --processor "$processor" \
                    --percent "$PERCENT" \
                    --detector "$detector" \
                    --profile-image-encoder \
                    --num-samples 110 \
                    --profile-warmup-calls 10 \
                    --merge-mlp

                echo "Evaluation completed for model_type=${model_type}, hq_checkpoint=${hq_checkpoint}, detector=${detector}, processor=${processor}"
            done
        done
    done
done

echo "All evaluations completed."
