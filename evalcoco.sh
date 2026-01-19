CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29519 benchmark_batch_inference_coco.py \
    --config-file quant/config/coco/rtn.yaml \
    --quantize-encoder \
    --n-bits 16 \
    --num-calib-samples 16 \
    --processor POSITIONAL_QUANT \
    --detector yolox 
    # --num-samples 100 \



# # Configuration
# YAML_FILE="./quant/config/coco/rtn.yaml"
# PYTHON_SCRIPT="benchmark_batch_inference_coco.py"
# CUDA_DEVICE=1
# MASTER_PORT=29500

# # Model types and corresponding HQ checkpoints
# MODEL_TYPES=("vit_b" "vit_l" "vit_h")
# HQ_CHECKPOINTS=("./ckts/sam_hq_vit_b.pth" "./ckts/sam_hq_vit_l.pth" "./ckts/sam_hq_vit_h.pth")
# PROCESSORS=("BASE" "POSITIONAL_PRUNE" "POSITIONAL_QUANT")
# # Function to update YAML with model_type and hq_checkpoint
# update_yaml_model() {
#     local model_type="$1"
#     local hq_checkpoint="$2"

#     # Escape special characters in the variables
#     escaped_model_type=$(printf '%s\n' "$model_type" | sed 's/[&/\]/\\&/g')
#     escaped_hq_checkpoint=$(printf '%s\n' "$hq_checkpoint" | sed 's/[&/\]/\\&/g')

#     # Update model_type inside the model block
#     sed -i -E "/^[[:space:]]*model:[[:space:]]*$/,/^[^[:space:]]/ \
# s/^([[:space:]]*model_type:[[:space:]]*).*/\1${escaped_model_type}/" "$YAML_FILE"

#     # Update hq_checkpoint inside the model block
#     sed -i -E "/^[[:space:]]*model:[[:space:]]*$/,/^[^[:space:]]/ \
# s/^([[:space:]]*hq_checkpoint:[[:space:]]*).*/\1${escaped_hq_checkpoint}/" "$YAML_FILE"

#     echo "[YAML updated] model_type=${model_type}, hq_checkpoint=${hq_checkpoint}"
# }

# # Iterate over all model types and HQ checkpoints
# for i in "${!MODEL_TYPES[@]}"; do
#     model_type="${MODEL_TYPES[$i]}"
#     hq_checkpoint="${HQ_CHECKPOINTS[$i]}"

#     # Update YAML file with current model_type and hq_checkpoint
#     update_yaml_model "$model_type" "$hq_checkpoint"
#     for j in "${!PROCESSORS[@]}"; do  # Use a different variable (e.g., j) for the inner loop
#         processor="${PROCESSORS[$j]}"
        
#         # Run evaluation script
#         echo "Running evaluation with model_type=${model_type}, hq_checkpoint=${hq_checkpoint}, processor=${processor}"
#         CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun --nproc_per_node=1 --master_port=$MASTER_PORT $PYTHON_SCRIPT \
#             --config-file $YAML_FILE \
#             --quantize-encoder \
#             --n-bits 16 \
#             --num-calib-samples 16 \
#             --processor "$processor" \
#             --detector dino

#         echo "Evaluation completed for model_type=${model_type}, hq_checkpoint=${hq_checkpoint}, processor=${processor}"
#     done
# done

# echo "All evaluations completed."