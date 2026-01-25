    

############### The script below is for running multiple config at the same time - you can run normal script by changing the config in the yaml file and without signal --train
## run bash for this file
YAML_FILE="./quant/config/coco/rtn.yaml"
PYTHON_SCRIPT="small_engine_train_duo.py"
CUDA_DEVICE=2
MASTER_PORT=29504
# CHECKPOINT_EVALUATION="/home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/pretrained_checkpoint/prune_rate/diffduo_sam_hq_epoch_torchnograd_distill10_vit_l_reg-weight_0.5_lr0.05_lr_drop2.pth"
# CHECKPOINT_EVALUATION="/home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/pretrained_checkpoint/prune_rate/diffduo_sam_hq_epoch_torchnograd_distill_balance10_vit_l_reg-weight_0.5_lr0.02_lr_drop2.pth"
CHECKPOINT_EVALUATION="/home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/pretrained_checkpoint/prune_rate/diffduo_sam_hq_epoch_torchnograd_distill_balance10_vit_l_reg-weight_5_lr0.05_lr_drop2.pth"

# threshold combinations: "local,global"

# local thresholds
# LOCAL_PERCENT=(  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1  )
LOCAL_PERCENT=(0.0625	0.1875	0.25	0.375	0.5	0.5625	0.6875	0.75	0.875	1)
# global thresholds
GLOBAL_PERCENT=(   0.5 )
# Update YAML (only inside train_prune_rate: block)
update_yaml_thresholds() {
    local thr_local="$1"
    local thr_global="$2"
    local training_method="diffduo"  # Added 'local' for function scope if desired; remove if global is intended
    local use_percentage=True
    # Replace quantization.percent_entropy
    sed -i -E "/^[[:space:]]*quantization:[[:space:]]*$/,/^[^[:space:]]/ \
s/^([[:space:]]*percent_entropy:[[:space:]]*).*/\1${thr_local}/" "$YAML_FILE"

    # Replace quantization.percent_entropy_global
    sed -i -E "/^[[:space:]]*quantization:[[:space:]]*$/,/^[^[:space:]]/ \
s/^([[:space:]]*percent_entropy_global:[[:space:]]*).*/\1${thr_global}/" "$YAML_FILE"

    sed -i -E "/^[[:space:]]*quantization:[[:space:]]*$/,/^[^[:space:]]/ \
s/^([[:space:]]*use_percentage:[[:space:]]*).*/\1${use_percentage}/" "$YAML_FILE"

    # Replace train_prune_rate.training_method (assuming nested under train_prune_rate:)
    sed -i -E "/^[[:space:]]*train_prune_rate:[[:space:]]*$/,/^[^[:space:]]/ \
s/^([[:space:]]*training_method:[[:space:]]*).*/\1${training_method}/" "$YAML_FILE"

    

    echo "[YAML updated] quantization.percent_entropy=${thr_local}, quantization.percent_entropy_global=${thr_global}, train_prune_rate.training_method=${training_method}"
}


for thr_global in "${GLOBAL_PERCENT[@]}"; do
    for thr_local in "${LOCAL_PERCENT[@]}"; do
        echo "Running: local=$thr_local, global=$thr_global"

        update_yaml_thresholds "$thr_local" "$thr_global"

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun \
            --nproc_per_node=1 \
            --master_port=$MASTER_PORT \
            "$PYTHON_SCRIPT" --config-file "$YAML_FILE" \
            --checkpoint-evaluation "$CHECKPOINT_EVALUATION" \
            --num-samples 5000 \
            --num-calib-samples 16
    done
done
