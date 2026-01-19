    

############### The script below is for running multiple config at the same time - you can run normal script by changing the config in the yaml file and without signal --train
## run bash for this file
YAML_FILE="./quant/config/coco/rtn.yaml"
PYTHON_SCRIPT="small_engine_train_duo.py"
CUDA_DEVICE=5
MASTER_PORT=29502
CHECKPOINT_EVALUATION="./pretrained_checkpoint/prune_rate/diffduo_sam_hq_epoch_torchnograd_distill10_vit_l_reg-weight_5_lr0.05_lr_drop2.pth"

# threshold combinations: "local,global"

# local thresholds
LOCAL_THRESHOLDS=(  0.000045  )

# global thresholds
GLOBAL_THRESHOLDS=(   0.99982 )
# Update YAML (only inside train_prune_rate: block)
update_yaml_thresholds() {
    local thr_local="$1"
    local thr_global="$2"

    # Replace train_prune_rate.threshold
    sed -i -E "/^[[:space:]]*train_prune_rate:[[:space:]]*$/,/^[^[:space:]]/ \
s/^([[:space:]]*threshold:[[:space:]]*).*/\1${thr_local}/" "$YAML_FILE"

    # Replace train_prune_rate.threshold_globle
    sed -i -E "/^[[:space:]]*train_prune_rate:[[:space:]]*$/,/^[^[:space:]]/ \
s/^([[:space:]]*threshold_globle:[[:space:]]*).*/\1${thr_global}/" "$YAML_FILE"

    echo "[YAML updated] train_prune_rate.threshold=${thr_local}, train_prune_rate.threshold_globle=${thr_global}"
}


for thr_global in "${GLOBAL_THRESHOLDS[@]}"; do
    for thr_local in "${LOCAL_THRESHOLDS[@]}"; do
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
