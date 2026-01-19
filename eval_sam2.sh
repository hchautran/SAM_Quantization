


ROOT=./data/sav/sav_test
GT_ROOT=./data/sav/sav_test/Annotations_6fps
PRED_ROOT=./outputs/sav_test_pred_pngs
CONFIG_PATH=./sam2/sam2/configs/sam2.1

CUDA_VISIBLE_DEVICES=0 python eval_sam2_hq44k.py \
    --model-cfg //home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/sam2_configs/sam2.1/sam2.1_hiera_b+.yaml \
    --checkpoint ./sam2_ckts/sam2.1_hiera_base_plus.pt  \
    --num-samples 500 \
    --percent-8heads   0.125 \
    --percent-200heads  0.5057 \
    --percent-400heads  0.7075 \
    --percent-2048heads  0.9956\
    --percent-4096heads  0.9656 \
    --high-entropy \
    --processor POSITIONAL_PRUNE_SAM2 \
    --batch-size  4 \
    --num-calib-samples 16 \
    --use-batch \
    --prune-global \
    # --use-batch \

# POSITIONAL_PRUNE_SAM2

# CUDA_VISIBLE_DEVICES=2 python eval_sam2_hq44k.py \
#     --model-cfg //home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/sam2_configs/sam2.1/sam2.1_hiera_b+.yaml \
#     --checkpoint ./sam2_ckts/sam2.1_hiera_base_plus.pt  \
#     --num-samples 400 \
#     --threshold 0.5 \
#     --threshold-global 0.00001 \
#     --processor TRAINING_PRUNE_RATE_SAM2_DUO \
#     --batch-size 4 \
#     --num-calib-samples 16 \
#     --prune-global \

# CUDA_VISIBLE_DEVICES=1 python eval_sam2_hq44k.py \
#     --model-cfg //home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/sam2_configs/sam2.1/sam2.1_hiera_b+.yaml \
#     --checkpoint ./sam2_ckts/sam2.1_hiera_base_plus.pt  \
#     --num-samples 400 \
#     --high-entropy \
#     --processor TRAINING_PRUNE_RATE_SAM2 \
#     --batch-size 1 \
#     --num-calib-samples 16 \
#     --prune-global \

# CUDA_VISIBLE_DEVICES=1 python eval_sam2_hq44k.py \
#     --model-cfg //home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/sam2_configs/sam2.1/sam2.1_hiera_b+.yaml \
#     --checkpoint ./sam2_ckts/sam2.1_hiera_base_plus.pt  \
#     --num-samples 3000 \
#     --high-entropy \
#     --processor TRAINING_PRUNE_RATE_SAM2_DIFF_DUO \
#     --batch-size 1 \
#     --num-calib-samples 16 \
#     --prune-global \
#     --threshold 0.8 \
#     --threshold-global  0.9999967\
#     # 0.5 0.999982


# THRESHOLDS=( 0.8 0.5 0.1 0.01 0.001 0.0001 0.00001 )
# THRESHOLD_GLOBALS=(0.5 0.999982 0.9999967)

# # Base command parameters


# # Loop through all combinations of thresholds

# for threshold_global in "${THRESHOLD_GLOBALS[@]}"; do
#     for threshold in "${THRESHOLDS[@]}"; do
#         echo "Running with threshold=$threshold and threshold-global=$threshold_global"
        
#         CUDA_VISIBLE_DEVICES=1 python eval_sam2_hq44k.py \
#             --model-cfg //home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/sam2_configs/sam2.1/sam2.1_hiera_b+.yaml \
#             --checkpoint ./sam2_ckts/sam2.1_hiera_base_plus.pt \
#             --num-samples 3000 \
#             --high-entropy \
#             --processor TRAINING_PRUNE_RATE_SAM2_DIFF_DUO \
#             --batch-size 1 \
#             --num-calib-samples 16 \
#             --prune-global \
#             --threshold $threshold \
#             --threshold-global $threshold_global
        
#         echo "Completed: threshold=$threshold, threshold-global=$threshold_global"
#         echo "----------------------------------------"
#     done
# done

# echo "All configurations completed!"


# python visualize_encoder_latency_sam2.py \
  # --model_config //home/chauht2/SAM_Quantization/sam2_configs/sam2.1/sam2.1_hiera_b+.yaml  \
  # --checkpoint ./sam2_ckts/sam2.1_hiera_base_plus.pt

# python visualize_encoder_memory_sam2.py \
  # --model_config //home/chauht2/SAM_Quantization/sam2_configs/sam2.1/sam2.1_hiera_b+.yaml  \
  # --checkpoint ./sam2_ckts/sam2.1_hiera_base_plus.pt


# python eval_sam2_hq44k.py \
#   --model-cfg //home/chauht2/SAM_Quantization/sam2_configs/sam2.1/sam2.1_hiera_b+.yaml \
#     --checkpoint ./sam2_ckts/sam2.1_hiera_base_plus.pt \
#     --processor POSITIONAL_PRUNE_SAM2 \
#     --num-calib-samples 32 \
#     --percent-entropy 0.4  \
#     --high-entropy \
#     --prune-global \
#     --use-batch \
#     --batch-size 2 \
#     --num-samples 100 

# python ./utils/vos_inference.py \
#   --sam2_cfg //home/chauht2/SAM_Quantization/sam2_configs/sam2.1/sam2.1_hiera_b+.yaml \
#   --sam2_checkpoint ./sam2_ckts/sam2.1_hiera_base_plus.pt \
#   --base_video_dir ${ROOT}/JPEGImages_24fps \
#   --input_mask_dir ${GT_ROOT} \
#   --video_list_file ${ROOT}/sav_test.txt \
#   --per_obj_png_file \
#   --output_mask_dir ${PRED_ROOT}

# CUDA_VISIBLE_DEVICES=0 python sam2_eval.py --gt_root ${GT_ROOT} --pred_root ${PRED_ROOT}