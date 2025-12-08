


ROOT=./data/sav/sav_test
GT_ROOT=./data/sav/sav_test/Annotations_6fps
PRED_ROOT=./outputs/sav_test_pred_pngs
CONFIG_PATH=./sam2/sam2/configs/sam2.1


python eval_sam2_hq44k.py \
    --model-cfg //home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/sam2_configs/sam2.1/sam2.1_hiera_b+.yaml \
    --checkpoint ./sam2_ckts/sam2.1_hiera_base_plus.pt  \
    --num-samples 100 \
    --percent-entropy  0.0  \
    --percent-entropy-global 0.0 \
    --high-entropy \
    --processor POSITIONAL_PRUNE_SAM2 \
    --batch-size 1
    # --use-batch \

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