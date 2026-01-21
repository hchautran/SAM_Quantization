# CUDA_VISIBLE_DEVICES=4 python small_engine.py --mode eval  --quantize-encoder --n-bits 8 --n-bits-mlp 8 --num-samples 400 --num-calib-samples 2 --config-file ./quant/config/hq44k/rtn.yaml 

# python profile_encoder_latency.py --model_type vit_l --checkpoint ./ckts/sam_hq_vit_l.pth --runs 10 --warmup 3

CUDA_VISIBLE_DEVICES=0 python benchmark_batch_inference.py \
    --batch-sizes  4 \
    --num-samples 3000 \
    --processor SUB_IMAGE_PRUNE \
    --percent 0.4  \
    --percent-global 0.5 \
    --prune-global \
    --n-bits 16 \
    --high-entropy  \
    --model-type vit_l \
    --model-ckt ./ckts/sam_hq_vit_l.pth \
    --quantize-encoder \
    --num-calib-samples 1 \
#     # --svdq-rank 32 \
#     # --svdq-precision int4 \
#     # --en-weight-quant svdq \
#     # --use-svdq \
#     # --high-entropy \
#     # --svdq-checkpoint ./output/svdq_sam_vit_l_r32_lowmem.pth \

# PERCENTS=" 0.0625	0.1875	0.25	0.375	0.5	0.5625	0.6875	0.75	0.875	1 " 
# # PERCENTS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1"
# processor="POSITIONAL_PRUNE" # HEAD_PRUNE POSITIONAL_PRUNE
# for p in $PERCENTS; do
#     echo "Running benchmark with percent: $p"
    
#     CUDA_VISIBLE_DEVICES=0 python benchmark_batch_inference.py \
#         --batch-sizes 8 \
#         --num-samples 3000 \
#         --processor $processor \
#         --percent $p \
#         --percent-global 0.5 \
#         --prune-global \
#         --n-bits 16 \
#         --high-entropy \
#         --model-type vit_h \
#         --model-ckt ./ckts/sam_hq_vit_h.pth \
#         --quantize-encoder \
#         --num-calib-samples 16
# done