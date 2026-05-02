# CUDA_VISIBLE_DEVICES=4 python small_engine.py --mode eval  --quantize-encoder --n-bits 8 --n-bits-mlp 8 --num-samples 400 --num-calib-samples 2 --config-file ./quant/config/hq44k/rtn.yaml 

# python profile_encoder_latency.py --model_type vit_l --checkpoint ./ckts/sam_hq_vit_l.pth --runs 10 --warmup 3

# CUDA_VISIBLE_DEVICES=0 python benchmark_batch_inference.py \
#     --batch-sizes  4 \
#     --num-samples 3000 \
#     --processor HEAD_PRUNE \
#     --percent 0.4  \
#     --percent-global 0.5 \
#     --prune-global \
#     --n-bits 16 \
#     --high-entropy  \
#     --model-type vit_h \
#     --model-ckt ./ckts/sam_hq_vith.pth \
#     --quantize-encoder \
#     --num-calib-samples 1 \
#     # --svdq-rank 32 \
#     # --svdq-precision int4 \
#     # --en-weight-quant svdq \
#     # --use-svdq \
#     # --high-entropy \
#     # --svdq-checkpoint ./output/svdq_sam_vit_l_r32_lowmem.pth \

# PERCENTS=".5" 
processor="PIECE_WISE_ATTN"
cuda_device=0
PERCENT="0.25 0.5 0.75"
GLOBAL_PERCENT="0.1 0.25 0.3 0.4 0.5 0.6 0.75 0.8 0.9"
# processor="POSITIONAL_PRUNE" # HEAD_PRUNE POSITIONAL_PRUNE
# processor="POSITIONAL_SPARGE" # HEAD_PRUNE POSITIONAL_PRUNE POSITIONAL_SPARGE PIECE_WISE_ATTN POSITIONAL_SPARSE_FUSED_POS
# processor="BASE " # HEAD_PRUNE POSITIONAL_PRUNE

# Live attention profiling for EncoderAttentionProcessor in processors/encoder/basic.py
export PYTHONUNBUFFERED=1
export SAM_PROFILE_ENCODER_ATTN=1
export SAM_PROFILE_ENCODER_ATTN_WARMUP=3
export SAM_PROFILE_ENCODER_ATTN_PRINT_EVERY=1
for pp in $PERCENT; do
    for p in $GLOBAL_PERCENT; do
        echo "Running benchmark with global percent: $p and local percent: $pp"

        CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$cuda_device python benchmark_batch_inference.py \
            --batch-sizes 8 \
            --num-samples 500 \
            --processor $processor \
            --percent $pp \
            --percent-global $p \
            --prune-global \
            --n-bits 16 \
            --model-type vit_l \
            --model-ckt ./ckts/sam_hq_vit_l.pth \
            --quantize-encoder \
            --num-calib-samples 1 \
            --high-entropy
    done
done
