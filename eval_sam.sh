# CUDA_VISIBLE_DEVICES=4 python small_engine.py --mode eval  --quantize-encoder --n-bits 8 --n-bits-mlp 8 --num-samples 400 --num-calib-samples 2 --config-file ./quant/config/hq44k/rtn.yaml 

# python profile_encoder_latency.py --model_type vit_l --checkpoint ./ckts/sam_hq_vit_l.pth --runs 10 --warmup 3

CUDA_VISIBLE_DEVICES=1 python benchmark_batch_inference.py \
    --batch-sizes  4 \
    --num-samples 3000 \
    --processor BASE \
    --percent 0.7  \
    --percent-global 0.5 \
    --prune-global \
    --n-bits 16 \
    --high-entropy  \
    --model-type vit_h \
    --model-ckt ./pretrained_checkpoint/sam_hq_vit_h.pth \
    --quantize-encoder \
    # --svdq-rank 32 \
    # --svdq-precision int4 \
    # --en-weight-quant svdq \
    # --use-svdq \
    # --high-entropy \
    # --svdq-checkpoint ./output/svdq_sam_vit_l_r32_lowmem.pth \
