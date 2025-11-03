
# python profile_encoder_latency.py --model_type vit_l --checkpoint ./pretrained_checkpoint/sam_hq_vit_l.pth --runs 10 --warmup 3

python benchmark_batch_inference.py \
    --config-file quant/config/hq44k/rtn.yaml \
    --batch-sizes 1 2 4 8 16 32 64 \
    --num-samples 100 \
    --quantize-encoder \
    --n-bits 16 
