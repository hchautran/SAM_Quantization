
# python profile_encoder_latency.py --model_type vit_l --checkpoint ./pretrained_checkpoint/sam_hq_vit_l.pth --runs 10 --warmup 3

CUDA_VISIBLE_DEVICES=0 python small_engine.py \
    --mode eval  \
    --quantize-encoder \
    --n-bits 16 \
    --n-bits-mlp 16 \
    --num-samples 400 \
    --num-calib-samples 16 \
    --config-file ./quant/config/hq44k/rtn.yaml
