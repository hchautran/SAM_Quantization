# CUDA_VISIBLE_DEVICES=4 python small_engine.py --mode eval  --quantize-encoder --n-bits 8 --n-bits-mlp 8 --num-samples 400 --num-calib-samples 2 --config-file ./quant/config/hq44k/rtn.yaml 

# python profile_encoder_latency.py --model_type vit_l --checkpoint ./ckts/sam_hq_vit_l.pth --runs 10 --warmup 3
python benchmark_batch_inference.py \
    --batch-sizes 1  \
