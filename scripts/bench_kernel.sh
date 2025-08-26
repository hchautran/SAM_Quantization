python3 /home/chau.th/SAM_Quantization/quant/qgemm/bench_int4.py \
    --sizes  1024x1024x4096 2048x2048x4096 4096x4096x8192 8096x8096x16192 \
    --warmup 5 --iters 20 --batch 1 2 4 8 16 32 --check-error