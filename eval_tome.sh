python benchmark_tome.py     \
    --algos  sparsesam_random sparsesam\
    --ratios 0.1  --sparsity 0.25 \
    --batch-sizes 2 \
    --model-ckt ./ckts/sam_hq_vit_l.pth  --model-type vit_l    \
    --num-samples 100 --no-wandb
