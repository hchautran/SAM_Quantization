python benchmark_tome.py     \
    --algos  none sparsesam \
    --ratios 0.5  --sparsity 0.5 \
    --batch-sizes 4 \
    --model-ckt ./ckts/sam_hq_vit_l.pth  --model-type vit_l    \
    --num-samples 100 --no-wandb
