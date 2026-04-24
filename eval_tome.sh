python benchmark_tome.py     \
    --algos  sparsesam gradtome \
    --ratios 0.25  --sparsity 0.5 \
    --batch-sizes 2 \
    --model-ckt ./ckts/sam_hq_vit_l.pth  --model-type vit_l    \
    --num-samples 100 --no-wandb
