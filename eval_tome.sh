python benchmark_tome.py     \
    --algos  sparsesam sparsesam_random\
    --ratios 0.75 0.5  0.25 0.1 0.05 --sparsity 0.25 \
    --batch-sizes 2 \
    --model-ckt ./ckts/sam_hq_vit_l.pth  --model-type vit_l    \
    --num-samples 100 --no-wandb
