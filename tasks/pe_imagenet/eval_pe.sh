# List supported PE configs
python eval_pe_clip.py --list-models
# → PE-Core-G14-448, PE-Core-L14-336, PE-Core-B16-224, PE-Core-S16-384, PE-Core-T16-384

# Single dataset (zero-shot classification)
# python eval_pe_clip.py \
#     --model PE-Core-L14-336 \
#     --dataset cifar10 \
#     --dataset-root ./data/cifar10

# Sweep several classification datasets in one run
# python eval_pe_clip.py \
#     --model PE-Core-L14-336 \
#     --dataset cifar10 cifar100 \
#     --dataset-root './data/{dataset}' \
#     --batch-size 128

# sparsesam: block-sparse attention + Z-group permute + MLP token pruning
# `--ratio` is the keep-bar fraction (lower = more compression).
# `--no-mlp-prune` keeps full MLP if you only want to test the attention path.
# python eval_pe_clip.py \
#     --model PE-Core-L14-336 \
#     --dataset cifar10 cifar100 \
#     --dataset-root './data/{dataset}' \
#     --batch-size 128 \
#     --dtype fp16 \
#     --algorithm sparsesam \
#     --ratio 0.75 0.5  \
#     --group-size 4 --no-mlp-prune

# Retrieval (e.g. COCO captions)
# python eval_pe_clip.py \
#     --model PE-Core-L14-336 \
#     --dataset mscoco_captions \
#     --dataset-root ./data/coco \
#     --task zeroshot_retrieval

# Local checkpoint, fp16
python eval_pe_clip.py --model PE-Core-L14-336 \
    --dataset imagenet1k --dataset-root ./data/imagenet \
    --dtype fp16 --algorithm sparsesam_partial tome_partial \
    --ratio 0.75 0.5 0.25 \
    # --no-mlp-prune



