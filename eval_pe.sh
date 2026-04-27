# List supported PE configs
python eval_pe_clip.py --list-models
# → PE-Core-G14-448, PE-Core-L14-336, PE-Core-B16-224, PE-Core-S16-384, PE-Core-T16-384

# Single dataset (zero-shot classification)
# python eval_pe_clip.py \
#     --model PE-Core-L14-336 \
#     --dataset cifar10 \
#     --dataset-root ./data/cifar10

# Sweep several classification datasets in one run
python eval_pe_clip.py \
    --model PE-Core-L14-336\
    --dataset cifar10 cifar100 \
    --dataset-root './data/{dataset}' \
    --batch-size 128 \
    # --algorithm flash_rope

# Retrieval (e.g. COCO captions)
python eval_pe_clip.py \
    --model PE-Core-L14-336 \
    --dataset mscoco_captions \
    --dataset-root ./data/coco \
    --task zeroshot_retrieval \
    # --algorithm flash_rope

# Local checkpoint, fp16
# python eval_pe_clip.py --model PE-Core-G14-448 \
#     --checkpoint ./pe_ckts/PE-Core-G14-448.pt \
#     --dataset imagenet1k --dataset-root ./data/imagenet \
#     --dtype fp16
