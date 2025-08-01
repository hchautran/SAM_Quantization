# torchrun \
#     --nproc_per_node=2 sam-hq/train/train.py \
#     --checkpoint ./pretrained_checkpoint/sam_vit_b.pth \
#     --model-type vit_b \
#     --output outputs \
#     --eval \
#     --restore-model ./work_dirs/hq_sam_vit_b/epoch_11.pth

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=1 quant/hq44k_engine_quan.py