export PYTHONPATH=$PYTHONPATH:$(pwd)/sam-hq:$(pwd)/Semantic-Segmentation-Anything/scripts

CUDA_VISIBLE_DEVICES=1,4 python benchmark_inference_ade20k.py \
    --config-file quant/config/coco/rtn.yaml \
    --quantize-encoder \
    --n-bits 16 \
    --num-calib-samples 16 \
    --processor POSITIONAL_QUANT \
    --dataset ade20k \
    --gt_path ./data/ade/ADEChallengeData2016/annotations/validation\
    --data_dir ./data/ade/ADEChallengeData2016/images/validation \
    --out_dir ./benchmark_results/ade20k/positional_quant_vit_l \
    --world_size 2 \
    # --save_img 