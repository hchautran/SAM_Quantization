export PYTHONPATH=$PYTHONPATH:$(pwd)/sam-hq:$(pwd)/Semantic-Segmentation-Anything/scripts

CUDA_VISIBLE_DEVICES=5 python benchmark_inference_ade20k.py \
    --config-file quant/config/coco/rtn.yaml \
    --quantize-encoder \
    --n-bits 16 \
    --num-calib-samples 2 \
    --processor POSITIONAL_PRUNE \
    --dataset ade20k \
    --gt_path ./data/ade/ADEChallengeData2016/annotations/validation\
    --data_dir ./data/ade/ADEChallengeData2016/images/validation \
    --out_dir ./benchmark_results/ade20k/positional_quant_vit_h_b0 \
    --world_size 1 \
    # --save_img 

# CUDA_VISIBLE_DEVICES=5 python ./Semantic-Segmentation-Anything/scripts/evaluation.py --gt_path ./data/ade/ADEChallengeData2016/annotations/validation --result_path ./benchmark_results/ade20k/base_vit_l_b0 --dataset ade20k
# CUDA_VISIBLE_DEVICES=5 python ./Semantic-Segmentation-Anything/scripts/evaluation.py --gt_path ./data/ade/ADEChallengeData2016/annotations/validation --result_path ./benchmark_results/ade20k/positional_prune_vit_l_b0 --dataset ade20k
# CUDA_VISIBLE_DEVICES=5 python ./Semantic-Segmentation-Anything/scripts/evaluation.py --gt_path ./data/ade/ADEChallengeData2016/annotations/validation --result_path ./benchmark_results/ade20k/positional_quant_vit_l_b0 --dataset ade20k
# CUDA_VISIBLE_DEVICES=5 python ./Semantic-Segmentation-Anything/scripts/evaluation.py --gt_path ./data/ade/ADEChallengeData2016/annotations/validation --result_path ./benchmark_results/ade20k/base_vit_b_b0 --dataset ade20k
# CUDA_VISIBLE_DEVICES=0 python ./Semantic-Segmentation-Anything/scripts/evaluation.py --gt_path ./data/ade/ADEChallengeData2016/annotations/validation --result_path ./benchmark_results/ade20k/positional_prune_vit_b_b0 --dataset ade20k
# CUDA_VISIBLE_DEVICES=1 python ./Semantic-Segmentation-Anything/scripts/evaluation.py --gt_path ./data/ade/ADEChallengeData2016/annotations/validation --result_path ./benchmark_results/ade20k/positional_quant_vit_b_b0 --dataset ade20k

