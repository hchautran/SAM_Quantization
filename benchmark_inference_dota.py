import torch
from tqdm import tqdm
import numpy as np
import cv2
import os
import argparse
from omegaconf import OmegaConf
from mmrotate.structures import RotatedBoxes
from mmrotate.utils import register_all_modules
from sam_mmrotate.data import build_data_loader, build_evaluator, build_visualizer
from sam_mmrotate.utils import show_box, show_mask
import matplotlib.pyplot as plt
from mmengine.structures import InstanceData
from segment_anything import sam_model_registry, SamPredictor
from mmrotate.structures import RotatedBoxes
from mmengine import ProgressBar
from mmdet.models.utils import samplelist_boxtype2tensor
from mmengine.registry import DATA_SAMPLERS, FUNCTIONS, EVALUATOR, VISUALIZERS
from processors import get_encoder_processor, DecoderDoNothingProcessor

from small_engine import Engine, override_args, get_default_datasets
register_all_modules(init_default_scope=True)

SHOW = False
FORMAT_ONLY = False
MERGE_PATCHES = False




if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Benchmark SAM encoder with batch inference'
    )

    # Config
    parser.add_argument('--config-file', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num-calib-samples', type=int, default=16,
                       help='Number of calibration samples')

    # Model parameters
    parser.add_argument('--processor', type=str, default='POSITIONAL_QUANT',
                       choices=['BASE','POSITIONAL_PRUNE', 'POSITIONAL_QUANT', 'HEAD_PRUNE'],
                       help='Processor to use')
    parser.add_argument('--quantize-encoder', action='store_true',
                       help='Enable encoder quantization')
    parser.add_argument('--quantize-decoder', action='store_true',
                       help='Enable decoder quantization')
    parser.add_argument('--detector',type=str, default='yolo',
                        choices=['yolox', 'dino', "hdetr"])

    # Quantization parameters
    parser.add_argument('--n-bits', type=int, default=16,
                       help='Number of quantization bits')
    parser.add_argument('--n-bits-mlp', type=int, default=4,
                       help='Number of quantization bits for MLP')
    parser.add_argument('--en-weight-quant', type=str, default='per_channel',
                       help='Encoder weight quantization method')
    parser.add_argument('--en-act-quant', type=str, default='per_token',
                       help='Encoder activation quantization method')
    parser.add_argument('--de-weight-quant', type=str, default='per_channel',
                       help='Decoder weight quantization method')
    parser.add_argument('--de-act-quant', type=str, default='per_token',
                       help='Decoder activation quantization method')
    parser.add_argument('--k-preserve', type=int, default=0,
                       help='Number of channels to preserve')

    # Output
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config_file)
    config = override_args(args, config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    print("Loading SAM model...")
    
    model_type= config.model.model_type
    checkpoint_path = config.model.hq_checkpoint
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)

    # Initialize engine
    engine = Engine(
        'batch_benchmark',
        quantize_encoder=args.quantize_encoder,
        quantize_decoder=args.quantize_decoder
    )

    # Get processor
    enc_processor = get_encoder_processor(args.processor)

    # Setup and calibrate
    print(f"Calibrating {args.processor}...")
    encoder_processor, decoder_processor = engine.setup_and_calibrate_processors(
        predictor,
        num_calib_samples=args.num_calib_samples,
        encoder_processor=enc_processor,
        decoder_processor=DecoderDoNothingProcessor("DO_NOTHING"),
        args_yaml=config,
    )

    # Apply quantization
    encoder_config = {
        'processor': encoder_processor,
        'n_bits': args.n_bits,
        'weight_quant': args.en_weight_quant,
        'act_quant': args.en_act_quant,
    } if args.quantize_encoder else None

    decoder_config = {
        'processor': decoder_processor,
        'n_bits': args.n_bits,
        'weight_quant': args.de_weight_quant,
        'act_quant': args.de_act_quant,
        'k_preserve': args.k_preserve
    } if args.quantize_decoder else None

    engine.apply_quantization(predictor, encoder_config, decoder_config, config)
    
    dataloader = build_data_loader('trainval_with_hbox')
    evaluator = build_evaluator(MERGE_PATCHES, FORMAT_ONLY)
    evaluator.dataset_meta = dataloader.dataset.metainfo


    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

        img = data['inputs'][0].permute(1, 2, 0).numpy()[:, :, ::-1]
        data_samples = data['data_samples']
        data_sample = data_samples[0]
        data_sample = data_sample.to(device=args.device)
        image_name= data['data_samples'][0].img_id + '.txt'
  
        # import ipdb; ipdb.set_trace()
        if (len(data_sample.h_gt_bboxes) >= 215):
            data_sample.gt_instances = data_sample.gt_instances[:215]
        h_bboxes = data_sample.h_gt_bboxes.tensor.to(device=args.device)[:215]
        labels = data_sample.gt_instances.labels.to(device=args.device)

        r_bboxes = []
        if len(h_bboxes) == 0:
            qualities = h_bboxes[:, 0]
            masks = h_bboxes.new_tensor((0, *img.shape[:2]))
        else:
            predictor.set_image(img)
            transformed_boxes = predictor.transform.apply_boxes_torch(h_bboxes, img.shape[:2])
            masks, qualities, lr_logits = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False)
            masks = masks.squeeze(1)
            qualities = qualities.squeeze(-1)
            for mask in masks:
                y, x = np.nonzero(mask.cpu().numpy())
                points = np.stack([x, y], axis=-1)
                (cx, cy), (w, h), a = cv2.minAreaRect(points)
                r_bboxes.append(np.array([cx, cy, w, h, a/180*np.pi]))

        results = InstanceData()
        results.bboxes = RotatedBoxes(r_bboxes)
        results.scores = qualities
        results.labels = labels
        results.masks = masks.cpu().numpy()
        results_list = [results]

        # add_pred_to_datasample
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)

        evaluator.process(data_samples=data_samples, data_batch=data)

        if SHOW:
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            for box in h_bboxes:
                show_box(box.cpu().numpy(), plt.gca())
            plt.axis('off')
            # plt.show()
            plt.savefig(f'./benchmark_results/DOTA/out_mask_{i}.png')

            # draw rbox with mmrotate
            visualizer = build_visualizer()
            visualizer.dataset_meta = dataloader.dataset.metainfo
            out_img = visualizer._draw_instances(
                img, results,
                dataloader.dataset.metainfo['classes'],
                dataloader.dataset.metainfo['palette'])
            # visualizer.show()
            cv2.imwrite(f'./benchmark_results/DOTA/out_rbox_{i}.png', out_img[:, :, ::-1])

    metrics = evaluator.evaluate(len(dataloader.dataset))
