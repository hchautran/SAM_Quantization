import os
import sys
import argparse
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Semantic-Segmentation-Anything', 'scripts'))
from pipeline import semantic_segment_anything_inference, eval_pipeline, img_load
from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.cityscapes_id2label import CONFIG as CONFIG_CITYSCAPES_ID2LABEL


# Add imports for SAM quantization components
from omegaconf import OmegaConf
from small_engine import Engineade20k, override_args
from processors import get_encoder_processor, DecoderDoNothingProcessor

import torch.distributed as dist
import torch.multiprocessing as mp
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12327'

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything with SAM quantization.')
    
    # Original SSA arguments
    parser.add_argument('--data_dir', help='specify the root path of images and masks')
    parser.add_argument('--ckpt_path', default='ckp/sam_vit_h_4b8939.pth', help='specify the root path of SAM checkpoint')
    parser.add_argument('--out_dir', help='the dir to save semantic annotations')
    parser.add_argument('--save_img', default=False, action='store_true', help='whether to save annotated images')
    parser.add_argument('--world_size', type=int, default=0, help='number of nodes')
    parser.add_argument('--dataset', type=str, default='ade20k', choices=['ade20k', 'cityscapes', 'foggy_driving'], help='specify the set of class names')
    parser.add_argument('--eval', default=False, action='store_true', help='whether to execute evalution')
    parser.add_argument('--gt_path', default=None, help='specify the path to gt annotations')
    parser.add_argument('--model', type=str, default='segformer', choices=['oneformer', 'segformer'], help='specify the semantic branch model')
    
    # Benchmark/Quantization arguments
    parser.add_argument('--config-file', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--num-calib-samples', type=int, default=16,
                       help='Number of calibration samples')
    
    # Model parameters
    parser.add_argument('--processor', type=str, default='POSITIONAL_PRUNE',
                       choices=['BASE','POSITIONAL_PRUNE', 'POSITIONAL_QUANT', 'HEAD_PRUNE'],
                       help='Processor to use')
    parser.add_argument('--quantize-encoder', action='store_true',
                       help='Enable encoder quantization')
    parser.add_argument('--quantize-decoder', action='store_true',
                       help='Enable decoder quantization')

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
    return args

def main(rank, args):
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    
    # Load config for quantization
    config = OmegaConf.load(args.config_file)
    config = override_args(args, config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize SAM model with quantization setup
    print("Loading SAM model...")
    model_type = config.model.model_type
    checkpoint_path = config.model.hq_checkpoint
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(rank)
    predictor = SamPredictor(sam)

    # Initialize engine for quantization
    engine = Engineade20k(
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
    
    # Now set up the mask generator with the quantized SAM model
    mask_branch_model = SamAutomaticMaskGenerator(
        model=predictor.model,
        points_per_side=128 if args.dataset == 'foggy_driving' else 64,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
        output_mode='coco_rle',
    )
    print('[Model loaded] Mask branch (SAM) is loaded.')
    
    # Set up semantic branch model
    if args.model == 'oneformer':
        try:
            from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        except KeyError as e:
            if "'Version'" in str(e):
                print("Transformers library metadata error detected. Trying to reload...")
                # Try importing again after clearing metadata cache
                import importlib
                import sys
                
                # Clear any cached metadata
                if 'transformers' in sys.modules:
                    del sys.modules['transformers']
                
                # Force reimport
                from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
            else:
                raise e
        
        if args.dataset == 'ade20k':
            semantic_branch_processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large").to(rank)
        elif args.dataset == 'cityscapes':
            semantic_branch_processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_cityscapes_swin_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_cityscapes_swin_large").to(rank)
        elif args.dataset == 'foggy_driving':
            semantic_branch_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_dinat_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_cityscapes_dinat_large").to(rank)
        else:
            raise NotImplementedError()
    elif args.model == 'segformer':
        try:
            from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
        except KeyError as e:
            if "'Version'" in str(e):
                print("Transformers library metadata error detected. Trying to reload...")
                # Try importing again after clearing metadata cache
                import importlib
                import importlib.metadata
                import sys
                
                # Clear any cached metadata
                if 'transformers' in sys.modules:
                    del sys.modules['transformers']
                
                # Force reimport
                from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
            else:
                raise e
        
        if args.dataset == 'ade20k':
            semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b5-finetuned-ade-640-640")
            semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-ade-640-640").to(rank)
        elif args.dataset == 'cityscapes' or args.dataset == 'foggy_driving':
            semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
            semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(rank)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    print('[Model loaded] Semantic branch (your own segmentor) is loaded.')
    
    # Load image filenames
    if args.dataset == 'ade20k':
        filenames = [fn_.replace('.jpg', '') for fn_ in os.listdir(args.data_dir) if '.jpg' in fn_]
    elif args.dataset == 'cityscapes' or args.dataset == 'foggy_driving':
        sub_folders = [fn_ for fn_ in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, fn_))]
        filenames = []
        for sub_folder in sub_folders:
            filenames += [os.path.join(sub_folder, fn_.replace('.png', '')) for fn_ in os.listdir(args.data_dir + sub_folder) if '.png' in fn_]
    
    # Distribute filenames across processes
    local_filenames = filenames[(len(filenames) // args.world_size + 1) * rank : (len(filenames) // args.world_size + 1) * (rank + 1)]
    print('[Image name loaded] get image filename list.')
    print('[SSA start] model inference starts.')

    # Process each image
    for i, file_name in enumerate(local_filenames):
        print('[Running] ', i, '/', len(local_filenames), ' ', file_name, ' on rank ', rank, '/', args.world_size)
        img = img_load(args.data_dir, file_name, args.dataset)
        
        if args.dataset == 'ade20k':
            id2label = CONFIG_ADE20K_ID2LABEL
        elif args.dataset == 'cityscapes' or args.dataset == 'foggy_driving':
            id2label = CONFIG_CITYSCAPES_ID2LABEL
        else:
            raise NotImplementedError()
            
        with torch.no_grad():
            semantic_segment_anything_inference(
                file_name, args.out_dir, rank, img=img, save_img=args.save_img,
                semantic_branch_processor=semantic_branch_processor,
                semantic_branch_model=semantic_branch_model,
                mask_branch_model=mask_branch_model,
                dataset=args.dataset,
                id2label=id2label,
                model=args.model
            )
    
    # Run evaluation if requested
    if args.eval and rank == 0:
        assert args.gt_path is not None
        eval_pipeline(args.gt_path, args.out_dir, args.dataset)

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args.world_size > 1:
        mp.spawn(main, args=(args,), nprocs=args.world_size, join=True)
    else:
        main(0, args)