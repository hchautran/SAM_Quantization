import os
import gc
import time
import logging
import sys
import argparse
import datetime
import os.path as osp
import functools
from typing import List, Dict
import numpy as np
import pandas as pd
import torch

# SAM imports
from segment_anything import SamPredictor, sam_model_registry

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "cute"))
sys.path.insert(0, os.path.join(_HERE, "sam-hq"))
sys.path.insert(0, os.path.join(_HERE, "PiToMe"))

# Local imports
from small_engine import Engine, override_args, get_default_datasets
from processors import get_encoder_processor, DecoderDoNothingProcessor
from train.utils.dataloader import get_im_gt_name_dict, Resize
from data_utils import OnlineDataset
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou
from quant.configmmdet.utils_ import parse_argsptq4sam
from PiToMe.algo.registry import apply_sam as apply_tome, remove_all_sam as remove_tome

import mmcv
from mmcv import Config, DictAction
from mmcv.utils import get_logger
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from quant.configmmdet.det_observer_instance_sam_ import DetObserverInstanceSAM
from mmdet.apis import multi_gpu_test, single_gpu_test
from omegaconf import OmegaConf
from prunning_rate.samprunediff_duo import image_encoder_monkey_patch_train_duo_diff
from prunning_rate.sampruneduo import image_encoder_monkey_patch_train_duo


def _disable_sparse_cute_visualization():
    try:
        from flash_attn_rel_pos import FlashAttentionForwardAmpere
    except Exception:
        return

    if not hasattr(FlashAttentionForwardAmpere, "_visualize_mma"):
        FlashAttentionForwardAmpere._visualize_mma = lambda *args, **kwargs: None
    if not hasattr(FlashAttentionForwardAmpere, "_visualize"):
        FlashAttentionForwardAmpere._visualize = lambda *args, **kwargs: None
    if not hasattr(FlashAttentionForwardAmpere, "_visualize_copy_tv"):
        FlashAttentionForwardAmpere._visualize_copy_tv = lambda *args, **kwargs: None


def _enable_half_image_encoder_for_predictor(predictor: SamPredictor) -> None:
    predictor.model.image_encoder.half()
    encoder_dtype = next(predictor.model.image_encoder.parameters()).dtype
    decoder_dtype = next(predictor.model.mask_decoder.parameters()).dtype

    @torch.no_grad()
    def set_torch_image_half_encoder(transformed_image: torch.Tensor, original_image_size):
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == predictor.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {predictor.model.image_encoder.img_size}."
        predictor.reset_image()

        predictor.original_size = original_image_size
        predictor.input_size = tuple(transformed_image.shape[-2:])
        input_image = predictor.model.preprocess(transformed_image).to(dtype=encoder_dtype)
        features, interm_features = predictor.model.image_encoder(input_image)
        predictor.features = features.to(dtype=decoder_dtype)
        predictor.interm_features = [
            feat.to(dtype=decoder_dtype) if torch.is_floating_point(feat) else feat
            for feat in interm_features
        ]
        predictor.is_image_set = True

    predictor.set_torch_image = set_torch_image_half_encoder


class ImageEncoderCudaProfiler:
    def __init__(self, warmup_calls: int = 10, max_profile_calls: int = None):
        self.warmup_calls = max(0, warmup_calls)
        self.max_profile_calls = max_profile_calls
        self._event_pairs = []
        self._wrapped = False

    def wrap_forward(self, module: torch.nn.Module) -> None:
        if self._wrapped:
            return

        original_forward = module.forward

        @functools.wraps(original_forward)
        def wrapped_forward(*args, **kwargs):
            if (
                self.max_profile_calls is not None
                and len(self._event_pairs) >= self.max_profile_calls
            ):
                return original_forward(*args, **kwargs)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = original_forward(*args, **kwargs)
            end_event.record()
            self._event_pairs.append((start_event, end_event))
            return output

        module.forward = wrapped_forward
        self._wrapped = True

    def summarize(self):
        if not self._event_pairs:
            return None

        torch.cuda.synchronize()
        elapsed_ms = np.array(
            [start.elapsed_time(end) for start, end in self._event_pairs],
            dtype=np.float64,
        )
        measured_ms = elapsed_ms[self.warmup_calls :]
        if measured_ms.size == 0:
            measured_ms = elapsed_ms

        return {
            "calls_total": int(elapsed_ms.size),
            "warmup_skipped": min(self.warmup_calls, int(elapsed_ms.size)),
            "calls_measured": int(measured_ms.size),
            "total_ms": float(measured_ms.sum()),
            "mean_ms": float(measured_ms.mean()),
            "median_ms": float(np.median(measured_ms)),
            "min_ms": float(measured_ms.min()),
            "max_ms": float(measured_ms.max()),
            "fps": float(1000.0 / measured_ms.mean()) if measured_ms.mean() > 0 else float("inf"),
        }

    @staticmethod
    def format_summary(summary: Dict[str, float]) -> str:
        return (
            "\nImage Encoder CUDA Profile\n"
            f"  Calls (total): {summary['calls_total']}\n"
            f"  Warmup skipped: {summary['warmup_skipped']}\n"
            f"  Calls (measured): {summary['calls_measured']}\n"
            f"  Total: {summary['total_ms']:.3f} ms\n"
            f"  Mean: {summary['mean_ms']:.3f} ms\n"
            f"  Median: {summary['median_ms']:.3f} ms\n"
            f"  Min: {summary['min_ms']:.3f} ms\n"
            f"  Max: {summary['max_ms']:.3f} ms\n"
            f"  Encoder-only FPS: {summary['fps']:.3f}\n"
        )


def setup_logger(path_log,state):
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(path_log, f'{state}.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
def print_model_structure(model, title="Model Structure"):
    print(f"\n{title}")
    print("=" * len(title))
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")
    print("=" * len(title))
def evaluate_loadptq4sam(predictor, config_ ):
    state="coco_"
    if config_.quantization.quanrtn:
        state +="rtn"
    if config_.quantization.quansmooth:
        state += "smooth"
    if config_.quantization.quanro:
        state += "ro"
    if config_.quantization.quandecoder:
        state += "decoder "
    if config_.quantization.quangptq:
        state += "gptq "
    if config_.quantization.rtn_cuda:
        state += "rtncuda "
    if config_.quantization.gptq_cuda:
        state += "gptqcuda "
    if config_.quantization.low_high_density != "none":
        state+= "lh "+ config_.quantization.low_high_density + str(config_.quantization.percent)
    if config_.quantization.qkT_v:
        state += "qkTv " + str(config_.quantization.percent)
        if config_.quantization.channel:
            state += "channel " 
        else:
            state += "token "
    if config_.quantization.centerQ:
        state += "centerQ"
    
    # Use parse_argsptq4sam_with_unknown to handle additional arguments
    args, unknown_args = parse_argsptq4sam()
    
    state += args.detector + "_" + args.processor + "_" + config_.model.model_type
    # Parse additional arguments that are not recognized by parse_argsptq4sam
    additional_parser = argparse.ArgumentParser()
    additional_parser.add_argument('--num-calib-samples', type=int, default=16,
                                 help='Number of calibration samples')
    additional_parser.add_argument('--n-bits', type=int, default=16,
                                 help='Number of quantization bits')
    additional_parser.add_argument('--config-file', type=str,
                                 help='Path to config YAML file')
    additional_parser.add_argument('--quantize-encoder', action='store_true',
                                 help='Enable encoder quantization')
    additional_parser.add_argument('--quantize-decoder', action='store_true',
                                 help='Enable decoder quantization')
    additional_parser.add_argument('--checkpoint-path', type=str)
    additional_parser.add_argument('--num-samples', type=int, default=None,
                                 help='Profile using only the first N image encoder calls, then continue evaluation normally')
    additional_parser.add_argument('--profile-image-encoder', action='store_true',
                                 help='Profile SAM image encoder CUDA time during COCO eval')
    additional_parser.add_argument('--profile-warmup-calls', type=int, default=10,
                                 help='Number of initial image encoder calls to skip in the report')
    additional_parser.add_argument('--percent', type=float, default=None,
                                 help='SAM patch ratio override')
    additional_parser.add_argument('--merge-mlp', action='store_true',
                                 help='Whether to merge MLP layers in TOME variants')
    additional_args = additional_parser.parse_args(unknown_args)
    
    if args.detector == 'yolox':
        args.config = "./quant/configmmdet/yolox/yolo_l-sam-vit-l.py"
    elif args.detector == 'dino':
        args.config = "./quant/configmmdet/focalnet_dino/focalnet-l-dino_sam-vit-l.py"
    elif args.detector == 'hdetr':
        args.config = "./quant/configmmdet/hdetr/r50-hdetr_sam-vit-l.py"
        
    logger =setup_logger(config_.data.logging_path,state)
    print("---------------Logging file: ", state ,"------------------")
    logger.info("Model type: %s", config_.model.model_type)
    logger.info("Processor: %s", args.processor)
    logger.info("Detector: %s", args.detector)
    logger.info("Percent: %s", additional_args.percent)
    logger.info("Merge mlp: %s", additional_args.merge_mlp)
    # args.* for the original parse_argsptq4sam arguments
    # additional_args.* for the new arguments
    
    brecq = args.brecq
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    
    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg) 
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = os.path.abspath(cfg.plugin_dir)
                # Add parent directory to Python path
                parent_dir = os.path.dirname(os.path.dirname(plugin_dir))
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                # Get module path relative to parent_dir
                rel_path = os.path.relpath(plugin_dir, parent_dir)
                _module_path = rel_path.replace('/', '.')
                
                # Remove leading dots for absolute import
                _module_path = _module_path.lstrip('.')
                
                print(f"Importing module: {_module_path}")
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                # print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif (cfg.model.get('backbone', None) is not None
        and 'init_cfg' in cfg.model.backbone):
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                    'Because we only support single GPU mode in '
                    'non-distributed testing. Use the first GPU '
                    'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()

    # cfg.device = 'cpu'
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    
    # if args.q_config:
    #     q_config = utils.parse_config(args.q_config)

    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)
    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
   
    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(args.work_dir, f'{timestamp}.log')
        logger = get_logger(name='ptq4sam', log_file=log_file, log_level=logging.INFO)
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    
    # cali_data = utils.load_calibration(cfg, distributed, q_config.calibrate)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    # import ipdb; ipdb.set_trace()
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    checkpoint = {}
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    
    
    model.det_model.to(cfg.device)
    #TODO: implement quantization for SAM predictor = model.predictor
    # if want to quantize ptq4sam need to look back the github repo
    
    model.replace_quant_sam(predictor)
    # move predictor to device
    model.predictor.model.to(cfg.device)
    model.to(cfg.device)

    image_encoder_profiler = None
    if additional_args.profile_image_encoder:
        if not torch.cuda.is_available():
            print("Skipping image encoder profiling because CUDA is not available.")
        else:
            max_profile_calls = None
            if additional_args.num_samples is not None and additional_args.num_samples > 0:
                max_profile_calls = additional_args.profile_warmup_calls + additional_args.num_samples
            image_encoder_profiler = ImageEncoderCudaProfiler(
                warmup_calls=additional_args.profile_warmup_calls,
                max_profile_calls=max_profile_calls
            )
            image_encoder_profiler.wrap_forward(model.predictor.model.image_encoder)
    
    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        if args.show_dir is not None and 'gt' in args.show_dir:
            gt = True
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                    args.show_score_thr, gt=gt)
        else:
            gt = False
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                    args.show_score_thr, gt=gt)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)

        # In multi_gpu_test, if tmpdir is None, some tesnors
        # will init on cuda by default, and no device choice supported.
        # Init a tmpdir to avoid error on npu here.
        if cfg.device == 'npu' and args.tmpdir is None:
            args.tmpdir = './npu_tmpdir'

        outputs = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
            or cfg.evaluation.get('gpu_collect', False))

    rank, _ = get_dist_info()
    if rank == 0:
        if image_encoder_profiler is not None:
            profile_summary = image_encoder_profiler.summarize()
            if profile_summary is not None:
                profile_report = image_encoder_profiler.format_summary(profile_summary)
                print(profile_report)
                logger.info(profile_report.strip())
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            print(metric)
            # logger.info(q_config)
            logger.info(metric)
            metric_dict = dict(config=args.config, metric=metric)
            if args.work_dir is not None and rank == 0:
                mmcv.dump(metric_dict, json_file)


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark SAM encoder with batch inference'
    )

    # Config
    parser.add_argument('--config-file', type=str, required=True,
                       help='Path to config YAML file')

    # Benchmark parameters
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                       default=[1, 2, 4, 8, 16],
                       help='Batch sizes to test')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples per batch size')
    parser.add_argument('--num-calib-samples', type=int, default=16,
                       help='Number of calibration samples')

    # Model parameters
    parser.add_argument('--processor', type=str, default='POSITIONAL_QUANT',
                       choices=['BASE','POSITIONAL_PRUNE', 'POSITIONAL_SPARGE', 'PIECE_WISE_ATTN', 'POSITIONAL_QUANT', 'PRUNE_RATE','HEAD_PRUNE','PRUNE_RATE_SPARSE',"PRUNE_RATE_DUO", 'SPARSE_PARTIAL', 'TOME_PARTIAL', 'GRAD_TOME'],
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
    parser.add_argument("--checkpoint-path", type=str, default="./checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument('--percent', type=float, default=None,
                       help='SAM patch ratio override for SPARSE_PARTIAL / TOME_PARTIAL / GRAD_TOME')

    # Output
    parser.add_argument('--output-dir', type=str, default='./benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--profile-image-encoder', action='store_true',
                       help='Profile SAM image encoder CUDA time during COCO eval')
    parser.add_argument('--profile-warmup-calls', type=int, default=10,
                       help='Number of initial image encoder calls to skip in the report')
    
    parser.add_argument('--merge-mlp', action='store_true',
                       help='Whether to merge MLP layers in TOME variants')

    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config_file)
    config = override_args(args, config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    sam_patch_algos = {
        "SPARSE_PARTIAL": "sparsesam",
        "TOME_PARTIAL": "tome",
        "GRAD_TOME": "gradtome",
    }
    use_sam_patch = args.processor in sam_patch_algos

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
    enc_processor = None if use_sam_patch else get_encoder_processor(args.processor)

    # Setup and calibrate
    print(f"Calibrating {args.processor}...")
    if not use_sam_patch and args.processor != "PRUNE_RATE_DUO":
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

    if args.processor == "PRUNE_RATE_SPARSE" or args.processor == "PRUNE_RATE":
        
        image_encoder_monkey_patch_train_duo_diff(
                predictor.model,
                processor=enc_processor,
                args_yaml=config,
                train = False,
            )
        checkpoint_path = args.checkpoint_path
        # checkpoint_path= "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/ckts/prune_rate/diffduo_sam_hq_epoch_torchnograd_distill_balance10_vit_h_reg-weight_0.5_lr0.02_lr_drop2.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        predictor.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        if config.quantization.use_percentage:
            enc_processor.calculate_pruned_heads_per_layer_percent_based(predictor)
    elif args.processor == "PRUNE_RATE_DUO" :
        
        enc_processor.set_params(config)
        image_encoder_monkey_patch_train_duo(
                predictor.model,
                processor=enc_processor,
                args_yaml= config,
                train = False,
            )
        
        checkpoint_path = args.checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        predictor.model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)

        
        if config.quantization.use_percentage:
            enc_processor.calculate_pruned_heads_per_layer_percent_based(predictor)
    elif use_sam_patch:
        # _enable_half_image_encoder_for_predictor(predictor)
        ratio = args.percent
        if ratio is None:
            ratio = float(config.quantization.percent_entropy)
        if args.processor == "SPARSE_PARTIAL":
            _disable_sparse_cute_visualization()
        remove_tome(predictor.model.image_encoder, mask_decoder=predictor.model.mask_decoder)
        apply_tome(
            predictor.model.image_encoder,
            sam_patch_algos[args.processor],
            args=args,
            ratio=ratio,
            mlp_merge=args.merge_mlp,
        )

    else:
        engine.apply_quantization(predictor, encoder_config, decoder_config, config)
    evaluate_loadptq4sam(predictor, config)
if __name__ == '__main__':
    main()
