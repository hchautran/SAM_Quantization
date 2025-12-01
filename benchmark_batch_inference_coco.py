import os
import gc
import time
import logging
import sys
import argparse
import datetime
import os.path as osp
from typing import List, Dict
import numpy as np
import pandas as pd
import torch

# SAM imports
from segment_anything import SamPredictor, sam_model_registry

# Local imports
from small_engine import Engine, override_args, get_default_datasets
from processors import get_encoder_processor, DecoderDoNothingProcessor
from train.utils.dataloader import get_im_gt_name_dict, Resize
from data_utils import OnlineDataset
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou
from quant.configmmdet.utils_ import parse_argsptq4sam

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
    
    if not distributed:
        if args.show_dir is not None and 'gt' in args.show_dir:
            gt = True
        else:
            gt = False
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        # ipdb.set_trace()
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
    # config = OmegaConf.load(args.config_file)
    # config = override_args(args, config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    print("Loading SAM model...")
    
    model_type= args.model_type
    checkpoint_path = args.hq_checkpoint
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
    evaluate_loadptq4sam(predictor, config)
if __name__ == '__main__':
    main()