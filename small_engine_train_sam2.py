# small_engine_train_sam2.py
#!/usr/bin/env python3
"""
Unified training engine for SAM2 with processor support.

Combines training pipeline from train.py with entropy processor integration
from eval_sam2_hq44k.py, structured as a unified engine similar to small_engine_train.py.

Usage:
    # Local training with processor
    python small_engine_train_sam2.py \
        -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
        --use-cluster 0 \
        --num-gpus 8 \
        --processor POSITIONAL_PRUNE_SAM2 \
        --num-calib-samples 32

    # Cluster training with processor
    python small_engine_train_sam2.py \
        -c configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml \
        --use-cluster 1 \
        --num-gpus 8 \
        --num-nodes 2 \
        --processor HEAD_PRUNE_SAM2
"""

import os
import logging
import random
import sys
import traceback
from argparse import ArgumentParser
from functools import partial

import submitit
import torch
import numpy as np
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from hydra.core.global_hydra import GlobalHydra 

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.backbones.hieradet import MultiScaleAttention

# Local imports
from training.utils.train_utils import makedir, register_omegaconf_resolvers
from train.utils.dataloader import get_im_gt_name_dict, Resize
from data_utils import OnlineDataset
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou

# SAM2 entropy processors
from processors.encoder.entropy_sam2 import (
    PositionalPruneSAM2Processor,
    HeadPruneSAM2Processor,
    PositionalQuantSAM2Processor,
)
from processors.sam2_observer import sam2_image_encoder_monkey_patch

os.environ["HYDRA_FULL_ERROR"] = "1"


# Registry for SAM2 processors
SAM2_PROCESSOR_REGISTRY = {
    "POSITIONAL_PRUNE_SAM2": PositionalPruneSAM2Processor,
    "HEAD_PRUNE_SAM2": HeadPruneSAM2Processor,
    "POSITIONAL_QUANT_SAM2": PositionalQuantSAM2Processor,
}


def get_sam2_processor(name: str, **kwargs):
    """Get SAM2 processor by name."""
    if name not in SAM2_PROCESSOR_REGISTRY:
        available = list(SAM2_PROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown SAM2 processor '{name}'. Available: {available}")
    return SAM2_PROCESSOR_REGISTRY[name](**kwargs)


def custom_collate_fn(batch):
    """Custom collate function to handle variable-sized ori_im fields."""
    ori_ims = [item['ori_im'] for item in batch]
    collated = {}
    for key in batch[0].keys():
        if key == 'ori_im':
            collated[key] = ori_ims
        elif key == 'ori_im_path' or key == 'ori_gt_path':
            collated[key] = [item[key] for item in batch]
        else:
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except:
                collated[key] = [item[key] for item in batch]
    return collated


def format_exception(e: Exception, limit=20):
    """Format exception with traceback."""
    traceback_str = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{traceback_str}"


class SAM2TrainingEngine:
    """Unified engine for SAM2 training with processor support."""

    def __init__(
        self,
        cfg,
    ):

        self.cfg = cfg    
    def monkey_patch(self, predictor, processor_config=None,args_yaml= None ,train=False):
        
        pass
    def setup_and_calibrate_processors(self, predictor, args):
        print(f"\n{'='*80}")
        print(f"Setting up {args.processor}")
        print(f"{'='*80}\n")

        # Get processor
        print(args.processor)
        processor = get_sam2_processor(args.processor)

        # Create mock args for set_params (if config file not provided)
        if args.config_file:
            config = OmegaConf.load(args.config_file)
        else:
            # Create minimal config
            config = OmegaConf.create({
                'quantization': {
                    'percent_entropy': args.percent_entropy,
                    'percent_entropy_global': args.percent_entropy_global,
                    'high_entropy': args.high_entropy,
                    'prune_global': args.prune_global,
                    'threshold': args.threshold,
                }
            })

        # Set processor parameters
        processor.set_params(config)
        print(f"✓ Processor parameters set")
        print(f"  Percent: {processor.percent}")
        print(f"  Global Percent: {processor.global_percent}")
        print(f"  High entropy: {processor.prunehighentropy}")
        print(f"  Global: {processor.prune_global}\n")

        # Calibrate processor
        print("Calibrating processor...")
        processor.calibrate(
            predictor=predictor,
            modules=MultiScaleAttention,
            num_samples=args.num_calib_samples
        )
        print("✓ Processor calibrated\n")
    def eval_hq44k(self,predictor):
        ##TODO: implement HQ44K evaluation base on eval_sam2_hq44k.py
        pass
    def train_model(self, predictor, args_yaml):
        pass
        

def single_proc_run(local_rank, main_port, cfg, world_size):
    """Single GPU process"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info(e)
    import ipdb; ipdb.set_trace()
    trainer = instantiate(cfg.trainer, _recursive_=False)
    import ipdb; ipdb.set_trace()
    trainer.run()


def single_node_runner(cfg, main_port: int):
    assert cfg.launcher.num_nodes == 1
    num_proc = cfg.launcher.gpus_per_node
    torch.multiprocessing.set_start_method(
        "spawn"
    )  # CUDA runtime does not support `fork`
    if num_proc == 1:
        # directly call single_proc so we can easily set breakpoints
        # mp.spawn does not let us set breakpoints
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=num_proc)
    else:
        mp_runner = torch.multiprocessing.start_processes
        args = (main_port, cfg, num_proc)
        # Note: using "fork" below, "spawn" causes time and error regressions. Using
        # spawn changes the default multiprocessing context to spawn, which doesn't
        # interact well with the dataloaders (likely due to the use of OpenCV).
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="spawn")


def add_pythonpath_to_sys_path():
    """Add PYTHONPATH to sys.path."""
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def extract_config_name(config_path: str) -> str:
    """Extract config name from full path for Hydra compose."""
    # Remove .yaml extension if present
    if config_path.endswith('.yaml'):
        config_path = config_path[:-5]
    
    # If it's a full path, extract relative part
    if 'sam2_configs/' in config_path:
        # Extract everything after sam2_configs/
        config_name = config_path.split('sam2_configs/')[-1]
    elif '/' in config_path:
        # If it contains slashes but not sam2_configs, assume it's already relative
        config_name = config_path
    else:
        # Just a filename
        config_name = config_path
    
    return config_name

def main(args) -> None:
    """Main training entry point."""
    
    # Convert config path to proper config name for Hydra
    config_name = extract_config_name(args.config)
    
    cfg = compose(config_name=config_name)
    
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", args.config
        )
    
    print("###################### Train App Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("############################################################")

    add_pythonpath_to_sys_path()
    makedir(cfg.launcher.experiment_log_dir)
    
    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))

    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)

    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))
    
    
        
        
    ######### setup model with processor #########
    # engine = SAM2TrainingEngine('hq44k',args)
    # if args.processor:
    #     engine.setup_and_calibrate_processors(engine.model, args)
    
    
    ######## setup Submitit ########
    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing submitit config"

    submitit_dir = cfg.launcher.experiment_log_dir
    submitit_dir = os.path.join(submitit_dir, "submitit_logs")
    
    # Prioritize cmd line args
    cfg.launcher.gpus_per_node = (
        args.num_gpus if args.num_gpus is not None else cfg.launcher.gpus_per_node
    )
    cfg.launcher.num_nodes = (
        args.num_nodes if args.num_nodes is not None else cfg.launcher.num_nodes
    )
    submitit_conf.use_cluster = (
        args.use_cluster if args.use_cluster is not None else submitit_conf.use_cluster
    )
     
    cfg.launcher.num_nodes = 1
    main_port = random.randint(
        submitit_conf.port_range[0], submitit_conf.port_range[1]
    )
    single_node_runner(cfg, main_port)


if __name__ == "__main__":
    
    
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="path to config file (e.g. sam2.1_training/sam2.1_hiera_b+_MOSE_finetune or full path)",
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="whether to launch on a cluster, 0: run locally, 1: run on a cluster",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--qos", type=str, default=None, help="SLURM qos")
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="number of GPUS per node"
    )
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes")
    
    # Processor arguments
    parser.add_argument(
        "--processor",
        type=str,
        default=None,
        choices=[None, 'POSITIONAL_PRUNE_SAM2', 'HEAD_PRUNE_SAM2', 'POSITIONAL_QUANT_SAM2'],
        help="SAM2 entropy processor to use (None = no processing)",
    )
    parser.add_argument(
        "--num-calib-samples",
        type=int,
        default=32,
        help="Number of calibration samples for entropy processor",
    )
    parser.add_argument(
        "--percent-entropy",
        type=float,
        default=0.5,
        help="Percentage of heads to prune/quantize",
    )
    parser.add_argument(
        "--percent-entropy-global",
        type=float,
        default=0.3,
        help="Percentage of global heads to prune/quantize",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Entropy threshold for pruning",
    )
    parser.add_argument(
        "--high-entropy",
        action="store_true",
        help="Prune high entropy heads (default: prune low entropy)",
    )
    parser.add_argument(
        "--prune-global",
        action="store_true",
        help="Apply global pruning across all layers",
    )
    
    args = parser.parse_args()
    GlobalHydra.instance().clear() 
    
    # Check if custom config directory exists
    config_dir = "/home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/sam2_configs"
    if os.path.exists(config_dir):
        # Initialize with custom config directory
        from hydra import initialize_config_dir
        initialize_config_dir(config_dir=config_dir, version_base="1.2")
    else:
        # Fall back to sam2 config module
        initialize_config_module("sam2", version_base="1.2")
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None
    register_omegaconf_resolvers()
    main(args)