
import logging
import os
import random
import sys
import traceback
from argparse import ArgumentParser

# Set CUDA memory allocation configuration to help with fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# import submitit
import torch

from hydra import compose, initialize_config_module, initialize
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra

from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from training.utils.train_utils import makedir, register_omegaconf_resolvers
from processors.encoder.entropy_sam2 import (
    PositionalPruneSAM2Processor,
    HeadPruneSAM2Processor,
    PositionalQuantSAM2Processor,
)
os.environ["HYDRA_FULL_ERROR"] = "1"

SAM2_PROCESSOR_REGISTRY = {
    "POSITIONAL_PRUNE_SAM2": PositionalPruneSAM2Processor,
    "HEAD_PRUNE_SAM2": HeadPruneSAM2Processor,
    "POSITIONAL_QUANT_SAM2": PositionalQuantSAM2Processor,
}

def cleanup_distributed():
    """Properly cleanup distributed training resources"""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logging.warning(f"Error during distributed cleanup: {e}")

def register_resolvers_safe():
    """Safely register OmegaConf resolvers, avoiding duplicates"""
    try:
        register_omegaconf_resolvers()
    except ValueError as e:
        if "is already registered" in str(e):
            # Resolver already exists, skip registration
            logging.debug(f"Resolver already registered: {e}")
        else:
            raise e

def single_proc_run(local_rank, main_port, cfg, world_size):
    """Single GPU process with proper cleanup"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    trainer = None
    try:
        register_resolvers_safe()  # Use safe registration
        trainer = instantiate(cfg.trainer, _recursive_=False)
        trainer.run()
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        if trainer and hasattr(trainer, 'cleanup'):
            trainer.cleanup()
        cleanup_distributed()
        torch.cuda.empty_cache()


def single_node_runner(cfg, main_port: int):
    assert cfg.launcher.num_nodes == 1
    num_proc = cfg.launcher.gpus_per_node
    torch.multiprocessing.set_start_method(
        "spawn"
    )  # CUDA runtime does not support `fork`
    try:
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
    finally:
        # Ensure cleanup after multiprocessing
        torch.cuda.empty_cache()

def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path
    




def main(args):
    cfg = compose(config_name=args.config)
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

    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing submitit config"

    submitit_dir = cfg.launcher.experiment_log_dir
    submitit_dir = os.path.join(submitit_dir, "submitit_logs")
    # Priotrize cmd line args
    cfg.launcher.gpus_per_node = (
        args.num_gpus if args.num_gpus is not None else cfg.launcher.gpus_per_node
    )
    cfg.launcher.num_nodes = (
        args.num_nodes if args.num_nodes is not None else cfg.launcher.num_nodes
    )
    submitit_conf.use_cluster = (
        args.use_cluster if args.use_cluster is not None else submitit_conf.use_cluster
    )
    
    
    ########### 1 Node only #############
    cfg.launcher.num_nodes = 1
    main_port = random.randint(submitit_conf.port_range[0], submitit_conf.port_range[1])
    single_node_runner(cfg, main_port)

    def evaluate(self):
        """Evaluate the SAM2 model (method stub)."""
        raise NotImplementedError("Evaluation method is not implemented.")

    


if __name__ == "__main__":

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    # initialize_config_module("sam2", version_base="1.2")
    config_path ="/home/22chi.nh/project/SAMquantization/SAM_Quantization/sam2_configs"
    initialize(config_path="sam2_configs", version_base="1.2")
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="path to config file (e.g. configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml)",
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="whether to launch on a cluster, 0: run locally, 1: run on a cluster",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="number of GPUS per node"
    )
    # Entropy processor parameters
    parser.add_argument('--processor', type=str, default=None,
                       choices=[None, 'POSITIONAL_PRUNE_SAM2', 'HEAD_PRUNE_SAM2', 'POSITIONAL_QUANT_SAM2'],
                       help='SAM2 entropy processor to use (None = no processing)')
    parser.add_argument('--config-file', type=str, default=None,
                       help='Path to config YAML file for processor parameters')
    parser.add_argument('--num-calib-samples', type=int, default=32,
                       help='Number of calibration samples for entropy processor')
    
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes")
    args = parser.parse_args()
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None
    register_omegaconf_resolvers()
    
    main(args)