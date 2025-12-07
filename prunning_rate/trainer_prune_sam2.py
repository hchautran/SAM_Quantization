from training.trainer import Trainer, LoggingConf, CheckpointConf, OptimConf, DistributedConf, CudaConf
from typing import Dict, Any, Optional, List,Mapping , Set, Callable
from training.optimizer import Optimizer, _unix_pattern_to_parameter_names, map_scheduler_cfgs_to_param_groups, get_module_cls_to_param_names, set_default_parameters
from training.utils.logger import Logger, setup_logging
from training.utils.data_utils import BatchedVideoDatapoint
from training.utils.distributed import all_reduce_max, barrier, get_rank
from training.utils.train_utils import (
    AverageMeter,
    collect_dict_keys,
    DurationMeter,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    human_readable_time,
    is_dist_avail_and_initialized,
    log_env_variables,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    set_seeds,
    setup_distributed_backend,
)
from omegaconf import OmegaConf
from processors.encoder.entropy_sam2 import (
    PositionalPruneSAM2Processor,
    HeadPruneSAM2Processor,
    PositionalQuantSAM2Processor,
    PositionalTrainingPruneRateSAM2Processor,
)
from sam2.modeling.backbones.hieradet import MultiScaleAttention
from sam2.sam2_image_predictor import SAM2ImagePredictor
from prunning_rate.sam2prune import monkey_patch_train_sam2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from iopath.common.file_io import g_pathmgr
import hydra
from hydra.utils import instantiate
import time
from collections import OrderedDict
import math
import wandb
import logging
from eval_sam2_hq44k import SAM2Evaluator, custom_collate_fn
from data_utils import OnlineDataset
from .utils import get_default_datasets
from train.utils.dataloader import get_im_gt_name_dict, Resize
from torchvision import transforms

SAM2_PROCESSOR_REGISTRY = {
    "POSITIONAL_PRUNE_SAM2": PositionalPruneSAM2Processor,
    "HEAD_PRUNE_SAM2": HeadPruneSAM2Processor,
    "POSITIONAL_QUANT_SAM2": PositionalQuantSAM2Processor,
    "TRAINING_PRUNE_RATE_SAM2": PositionalTrainingPruneRateSAM2Processor,
}
def get_sam2_processor(name: str, **kwargs):
    """Get SAM2 processor by name."""
    if name not in SAM2_PROCESSOR_REGISTRY:
        available = list(SAM2_PROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown SAM2 processor '{name}'. Available: {available}")
    return SAM2_PROCESSOR_REGISTRY[name](**kwargs)
def print_model_structure(model, title="Model Structure"):
    print(f"\n{title}")
    print("=" * len(title))
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")
    print("=" * len(title))
def construct_optimizer(
    model: torch.nn.Module,
    optimizer_conf: Any,
    options_conf: Mapping[str, List] = None,
    param_group_modifiers_conf: List[Callable] = None,
    param_allowlist: Optional[Set[str]] = None,
    validate_param_groups=True,
) -> Optimizer:
    """
    Constructs a stochastic gradient descent or ADAM (or ADAMw) optimizer
    with momentum. i.e, constructs a torch.optim.Optimizer with zero-weight decay
    Batchnorm and/or no-update 1-D parameters support, based on the config.

    Supports wrapping the optimizer with Layer-wise Adaptive Rate Scaling
    (LARS): https://arxiv.org/abs/1708.03888

    Args:
        model: model to perform stochastic gradient descent
            optimization or ADAM optimization.
        optimizer_conf: Hydra config consisting a partial torch optimizer like SGD or
            ADAM, still missing the params argument which this function provides to
            produce the final optimizer
        param_group_modifiers_conf: Optional user specified functions which can modify
            the final scheduler configs before the optimizer's param groups are built
        param_allowlist: The parameters to optimize. Parameters which are not part of
            this allowlist will be skipped.
        validate_param_groups: If enabled, valides that the produced param_groups don't
            overlap and cover all the model parameters.
    """
    if param_allowlist is None:
        param_allowlist = {name for name, _ in model.named_parameters()}

    named_parameters = {
        name: param
        for name, param in model.named_parameters()
        if name in param_allowlist
    }

    if not options_conf:
        optimizer = hydra.utils.instantiate(optimizer_conf, named_parameters.values())
        return Optimizer(optimizer)

    all_parameter_names = {
        name for name, _ in model.named_parameters() if name in param_allowlist
    }
    module_cls_to_all_param_names = get_module_cls_to_param_names(
        model, param_allowlist
    )

    scheduler_cfgs_per_option = hydra.utils.instantiate(options_conf)
    all_scheduler_cfgs = []
    for option, scheduler_cfgs in scheduler_cfgs_per_option.items():
        for config in scheduler_cfgs:
            config.option = option
            config.parameter_names = _unix_pattern_to_parameter_names(
                config, all_parameter_names, module_cls_to_all_param_names
            )
  
        set_default_parameters(scheduler_cfgs, all_parameter_names)
        all_scheduler_cfgs.append(scheduler_cfgs)
   
    filtered_scheduler_cfgs = []
    for scheduler_cfg_group in all_scheduler_cfgs:
        filtered_group = []
        for scheduler_cfg in scheduler_cfg_group:
            # Keep only configs that have both 'scheduler' and 'option' keys
            if 'scheduler' in scheduler_cfg and 'option' in scheduler_cfg:
                filtered_group.append(scheduler_cfg)
            else:
                print(f"Filtering out malformed config: {scheduler_cfg}")
        
        # Only add the group if it has valid configs
        if filtered_group:
            filtered_scheduler_cfgs.append(filtered_group)

    all_scheduler_cfgs = filtered_scheduler_cfgs
    
    if param_group_modifiers_conf:
        for custom_param_modifier in param_group_modifiers_conf:
            custom_param_modifier = hydra.utils.instantiate(custom_param_modifier)
            all_scheduler_cfgs = custom_param_modifier(
                scheduler_cfgs=all_scheduler_cfgs, model=model
            )
    schedulers, param_groups = map_scheduler_cfgs_to_param_groups(
        all_scheduler_cfgs, named_parameters
    )

    optimizer = hydra.utils.instantiate(optimizer_conf, param_groups)
    return Optimizer(optimizer, schedulers)
class TrainerPruneRate(Trainer):
    EPSILON = 1e-8

    def __init__(
        self,
        *,  # the order of these args can change at any time, so they are keyword-only
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        accelerator: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        optim_overrides: Optional[List[Dict[str, Any]]] = None,
        meters: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        processor_args: Optional[Dict[str, Any]] = None,
        target_flop: int = 1,   # 1e11
        flops_scale: int = 1, 
    ):

        self._setup_env_variables(env_variables)
        self._setup_timers()
        
        self.flops_scale = flops_scale
        self.target_flop = target_flop
        self.data_conf = data
        self.model_conf = model
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint).infer_missing()
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.optim_conf = OptimConf(**optim) if optim is not None else None
        self.meters_conf = meters
        self.loss_conf = loss
        distributed = DistributedConf(**distributed or {})
        cuda = CudaConf(**cuda or {})
        self.where = 0.0

        self._infer_distributed_backend_if_none(distributed, accelerator)

        self._setup_device(accelerator)

        self._setup_torch_dist_and_backend(cuda, distributed)

        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
        )

        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        log_env_variables()

        # assert (
        #     is_dist_avail_and_initialized()
        # ), "Torch distributed needs to be initialized before calling the trainer."

        self._setup_components()  # Except Optimizer everything is setup here.
        self._move_to_device()
        
        self._setup_dataloaders()

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")

        if self.checkpoint_conf.resume_from is not None:
            assert os.path.exists(
                self.checkpoint_conf.resume_from
            ), f"The 'resume_from' checkpoint {self.checkpoint_conf.resume_from} does not exist!"
            dst = os.path.join(self.checkpoint_conf.save_dir, "checkpoint.pt")
            if self.distributed_rank == 0 and not os.path.exists(dst):
                # Copy the "resume_from" checkpoint to the checkpoint folder
                # if there is not a checkpoint to resume from already there
                makedir(self.checkpoint_conf.save_dir)
                g_pathmgr.copy(self.checkpoint_conf.resume_from, dst)
            barrier()
        self.load_checkpoint()
        
        # Setup processor for traininig the pruning rate
        self.processor_args = processor_args
        self.setup_processor(self.processor_args)
        self._move_to_device()
        self._construct_optimizers()
        self._initialize_wandb()
        ##############
        self._setup_ddp_distributed_training(distributed, accelerator)
        barrier()
    def verify_gradient_calculation(self, step):
        """Verify that gradients are calculated only for selected_probability parameters. Add this method after backward in _run_tep method  to check"""
        if step % 10 == 0:  # Log every 10 steps to avoid spam
            print(f"\n=== GRADIENT VERIFICATION - Step {step} ===")
            
            params_with_grad = []
            params_should_have_grad = []
            unexpected_grads = []
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    has_gradient = param.grad is not None and torch.any(param.grad != 0)
                    
                    if "selected_probability" in name:
                        params_should_have_grad.append(name)
                        if has_gradient:
                            params_with_grad.append(name)
                            print(f"✓ CORRECT: {name} has gradient (norm: {param.grad.norm().item():.6f})")
                        else:
                            print(f"⚠ WARNING: {name} requires_grad=True but no gradient!")
                    else:
                        # This shouldn't happen based on your monkey patching
                        if has_gradient:
                            unexpected_grads.append(name)
                            print(f"❌ UNEXPECTED: {name} has gradient but shouldn't!")
            
            print(f"\nSummary:")
            print(f"  Expected parameters with gradients: {len(params_should_have_grad)}")
            print(f"  Actually calculated gradients: {len(params_with_grad)}")
            print(f"  Unexpected gradients: {len(unexpected_grads)}")
            
            if len(params_with_grad) == len(params_should_have_grad) and len(unexpected_grads) == 0:
                print(f"PERFECT: All {len(params_with_grad)} selected_probability parameters have gradients!")
            print("=" * 50)
    def _initialize_wandb(self):
        """Initialize wandb logging"""
        if self.distributed_rank == 0:  # Only log from main process
            # Extract model type and learning rate info
            model_type = self.processor_args.get('model_type', 'unknown')
            base_lr = self.optim_conf.options.get('lr', [{}])[0].get('scheduler', {}).get('start_value', 'unknown')
            vision_lr = None
            if len(self.optim_conf.options.get('lr', [])) > 1:
                vision_lr = self.optim_conf.options['lr'][1].get('scheduler', {}).get('start_value', 'unknown')
            
            # Create project name
            project_name = f"sam2_prune_{model_type}_base_{base_lr}" + "target_flop-" + str(self.target_flop) + f"_flopscale_{self.flops_scale}"
            if vision_lr:
                project_name += f"_vision_{vision_lr}"
            
            wandb.init(
                name = project_name,
                project=f"SAM2_Pruning_Rate_Training_{model_type}",
                config={
                    "model_type": model_type,
                    "base_lr": base_lr,
                    "vision_lr": vision_lr,
                    "target_flop": self.target_flop,
                    "flops_scale": self.flops_scale,
                    "max_epochs": self.max_epochs,
                    "batch_size": self.data_conf.get('train', {}).get('batch_sizes', [None])[0]
                }
            )
            print(f"Wandb initialized with project: {project_name}")
    def setup_processor(self, args):
        print(f"\n{'='*80}")
        print(f"Setting up {args.processor}")
        print(f"{'='*80}\n")
        predictor = SAM2ImagePredictor(self.model)
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
       
        # Apply monkey patch to integrate processor into self.model
        print("Applying monkey patch to SAM2 image encoder...")
        monkey_patch_train_sam2(
            model=self.model,
            processor=processor,
            model_type= args.model_type,
            train=args.train,
        )
        print("Monkey patch applied.")
        
    
    def _step(
        self,
        batch: BatchedVideoDatapoint,
        model: nn.Module,
        phase: str,
    ):
        
        outputs, flops = model(batch)
        targets = batch.masks
        batch_size = len(batch.img_batch)

        key = batch.dict_key  # key for dataset
        
        loss_flops = (flops/1e11 - self.target_flop)**2
        loss = self.loss[key](outputs, targets)
        loss_str = f"Losses/{phase}_{key}_loss"
        
        # Log to wandb
        if self.distributed_rank == 0:
            wandb_log = {
                "step": self.steps[phase],
                "epoch": self.epoch,
                "flops": flops/1e11,
                "flops_loss": self.flops_scale * loss_flops.item(),
                "target_flop": self.target_flop
            }
            
            # Log individual loss components if loss is a dict
            if isinstance(loss, dict):
                for loss_name, loss_val in loss.items():
                    wandb_log[f"loss/{loss_name}"] = loss_val.item() if hasattr(loss_val, 'item') else loss_val
            else:
                wandb_log["loss/total"] = loss.item() if hasattr(loss, 'item') else loss
            
            wandb.log(wandb_log)

        loss_log_str = os.path.join("Step_Losses", loss_str)

        # loss contains multiple sub-components we wish to log
        step_losses = {}
        if isinstance(loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()}
            )
            loss = self._log_loss_detailed_and_return_core_loss(
                loss, loss_log_str, self.steps[phase]
            )

        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(
                loss_log_str,
                loss,
                self.steps[phase],
            )

        self.steps[phase] += 1
        
        ################ add flops loss ################
        loss = loss + self.flops_scale*loss_flops
        if self.distributed_rank == 0 :
            wandb.log({
                "loss/total_with_flops": loss.item() if hasattr(loss, 'item') else loss,
            })
        ################################################
        ret_tuple = {loss_str: loss}, batch_size, step_losses

        if phase in self.meters and key in self.meters[phase]:
            meters_dict = self.meters[phase][key]
            if meters_dict is not None:
                for _, meter in meters_dict.items():
                    meter.update(
                        find_stages=outputs,
                        find_metadatas=batch.metadata,
                    )

        return ret_tuple
    def _construct_optimizers(self):

        param_allowlist=[]
        for name, param in self.model.named_parameters():
            if 'selected_probability' in name:
                param_allowlist.append(name)
                print(f"Training parameter: {name}")
        self.optim = construct_optimizer(
            self.model,
            self.optim_conf.optimizer,
            self.optim_conf.options,
            self.optim_conf.param_group_modifiers,
            param_allowlist=param_allowlist,
        )

    def run_val(self):
        self.run_val_hq44k()

        ##TODO : implement evaluation on other datasets here


    def run_val_hq44k(self):
        predictor = SAM2ImagePredictor(self.model.module)
        datasets = get_default_datasets()
        valid_im_gt_list = get_im_gt_name_dict([datasets[0]], flag="valid")

        gos_dataset = OnlineDataset(
            [valid_im_gt_list[0]],
            transform=transforms.Compose([Resize([1024, 1024])]),
            eval_ori_resolution=True
        )

        dataloader = DataLoader(
            gos_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            pin_memory=True ,
            collate_fn=custom_collate_fn if 1 > 1 else None,
        )
        evaluator = SAM2Evaluator()
        results = evaluator.eval_hq44k(
            predictor=predictor,
            dataloader=dataloader,
            num_samples=100,
            use_batch=False
        )
        ################ log wandb results################
        if self.distributed_rank == 0:
            wandb_eval_log = {
                "eval/miou": results['miou'],
                "eval/miou_std": results['miou_std'],
                "eval/boundary_iou": results['boundary_iou'],
                "eval/boundary_iou_std": results['boundary_iou_std'],
                "eval/num_samples": results['num_samples'],
                "epoch": self.epoch,
                "step": self.steps.get('train', 0)  # Use training step count
            }
            wandb.log(wandb_eval_log)
        del dataloader
        del evaluator
    def train_epoch(self, train_loader):

        # Init stat meters
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f")
        data_time_meter = AverageMeter("Data Time", self.device, ":.2f")
        mem_meter = MemMeter("Mem (GB)", self.device, ":.2f")
        data_times = []
        phase = Phase.TRAIN

        iters_per_epoch = len(train_loader)

        loss_names = []
        for batch_key in self.loss.keys():
            loss_names.append(f"Losses/{phase}_{batch_key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        progress = ProgressMeter(
            iters_per_epoch,
            [
                batch_time_meter,
                data_time_meter,
                mem_meter,
                self.time_elapsed_meter,
                *loss_mts.values(),
            ],
            self._get_meters([phase]),
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        # Model training loop
        self.model.train()
        end = time.time()

        for data_iter, batch in enumerate(train_loader):
            # measure data loading time
            data_time_meter.update(time.time() - end)
            data_times.append(data_time_meter.val)
            batch = batch.to(
                self.device, non_blocking=True
            )  # move tensors in a tensorclass

            try:
                self._run_step(batch, phase, loss_mts, extra_loss_mts)
                
                # Explicit memory cleanup after each batch to prevent OOM
                # This is especially important for variable batch sizes in SAM2
                torch.cuda.empty_cache()
                
                # Delete batch reference to help with memory cleanup
                del batch
                
                # compute gradient and do optim step
                exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
                self.where = float(exact_epoch) / self.max_epochs
                assert self.where <= 1 + self.EPSILON
                if self.where < 1.0:
                    self.optim.step_schedulers(
                        self.where, step=int(exact_epoch * iters_per_epoch)
                    )
                else:
                    logging.warning(
                        f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                    )

                # Log schedulers
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for j, param_group in enumerate(self.optim.optimizer.param_groups):
                        for option in self.optim.schedulers[j]:
                            optim_prefix = (
                                "" + f"{j}_"
                                if len(self.optim.optimizer.param_groups) > 1
                                else ""
                            )
                            self.logger.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )

                # Clipping gradients and detecting diverging gradients
                if self.gradient_clipper is not None:
                    self.scaler.unscale_(self.optim.optimizer)
                    self.gradient_clipper(model=self.model)

                if self.gradient_logger is not None:
                    self.gradient_logger(
                        self.model, rank=self.distributed_rank, where=self.where
                    )

                # Optimizer step: the scaler will make sure gradients are not
                # applied if the gradients are infinite
             
                self.scaler.step(self.optim.optimizer)
                self.scaler.update()

                

                # measure elapsed time
                batch_time_meter.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )

                mem_meter.update(reset_peak_usage=True)

                # Log training time to wandb
                if self.distributed_rank == 0 and data_iter % self.logging_conf.log_scalar_frequency == 0:
                    wandb.log({
                        "time/batch_time": batch_time_meter.val,
                        "time/data_time": data_time_meter.val,
                        "time/elapsed_time": self.time_elapsed_meter.val,
                        "memory/gpu_memory_gb": mem_meter.val,
                        "step": self.steps[phase]
                    })

                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)

                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    # Log progress meters.
                    for progress_meter in progress.meters:
                        self.logger.log(
                            os.path.join("Step_Stats", phase, progress_meter.name),
                            progress_meter.val,
                            self.steps[phase],
                        )

            # Catching NaN/Inf errors in the loss
            except FloatingPointError as e:
                raise e

        self.est_epoch_time[Phase.TRAIN] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(Phase.TRAIN)
        self._log_sync_data_times(Phase.TRAIN, data_times)

        out_dict = self._log_meters_and_save_best_ckpts([Phase.TRAIN])

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg
        out_dict.update(self._get_trainer_state(phase))
        logging.info(f"Losses and meters: {out_dict}")
        self._reset_meters([phase])
        return out_dict
    
