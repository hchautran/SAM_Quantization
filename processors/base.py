"""Base processor classes and utilities for SAM quantization."""

import torch
import torch.nn as nn
from functools import partial
from abc import abstractmethod
from accelerate import Accelerator
from tqdm.auto import tqdm
import logging
import os
from typing import Union 

from train.utils.dataloader import get_im_gt_name_dict, Resize
from data_utils import OnlineDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from segment_anything import SamPredictor
from seginw.segment_anything import SamPredictor as SamPredictor_
import train.utils.misc as misc


def create_calib_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1):
    """Create calibration dataloaders from image-ground truth name lists."""
    gos_dataloaders = []
    gos_datasets = []
    for i in range(len(name_im_gt_list)):
        gos_dataset = OnlineDataset([name_im_gt_list[i]], transform=transforms.Compose(my_transforms), eval_ori_resolution=True)
        dataloader = DataLoader(gos_dataset, batch_size, drop_last=False)
        gos_dataloaders.append(dataloader)
        gos_datasets.append(gos_dataset)
    return gos_dataloaders, gos_datasets


def setup_logger(path_log, state):
    """Setup logger for calibration logs."""
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


class AttentionProcessor():
    """Base class for attention processors."""

    def __init__(self, strategy_name: str) -> None:
        self.accelerator = Accelerator()
        self.strategy_name = strategy_name
        self.device = self.accelerator.device
        self.stat = {}
        dataset_dis = {
            "name": "DIS5K-VD",
            "im_dir": "./data/DIS5K/DIS-VD/im",
            "gt_dir": "./data/DIS5K/DIS-VD/gt",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        }
        valid_im_gt_list = get_im_gt_name_dict([dataset_dis], flag="valid")
        self.dataloaders, self.datasets = create_calib_dataloaders(
            valid_im_gt_list,
            my_transforms=[
                Resize([1024, 1024])
            ],
            batch_size=1,
        )

    @abstractmethod
    def process(self, x: torch.Tensor, module_name: str = None):
        """Process tensor through attention mechanism."""
        pass

    @abstractmethod
    def stat_linear(self, X, Y, name):
        """Collect statistics for linear layers."""
        pass

    @abstractmethod
    def stat_attn(self, X, Y, name, n_heads):
        """Collect statistics for attention layers."""
        pass

    def set_params(self, yaml_config):
        """Take quantization configuration."""
        pass

    def _register_hooks(self, predictor, modules):
        """
        Register forward hooks for calibration. Override in subclasses for custom behavior.

        Args:
            predictor: SamPredictor instance
            modules: Module types to register hooks for

        Returns:
            Tuple of (linear_hooks, attn_hooks)
        """
        def stat_linear_hook(module, X, Y: torch.Tensor, name, linear_name):
            if isinstance(X, tuple):
                X = X[0]
            self.stat_linear(X, Y, name, linear_name)

        def stat_attn_hook(module, X, Y: torch.Tensor, name, n_heads):
            self.stat_attn(X, Y, name, n_heads)

        linear_hooks = []
        attn_hooks = []

        # Default: register hooks for decoder attention modules
        for name, component in predictor.model.named_modules():
            if isinstance(component, (modules)):
                for linear_name, m in component.named_modules():
                    if isinstance(m, nn.Linear):
                        linear_hooks.append(
                            m.register_forward_hook(partial(stat_linear_hook, name=name, linear_name=linear_name))
                        )
                attn_hooks.append(
                    component.register_forward_hook(partial(stat_attn_hook, name=name, n_heads=component.num_heads))
                )

        return linear_hooks, attn_hooks

    def _run_forward_pass(self, predictor, data_val):
        """
        Run forward pass through the model. Override in subclasses for custom behavior.

        Args:
            predictor: SamPredictor instance
            data_val: Data batch dictionary
        """
        imgs = data_val['image'].permute(0, 2, 3, 1).cpu().numpy()
        predictor.set_image(imgs.squeeze())
        labels_boxes = misc.masks_to_boxes(data_val['label'][:, 0, :, :]).cpu().numpy()
        masks, scores, logits = predictor.predict(
            box=labels_boxes,
            hq_token_only=False
        )

    def calibrate(self,  predictor: Union[SamPredictor, SamPredictor_, None], modules, num_samples=32):
        """
        Calibrate the processor using sample images.

        Args:
            predictor: SamPredictor instance
            modules: Module types to calibrate
            num_samples: Number of calibration samples
        """
        # Register hooks
        linear_hooks, attn_hooks = self._register_hooks(predictor, modules)

        logger = setup_logger('./calib_logs', self.strategy_name)
        logger.info(f'______Using: {self.strategy_name}_______')

        # Run calibration
        for k in range(len(self.dataloaders)):
            dataloader = self.accelerator.prepare(self.dataloaders[k])
            print('valid_dataloader len:', len(dataloader))
            progress_bar = tqdm(total=num_samples, desc=f"Calibrating")
            for i, data_val in enumerate(dataloader):
                if i == num_samples:
                    break
                self._run_forward_pass(predictor, data_val)
                progress_bar.update(1)

        # Remove hooks
        for h in linear_hooks:
            h.remove()
        for h in attn_hooks:
            h.remove()

        logger.info(f'Calibration complete. Collected stats for {len(self.stat)} modules.')

    def smooth_model(self, predictor: Union[SamPredictor, SamPredictor_, None], centerQ=False):
        """Apply smoothing to the model."""
        from RTN_quantization import utils as rtn_utils
        assert act_scales_file is not None, "Run Smooth_sam.py to generate act_scales_file"
        act_scales = torch.load(act_scales_file)
        if centerQ:
            predictor.model = rtn_utils.smooth_sam(predictor.model, act_scales, alpha=0.5, do_smooth_attn_encoder=False)
        else:
            predictor.model = rtn_utils.smooth_sam(predictor.model, act_scales, alpha=0.5)

    def quarot_model(self, predictor: Union[SamPredictor, SamPredictor_, None], rot_args, rtn_ro, decoder=False, centerQ=False):
        """Apply QuaRot rotation to the model."""
        from quarot import rotate_sam
        rotate_sam.rotate_sam(predictor.model, rot_args, rtn_ro, decoder=False, centerQ=centerQ)

    def clear_dict(self):
        """Clear statistics dictionary."""
        self.stat = {}
