import math
import torch
import torch.nn as nn
from functools import  partial
import numpy as np
import os
import logging
import time
from typing import Optional
from train.utils.dataloader import get_im_gt_name_dict, Resize
from abc import abstractmethod
from data_utils import OnlineDataset
from torchvision import transforms
from segment_anything.modeling.transformer import  Attention as  DecoderAttention
from train.segment_anything_training.modeling.image_encoder import Attention as EncoderAttentionTraining
from seginw.segment_anything.modeling.image_encoder import Attention as EncoderAttention
from torch.utils.data import DataLoader
from data_utils import OnlineDataset
from segment_anything import SamPredictor, sam_model_registry
from seginw.segment_anything import SamPredictor as SamPredictor_
from matplotlib import pyplot as plt
from functools import partial
from accelerate import Accelerator
import train.utils.misc as misc
from tqdm.auto import tqdm
from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
# from utils import show_mask_image
from collections import defaultdict
from quarot import rotate_sam, rotation_utils
from per_tensor_channel_group import quantize_activation_per_token_absmax, quantize_weight_per_channel_absmax, quantize_activation_low_high_density_activation_index
from utils import quantize_activation_per_highblock_abmax, find_O_qha

ENCODER_PROCESSOR_REGISTRY = {}
def register_encoder_processor(name: str):
    """Decorator to register encoder attention processors."""
    def decorator(cls):
        ENCODER_PROCESSOR_REGISTRY[name] = cls
        return cls
    return decorator

def get_encoder_processor(name: str, **kwargs):
    """Get encoder processor by name."""
    if name not in ENCODER_PROCESSOR_REGISTRY:
        available = list(ENCODER_PROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown encoder processor '{name}'. Available: {available}")
    return ENCODER_PROCESSOR_REGISTRY[name](**kwargs)

def create_calib_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1 ):
    gos_dataloaders = []
    gos_datasets = []
    for i in range(len(name_im_gt_list)):   
        gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
        dataloader = DataLoader(gos_dataset, batch_size, drop_last=False)
        gos_dataloaders.append(dataloader)
        gos_datasets.append(gos_dataset) 
    return gos_dataloaders, gos_datasets

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

@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    # print('got here', n_bits)
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

class ProcessStrategy():
    def __init__(self, strategy_name:str) -> None:
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
            my_transforms = [
                        Resize([1024, 1024])
                    ],
            batch_size=1,
        )

    @abstractmethod
    def process(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor):
        pass


    @abstractmethod
    def stat_tensor(self, X, Y, name):
        pass

    @abstractmethod
    def stat_attn(self, X, Y, name, n_heads):
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
        def stat_linear_hook(module, X, Y:torch.Tensor, name, linear_name):
            if isinstance(X, tuple):
                X = X[0]
            self.stat_linear(X, Y, name, linear_name)

        def stat_attn_hook(module, X, Y:torch.Tensor, name, n_heads ):
            self.stat_attn(X, Y,  name, n_heads)

        linear_hooks = []
        attn_hooks = []

        # Default: register hooks for decoder attention modules
        for name, component in predictor.model.named_modules():
            if isinstance(component, Attention) or isinstance(component, Attention_) or isinstance(component, Attention__):
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
        labels_boxes = misc.masks_to_boxes(data_val['label'][:,0,:,:]).cpu().numpy()
        masks, scores, logits = predictor.predict(
            box=labels_boxes,
            hq_token_only=False
        )

    def calibrate(self, predictor: SamPredictor | SamPredictor_ | None, modules, num_samples=32):
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

                # Run forward pass
                self._run_forward_pass(predictor, data_val)
                progress_bar.update(1)

        # Remove hooks
        for h in linear_hooks:
            h.remove()
        for h in attn_hooks:
            h.remove()

        logger.info(f'Calibration complete. Collected stats for {len(self.stat)} modules.')

    def clear_dict(self):
        self.stat = {}

class AttentionProcessor():
    def __init__(self, strategy_name:str) -> None:
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
            my_transforms = [
                        Resize([1024, 1024])
                    ],
            batch_size=1,
        )

    @abstractmethod
    def process(self, x:torch.Tensor, module_name:str=None):
        pass

    @abstractmethod
    def stat_linear(self, X, Y, name):
        pass

    @abstractmethod
    def stat_attn(self, X, Y, name, n_heads):
        pass

    def _take_Q(self,yaml_config):
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
        def stat_linear_hook(module, X, Y:torch.Tensor, name, linear_name):
            if isinstance(X, tuple):
                X = X[0]
            self.stat_linear(X, Y, name, linear_name)

        def stat_attn_hook(module, X, Y:torch.Tensor, name, n_heads ):
            self.stat_attn(X, Y,  name, n_heads)

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
        labels_boxes = misc.masks_to_boxes(data_val['label'][:,0,:,:]).cpu().numpy()
        masks, scores, logits = predictor.predict(
            box=labels_boxes,
            hq_token_only=False
        )

    def calibrate(self, predictor: SamPredictor | SamPredictor_ | None, modules, num_samples=32):
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
    def smooth_model(self, predictor: SamPredictor | SamPredictor_ | None,act_scales_file =None, centerQ= False):
        from RTN_quantization import utils as rtn_utils
        assert act_scales_file is not None, "Run Smooth_sam.py to generate act_scales_file"
        act_scales = torch.load(act_scales_file)
        if centerQ:
            predictor.model = rtn_utils.smooth_sam(predictor.model, act_scales, alpha=0.5, do_smooth_attn_encoder=False)
        else:
            predictor.model = rtn_utils.smooth_sam(predictor.model, act_scales, alpha=0.5)
    def quarot_model(self, predictor: SamPredictor | SamPredictor_ | None,rot_args, rtn_ro ,decoder = False , centerQ= False):
        
        rotate_sam.rotate_sam(predictor.model, rot_args, rtn_ro, decoder=False, centerQ=centerQ)
        

    def clear_dict(self):
        self.stat = {}





# ============================================================================
# Image Encoder Processor
# ============================================================================

@register_encoder_processor("BASE")
class EncoderAttentionProcessor(AttentionProcessor):
    """
    Processor for calibrating and processing image encoder attention layers.

    This processor is designed specifically for the ViT-based image encoder
    in SAM, which has a different architecture than the mask decoder.
    """

    def __init__(self, strategy_name:str='base'):
        super().__init__(strategy_name)
        self.stat = {}



    def stat_linear(self, X, Y: torch.Tensor, name, linear_name):
        """
        Collect statistics for linear layers (QKV projections).

        Args:
            X: Input tensor
            Y: Output tensor from linear layer
            name: Module name
            linear_name: Linear layer name (e.g., 'qkv', 'proj')
        """

        pass 

    def stat_attn(self, X, Y: torch.Tensor, name, n_heads):
        pass


    def _separate_heads_encoder(self, qkv: torch.Tensor, num_heads: int):
        """
        Separate QKV tensor into heads for encoder attention.

        Args:
            qkv: Combined QKV tensor of shape (B, H*W, 3, num_heads, C_per_head)
            num_heads: Number of attention heads

        Returns:
            q, k, v tensors each of shape (B, num_heads, H*W, C_per_head)
        """
        # qkv shape: (B, H*W, 3, num_heads, C_per_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, H*W, C_per_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v



    def process(self,  x:torch.Tensor, module, module_name:str=None):
        """
        Process Q, K, V tensors with quantization for encoder attention.

        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            name: Module name
            n_bits: Number of bits for quantization

        Returns:
            Processed Q, K, V tensors
        """
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x

class EncoderRecenterAttentionProcessor(AttentionProcessor):
    """
    Processor for calibrating and processing image encoder attention layers.

    This processor is designed specifically for the ViT-based image encoder
    in SAM, which has a different architecture than the mask decoder.
    """

    def __init__(self, strategy_name):
        super().__init__(strategy_name='recenterd')
        self.stat = {}

    def stat_linear(self, X, Y: torch.Tensor, name, linear_name):
        pass 

    def stat_attn(self, X, Y: torch.Tensor, name, n_heads):
        pass

    def _separate_heads_encoder(self, qkv: torch.Tensor, num_heads: int):
        """
        Separate QKV tensor into heads for encoder attention.

        Args:
            qkv: Combined QKV tensor of shape (B, H*W, 3, num_heads, C_per_head)
            num_heads: Number of attention heads

        Returns:
            q, k, v tensors each of shape (B, num_heads, H*W, C_per_head)
        """
        # qkv shape: (B, H*W, 3, num_heads, C_per_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, H*W, C_per_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v



    def process(self,  x:torch.Tensor, module, module_name:str=None):
        """
        Process Q, K, V tensors with quantization for encoder attention.

        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            name: Module name
            n_bits: Number of bits for quantization

        Returns:
            Processed Q, K, V tensors
        """
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        x_mean = x.mean(1, keepdim=True).mean(2, keepdim=True)
        x_hat = x - x_mean
        qkv_hat = module.qkv(x_hat).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv_mean= module.qkv(x_mean).reshape(B, 1, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q_hat, k_hat, v_hat = qkv_hat.reshape(3, B * module.num_heads, H * W, -1).unbind(0)
        q_mean, k_mean, v_mean= qkv_mean.reshape(3, B * module.num_heads, 1, -1).unbind(0)

        attn = (q_hat * module.scale) @ k_hat.transpose(-2, -1)
        # attn_mean = (q_mean * self.scale) @ k_hat.transpose(-2, -1)
        # attn = attn + attn_mean 

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q_hat + q_mean, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = ((attn @ v_hat) + v_mean).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x

@register_encoder_processor("FAKE_PRUNE")
class FakePruneProcessor(AttentionProcessor):
    def __init__(self, strategy_name:str='fake_prune'):
        super().__init__(strategy_name)
        self.stat = {}
        # Store all entropy values per position for each head
        self.entropy_stats = defaultdict(lambda: {"entropy_per_position": []})
        self.threshold = 5.0
        self.percent = None
        self.prunehighentropy = False
    def _take_Q(self,args):
        self.threshold = 5.0
        self.percent =args.quantization.percent_entropy
        self.prunehighentropy= args.quantization.high_entropy
    def calculate_entropy(self, attn_head):
        """Calculate entropy for each position in a single attention head"""
        # Convert to tensor if needed
        if isinstance(attn_head, np.ndarray):
            attn_head = torch.from_numpy(attn_head)
        
        # Ensure attention is normalized (should already be after softmax)
        eps = 1e-12
        attn_normalized = torch.clamp(attn_head, min=eps)
        
        # Calculate entropy for each query position (each row)
        entropy_per_position = -torch.sum(attn_normalized * torch.log(attn_normalized), dim=-1)
        
        return entropy_per_position
    
    def _register_hooks(self, predictor, modules):
        """Register hooks to capture attention patterns during forward pass"""
        from segment_anything.modeling.image_encoder import Attention
        
        def attention_hook(module, input, output, name):
            """Hook to capture attention patterns after softmax"""
            # Get input tensor
            x = input[0] if isinstance(input, tuple) else input
            B, H, W, _ = x.shape
            
            # Recompute attention to get the attention matrix
            qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)
            
            attn = (q * module.scale) @ k.transpose(-2, -1)
            
            if module.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))
            
            attn = attn.softmax(dim=-1)
            B_nhead, N, _ = attn.shape  # B * num_heads, N, N

            for head_idx in range(B_nhead):
                
                attn_head = attn[head_idx]  # Shape: (N, N)
                
                # Calculate entropy for each position
                entropy_per_position = self.calculate_entropy(attn_head)
                
                # Accumulate entropy values for this head
                head_key = f"{name}.head_{head_idx}"
                self.entropy_stats[head_key]["entropy_per_position"].extend(entropy_per_position.tolist())

        hooks = []
        
        # Register hooks for all encoder attention modules
        for name, component in predictor.model.image_encoder.named_modules():
            if isinstance(component, modules):
                print(f"Registering attention hook for {name}")
                hooks.append(
                    component.register_forward_hook(
                        partial(attention_hook, name=name)
                    )
                )
        return hooks, []  # Return (attention_hooks, linear_hooks)
    
    def _run_forward_pass(self, predictor, data_val):
        """Run forward pass through image encoder only"""
        imgs = data_val['image'].permute(0, 2, 3, 1).cpu().numpy()
        predictor.set_image(imgs.squeeze())
    
    def calibrate(self, predictor, modules, num_samples=32):
        """
        Custom calibration that accumulates all entropy values, then calculates final statistics
        """
        # Reset entropy statistics - store all entropy values
        self.entropy_stats = defaultdict(lambda: {"entropy_per_position": []})
        
        # Register hooks (only attention hooks needed)
        attention_hooks, _ = self._register_hooks(predictor, modules)
        
        logger = setup_logger('./calib_logs', self.strategy_name)
        print(f'______Using: {self.strategy_name}_______')
        print('Collecting entropy values for all attention heads during calibration')
        # Run calibration - accumulate all entropy values
        total_processed = 0
        for k in range(len(self.dataloaders)):
            dataloader = self.accelerator.prepare(self.dataloaders[k])
            print(f'Dataloader {k} length:', len(dataloader))
            progress_bar = tqdm(total=min(num_samples, len(dataloader)), 
                              desc=f"Collecting entropy data")
            
            for i, data_val in enumerate(dataloader):
                if total_processed >= num_samples:
                    break
                
                # Run forward pass to trigger hooks and accumulate entropy values
                self._run_forward_pass(predictor, data_val)
                
                total_processed += 1
                progress_bar.update(1)
                
                # Log progress for first few samples
                if total_processed <= 3:
                    sample_head_key = list(self.entropy_stats.keys())[0] if self.entropy_stats else None
                    if sample_head_key:
                        current_length = len(self.entropy_stats[sample_head_key]["entropy_per_position"])
                        print(f'After sample {total_processed}: {sample_head_key} has {current_length} entropy values')
            
            if total_processed >= num_samples:
                break
        
        # Remove hooks
        for hook in attention_hooks:
            hook.remove()
        
        # Calculate final statistics from all accumulated entropy values
        print('Calculating final entropy variance and mean from accumulated data...')
        self.final_entropy_stats = {}
        if self.percent is  None:
            for head_key, stats in self.entropy_stats.items():
                entropy_values = stats["entropy_per_position"]
                if len(entropy_values) > 0:
                    # Convert to tensor for calculation
                    entropy_tensor = torch.tensor(entropy_values)
                    
                    # Calculate mean
                    entropy_mean = torch.mean(entropy_tensor)
                    
                    # Parse layer and head index from key format: 'blocks.22.attn.head_233'
                    parts = head_key.split('.')
                    layer_name = f"image_encoder.{parts[0]}.{parts[1]}.{parts[2]}"
                    head_idx = int(parts[3].split('_')[1])  # Extract head number
                    
                    # Only store heads whose entropy mean is greater than threshold
                    if entropy_mean.item() > self.threshold:
                        if layer_name not in self.final_entropy_stats:
                            self.final_entropy_stats[layer_name] = []
                        self.final_entropy_stats[layer_name].append(head_idx)
        else :
            # Group entropy stats by layer
            layer_heads = {}
            for head_key, stats in self.entropy_stats.items():
                entropy_values = stats["entropy_per_position"]
                if len(entropy_values) > 0:
                    # Convert to tensor and calculate mean
                    entropy_tensor = torch.tensor(entropy_values)
                    entropy_mean = torch.mean(entropy_tensor)
                    
                    # Parse layer and head index from key format: 'blocks.22.attn.head_233'
                    parts = head_key.split('.')
                    layer_name = f"image_encoder.{parts[0]}.{parts[1]}.{parts[2]}"
                    head_idx = int(parts[3].split('_')[1])
                    
                    if layer_name not in layer_heads:
                        layer_heads[layer_name] = []
                    layer_heads[layer_name].append((head_idx, entropy_mean.item()))
         
            # For each layer, select top percent of heads with highest entropy
            for layer_name, heads_with_entropy in layer_heads.items():
                if len(heads_with_entropy) > 0:
                    # Sort heads by entropy mean in descending order
                    if self.prunehighentropy:
                        heads_with_entropy.sort(key=lambda x: x[1], reverse=True)
                    else :
                        heads_with_entropy.sort(key=lambda x: x[1])
                    # Calculate number of heads to select based on percentage
                    num_heads_to_select = max(1, int(len(heads_with_entropy) * self.percent ))

                    # Select top percent of heads
                    selected_heads = heads_with_entropy[:num_heads_to_select]
                    
                    # Store only the head indices
                    self.final_entropy_stats[layer_name] = [head_idx for head_idx, _ in selected_heads]
            
        # Log sample statistics
        for layer_name, head_list in list(self.final_entropy_stats.items()):
            print(f'Layer {layer_name}: {len(head_list)} heads with high entropy: {head_list[:10]}{"..." if len(head_list) > 10 else ""}')
   
    
    def get_entropy_stats(self):
        """Return the collected entropy statistics"""
        return getattr(self, 'final_entropy_stats', {})
    
    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Standard attention processing - no modifications for fake pruning"""
        if not any(num in module_name for num in ["5", "11", "17", "23"]): # modules whose shape is (16, 4096, 4096)
            list_prune_head= module.processor.final_entropy_stats.get(module_name, [])
            # list_prune_head= []
        else:
            list_prune_head= []
            # list_prune_head= module.processor.final_entropy_stats.get(module_name, [])
        
        B, H, W, _ = x.shape
        # Standard attention computation
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        B_nhead, N, _ = attn.shape  # B * num_heads, N, N

        if list_prune_head:
            attn[list_prune_head, :, :] = 1.0 / N
    
        x = (attn @ v).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x
@register_encoder_processor("FAKE_PRUNE_ALL_HEADS_SAME_TOKEN")
class FakePruneAllHeadsSameTokenProcessor(FakePruneProcessor):
    def __init__(self, strategy_name:str='fake_prune_all_heads_same_token'):
        super().__init__(strategy_name)
    def _take_Q(self, args):
        self.threshold = 5.0
        self.percent = args.quantization.percent_entropy
        self.prunehighentropy = args.quantization.high_entropy
    def calibrate(self, predictor, modules, num_samples=32):
        """
        Calibrate by collecting entropy scores for all tokens across all heads,
        then select tokens to prune based on statistics across heads per layer
        """
        # Reset token entropy statistics
        self.entropy_stats = defaultdict(lambda: {"entropy_per_position": []})
        
        # Register hooks (only attention hooks needed)
        attention_hooks, _ = self._register_hooks(predictor, modules)
        
        logger = setup_logger('./calib_logs', self.strategy_name)
        print(f'______Using: {self.strategy_name}_______')
        print('Collecting entropy values for all attention heads during calibration')
        # Run calibration - accumulate all entropy values
        total_processed = 0
        for k in range(len(self.dataloaders)):
            dataloader = self.accelerator.prepare(self.dataloaders[k])
            print(f'Dataloader {k} length:', len(dataloader))
            progress_bar = tqdm(total=min(num_samples, len(dataloader)), 
                              desc=f"Collecting entropy data")
            
            for i, data_val in enumerate(dataloader):
                if total_processed >= num_samples:
                    break
                
                # Run forward pass to trigger hooks and accumulate entropy values
                self._run_forward_pass(predictor, data_val)
                
                total_processed += 1
                progress_bar.update(1)
                
                # Log progress for first few samples
                if total_processed <= 3:
                    sample_head_key = list(self.entropy_stats.keys())[0] if self.entropy_stats else None
                    if sample_head_key:
                        current_length = len(self.entropy_stats[sample_head_key]["entropy_per_position"])
                        print(f'After sample {total_processed}: {sample_head_key} has {current_length} entropy values')
                   
            if total_processed >= num_samples:
                break
        
        # Remove hooks
        for hook in attention_hooks:
            hook.remove()
        
        # Calculate token statistics and select tokens to prune
        print('Calculating token-wise statistics across heads for each layer...')
        self.final_token_prune_stats = {}
  
        for layer_name, token_entropy_dict in self.entropy_stats.items():
            if not token_entropy_dict:
                continue
                
            # Calculate mean entropy for each token position across all heads and samples
            token_mean_entropies = []

            entropy_value = torch.tensor(token_entropy_dict["entropy_per_position"]).view(total_processed, -1)
            
            entropy_value = entropy_value.mean(dim=0)
            layer_number = layer_name.split('.')[1]
            if not layer_number in self.final_token_prune_stats:
                self.final_token_prune_stats[layer_number] = [entropy_value]
            else:
                self.final_token_prune_stats[layer_number].append(entropy_value)
    
        # calculate average over heads of each layer
    
        self.final_token_prune_stats = {layer_name: torch.mean(torch.stack(token_indices), dim=0) for layer_name, token_indices in self.final_token_prune_stats.items()}
      
        if self.percent is None:
            # use threshold - select indices where values are greater than threshold
            for layer_name, token_entropies in self.final_token_prune_stats.items():
                self.final_token_prune_stats[layer_name] = torch.where(token_entropies > self.threshold)[0].tolist()
        else:
            # sort and select top percent - return indexes
            for layer_name, token_entropies in self.final_token_prune_stats.items():
                import ipdb; ipdb.set_trace()
                self.final_token_prune_stats[layer_name] = torch.topk(token_entropies, int(len(token_entropies) * self.percent / 100), largest=False)[1].tolist()
        import ipdb; ipdb.set_trace()
    def get_entropy_stats(self):
        """Return the collected token pruning statistics"""
        return getattr(self, 'final_token_prune_stats', {})
    
    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Process attention with same token pruning across all heads"""
        # Get tokens to prune for this module
        tokens_to_prune = self.final_token_prune_stats.get(module_name, [])
        
        B, H, W, _ = x.shape
        
        # Standard attention computation
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        B_nhead, N, _ = attn.shape  # B * num_heads, N, N

        # Apply token pruning: set attention scores to uniform for selected tokens
        if tokens_to_prune:
            for token_idx in tokens_to_prune:
                if token_idx < N:  # Ensure token index is valid
                    # Set attention scores for this token position across all heads to uniform
                    attn[:, token_idx, :] = 1.0 / N
    
        x = (attn @ v).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x    
@register_encoder_processor("FAKE_PRUNE_WQ_WK")
class FakePruneWeightWqQkProcessor(FakePruneProcessor): 
    def __init__(self, strategy_name:str='prune Wq Wk based on entropy attention scores'):
        super().__init__(strategy_name)
        self.headthreshold =20
    def _take_Q(self, args):
        self.threshold = 5.0
        self.percent = args.quantization.percent_entropy
        self.prunehighentropy = args.quantization.high_entropy
        self.headthreshold = getattr(args.quantization, 'head_threshold', 18)
    def calibrate(self, predictor, modules, num_samples=32):
        """
        Custom calibration that finds weight heads to prune based on entropy
        """
        # Reset entropy statistics
        self.entropy_stats = defaultdict(lambda: {"entropy_per_position": []})
        
        # Get num_weight_heads from predictor
        num_weight_heads = predictor.model.image_encoder.blocks[0].attn.num_heads  # Assuming 16
        
        # Register hooks
        attention_hooks, _ = self._register_hooks(predictor, modules)
        
        logger = setup_logger('./calib_logs', self.strategy_name)
        print(f'______Using: {self.strategy_name}_______')
        print('Collecting entropy values for weight head pruning during calibration')
        
        # Run calibration
        total_processed = 0
        for k in range(len(self.dataloaders)):
            dataloader = self.accelerator.prepare(self.dataloaders[k])
            print(f'Dataloader {k} length:', len(dataloader))
            progress_bar = tqdm(total=min(num_samples, len(dataloader)), 
                              desc=f"Collecting entropy data for weight heads")
            
            for i, data_val in enumerate(dataloader):
                if total_processed >= num_samples:
                    break
                
                self._run_forward_pass(predictor, data_val)
                total_processed += 1
                progress_bar.update(1)
            
            if total_processed >= num_samples:
                break
        
        # Remove hooks
        for hook in attention_hooks:
            hook.remove()
        
        # Convert attention head indices to weight head indices and count frequency
        print('Converting attention heads to weight heads and counting frequency...')
        layer_weight_head_counts = defaultdict(lambda: defaultdict(int))
        
        # First, get high-entropy attention heads like FakePruneProcessor
        temp_entropy_stats = []
    
        # Use percentage-based selection
        self.final_weight_head_stats = {}
        layer_heads = {}
        for head_key, stats in self.entropy_stats.items():
            entropy_values = stats["entropy_per_position"]
            if len(entropy_values) > 0:
                entropy_tensor = torch.tensor(entropy_values)
                
                parts = head_key.split('.')
                layer_name = f"image_encoder.{parts[0]}.{parts[1]}.{parts[2]}"
                head_idx = int(parts[3].split('_')[1]) % num_weight_heads
                
                if layer_name not in layer_heads:
                    layer_heads[layer_name] = {}
                if head_idx not in layer_heads[layer_name]:
                    layer_heads[layer_name][head_idx] = []
                layer_heads[layer_name][head_idx].extend(entropy_tensor.tolist())
        
        # Calculate number of heads to select
        num_heads_to_select = int(self.percent * num_weight_heads)
        
        for layer_name, heads in layer_heads.items():
            if  any(num in layer_name for num in ["5", "11", "17", "23"]):
                continue
            head_stats = []
            
            for head_idx, entropy_values in heads.items():
                if len(entropy_values) > 0:
                    entropy_tensor = torch.tensor(entropy_values)
                    entropy_mean = torch.mean(entropy_tensor)
                    entropy_variance = torch.var(entropy_tensor)
                    head_stats.append((layer_name,head_idx, entropy_mean.item(), entropy_variance.item()))
            temp_entropy_stats.extend(head_stats)
            # Sort by entropy mean (highest first) and select top heads
            head_stats.sort(key=lambda x: x[2], reverse=True)
            selected_head_indices = [head_idx for _,head_idx, _, _ in head_stats[:num_heads_to_select]]
            
            self.final_weight_head_stats[layer_name] = selected_head_indices
        # temp_entropy_stats.sort(key=lambda x: x[2], reverse=True)

        # # Select top 70% heads globally
        # total_heads = len(temp_entropy_stats)
        # num_heads_to_select = int(self.percent * total_heads)
        
        # top_heads = temp_entropy_stats[:num_heads_to_select]

        # # Organize selected heads by layer
        # for layer_name in layer_heads.keys():
        #     if any(num in layer_name for num in ["5", "11", "17", "23"]):
        #         continue
        #     self.final_weight_head_stats[layer_name] = []

        # for layer_name, head_idx, _, _ in top_heads:
        #     self.final_weight_head_stats[layer_name].append(head_idx)
            
        # import ipdb; ipdb.set_trace()
        
        print('Zeroing out weights for selected weight heads...')
        for name, module in predictor.model.image_encoder.named_modules():
            if hasattr(module, 'qkv') and hasattr(module, 'num_heads'):
                module_name = f"image_encoder.{name}"
                if module_name in self.final_weight_head_stats:
                    list_prune_weight_heads = self.final_weight_head_stats[module_name]
                    
                    num_weight_heads = module.num_heads
                    head_dim = module.qkv.weight.data.shape[0] // (3 * num_weight_heads)
                    
                    with torch.no_grad():
                        for weight_head_idx in list_prune_weight_heads:
                            # Calculate weight indices for this weight head
                            start_q = weight_head_idx * head_dim
                            end_q = (weight_head_idx + 1) * head_dim
                            start_k = num_weight_heads * head_dim + weight_head_idx * head_dim
                            end_k = num_weight_heads * head_dim + (weight_head_idx + 1) * head_dim
                            
                            # Zero out Wq weights for this weight head
                            module.qkv.weight.data[start_q:end_q, :] = 0.0
                            # Zero out Wk weights for this weight head  
                            module.qkv.weight.data[start_k:end_k, :] = 0.0
                            
                            if hasattr(module.qkv, 'bias') and module.qkv.bias is not None:
                                module.qkv.bias.data[start_q:end_q] = 0.0  # Wq bias
                                module.qkv.bias.data[start_k:end_k] = 0.0  # Wk bias
                    
                    print(f'Zeroed weights for {len(list_prune_weight_heads)} weight heads in {module_name}')
        
        # Log results
    
        for layer_name, weight_head_list in self.final_weight_head_stats.items():
            print(f'Layer {layer_name}: {len(weight_head_list)} weight heads to prune: {weight_head_list}')
    
    def get_entropy_stats(self):
        """Return the collected weight head statistics"""
        return getattr(self, 'final_weight_head_stats', {})
    
    def process(self, x: torch.Tensor, module, module_name: str = None):
        """Process attention with weight head pruning - weights already zeroed in calibration"""
        B, H, W, _ = x.shape
        
        # Standard attention computation
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)
 
        attn = (q * module.scale) @ k.transpose(-2, -1)
        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        B_nhead, N, _ = attn.shape  # B * num_heads, N, N

        

        x = (attn @ v).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x


        

@register_encoder_processor("SMOOTH_MEAN_Q")
class EncoderAttentionProcessorSmoothMeanQ(AttentionProcessor):
    """
    Processor for calibrating and processing image encoder attention layers.

    This processor is designed specifically for the ViT-based image encoder
    in SAM, which has a different architecture than the mask decoder.
    """

    def __init__(self, strategy_name:str='base'):
        super().__init__(strategy_name)
        self.stat = {}

    def stat_tensor(self,name, tensor):
        """Calculate per-channel maximum absolute values"""
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return
        if tensor.dim() < 2:
            return
        
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        current_max = torch.max(tensor, dim=0)[0].float().cpu()
        
        if name in self.stat:
            self.stat[name]["act_scales"] = torch.max(self.stat[name]["act_scales"], current_max)
        else:
            self.stat[name] = defaultdict()
            self.stat[name]['act_scales'] = current_max

    def stat_linear(self, X, Y: torch.Tensor, name, linear_name):
        """
        Collect statistics for linear layers (QKV projections).

        Args:
            X: Input tensor
            Y: Output tensor from linear layer
            name: Module name
            linear_name: Linear layer name (e.g., 'qkv', 'proj')
        """

        self.stat_tensor(name, X)    
        self.stat[name][linear_name] = Y
        self.stat[name]["input"+linear_name] = X

    def stat_attn(self, X, Y: torch.Tensor, name, n_heads):
        pass

    def _separate_heads_encoder(self, qkv: torch.Tensor, num_heads: int):
        """
        Separate QKV tensor into heads for encoder attention.

        Args:
            qkv: Combined QKV tensor of shape (B, H*W, 3, num_heads, C_per_head)
            num_heads: Number of attention heads

        Returns:
            q, k, v tensors each of shape (B, num_heads, H*W, C_per_head)
        """
        # qkv shape: (B, H*W, 3, num_heads, C_per_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, H*W, C_per_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v
    
    def _register_hooks(self, predictor, modules):
        """
        Register hooks specifically for image encoder attention modules.

        Args:
            predictor: SamPredictor instance
            modules: Module types to register hooks for

        Returns:
            Tuple of (linear_hooks, attn_hooks)
        """
        from segment_anything.modeling.image_encoder import Attention

        def stat_linear_hook(module, X, Y: torch.Tensor, name, linear_name):
            if isinstance(X, tuple):
                X = X[0]
            self.stat_linear(X, Y, name, linear_name)

        def stat_attn_hook(module, X, Y: torch.Tensor, name, n_heads):
            self.stat_attn(X, Y, name, n_heads)

        linear_hooks = []
        attn_hooks = []

        # Register hooks for image encoder blocks
        for name, component in predictor.model.image_encoder.named_modules():
        # for name, component in predictor.image_encoder.named_modules():
            if isinstance(component, (modules)) :
                # Hook the QKV linear layer
                for linear_name, m in component.named_modules():
                    if isinstance(m, nn.Linear) and linear_name == 'qkv':
                        print(f"Registering hook for {name}.{linear_name}")
                        linear_hooks.append(
                            m.register_forward_hook(
                                partial(stat_linear_hook, name=name, linear_name=linear_name)
                            )
                        )
                # Hook the attention module
                attn_hooks.append(
                    component.register_forward_hook(
                        partial(stat_attn_hook, name=name, n_heads=component.num_heads)
                    )
                )

        return linear_hooks, attn_hooks
    def _run_forward_pass(self, predictor, data_val):
        """
        Run forward pass through the image encoder only (not full prediction).

        Args:
            predictor: SamPredictor instance
            data_val: Data batch dictionary
        """
        imgs = data_val['image'].permute(0, 2, 3, 1).cpu().numpy()
        # This will trigger the image encoder forward pass
        predictor.set_image(imgs.squeeze())
    def process(self,  x:torch.Tensor, module, module_name:str=None):
        """
        Process Q, K, V tensors with quantization for encoder attention.

        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            name: Module name
            n_bits: Number of bits for quantization

        Returns:
            Processed Q, K, V tensors
        """
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x
@register_encoder_processor("COMPENSATE")
class EncoderAttentionProcessorCompensate(EncoderAttentionProcessorSmoothMeanQ):
    def __init__(self, strategy_name:str='base'):
        super().__init__(strategy_name)
    def _take_Q(self,  args = None):
        pass
        # self.n_bits_act= args.quantization.n_bits

    def process(self,  x:torch.Tensor, module, module_name:str=None):
        """
        Process Q, K, V tensors with quantization for encoder attention.

        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            name: Module name
            n_bits: Number of bits for quantization

        Returns:
            Processed Q, K, V tensors
        """
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))
      
        
        
        attn = attn.softmax(dim=-1)
        attn_qha , indicies = quantize_activation_per_highblock_abmax(attn, n_bits=4, percent=0.5, block_size= 1 )
        O_qha = find_O_qha(qattn=attn_qha, v=v, indices = indicies, n_bits=4, block_size=1)
        
        x = (O_qha).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x
@register_encoder_processor("SMOOTH")
class EncoderAttentionProcessorSmooth(EncoderAttentionProcessorSmoothMeanQ):
    def __init__(self, strategy_name:str='base'):
        super().__init__(strategy_name)
    def _take_Q(self,  args = None):
        self.qkT_v= args.quantization.qkT_v
        self.n_bits_act= args.quantization.n_bits
    def process(self,  x:torch.Tensor, module, module_name:str=None):
        """
        Process Q, K, V tensors with quantization for encoder attention.

        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            name: Module name
            n_bits: Number of bits for quantization

        Returns:
            Processed Q, K, V tensors
        """
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))
        if self.qkT_v:
            attn = quantize_activation_per_token_absmax(attn, n_bits=self.n_bits_act)
            v = quantize_weight_per_channel_absmax(v.permute(0, 2, 1), n_bits=self.n_bits_act).permute(0, 2, 1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x
@register_encoder_processor("HIGH_LOW_ATTN_V")
class EncoderAttentionProcessorHighLow(EncoderAttentionProcessorSmoothMeanQ):
    def __init__(self, strategy_name:str='base'):
        super().__init__(strategy_name)
    def _take_Q(self,  args = None):
        self.n_bits_act= args.quantization.n_bits
        self.percent =args.quantization.percent
        
    def process(self,  x:torch.Tensor, module, module_name:str=None):
        """
        Process Q, K, V tensors with quantization for encoder attention.

        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            name: Module name
            n_bits: Number of bits for quantization

        Returns:
            Processed Q, K, V tensors
        """
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))
        
        attn, indices = quantize_activation_low_high_density_activation_index(attn, n_bits=self.n_bits_act,percent=self.percent, quantizehigh=True, )       
        v=quantize_activation_low_high_density_activation_index(v, n_bits=self.n_bits_act, percent=self.percent, quantizehigh=True, indices=indices)[0]
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x

class EncoderAttentionProcessorSmoothLogQ(EncoderAttentionProcessorSmoothMeanQ):
    def __init__(self, strategy_name:str='base'):
        super().__init__(strategy_name)
    def _take_Q(self,  args = None):
        self.qkT_v= args.quantization.qkT_v
        self.n_bits_act= args.quantization.n_bits
    def process(self,  x:torch.Tensor, module, module_name:str=None):
        """
        Process Q, K, V tensors with quantization for encoder attention.

        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            name: Module name
            n_bits: Number of bits for quantization

        Returns:
            Processed Q, K, V tensors
        """
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))
        if self.qkT_v:
            from per_tensor_channel_group import quantize_activation_log_per_token_absmax, quantize_weight_log_per_channel
            attn = quantize_activation_log_per_token_absmax(attn, n_bits=self.n_bits_act)
            v = quantize_weight_log_per_channel(v.permute(0, 2, 1), n_bits=self.n_bits_act).permute(0, 2, 1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)

        return x

@register_encoder_processor("QUAROT")
class EncoderAttentionProcessorQuarot(EncoderAttentionProcessorSmoothMeanQ):
    def __init__(self, strategy_name:str='base'):
        super().__init__(strategy_name)

    def _take_Q(self,  args = None):
        self.Q = rotation_utils.get_orthogonal_matrix(args.quarot_inf.hidden_size_image_en,
                                                      args.quarot_inf.rotate_mode,
                                                      device = args.quarot_inf.device,
                                                      seed=args.quarot_inf.seed)
        self.qkT_v= args.quantization.qkT_v
        self.n_bits_act= args.rtn_ro_config.n_bits
    def process(self,  x:torch.Tensor, module, module_name:str=None):
        
        # Apply Q matrix multiplication if provided
        if self.Q is not None:
            B, H, W, C = x.shape
            self.Q = self.Q.to( dtype=x.dtype)
            
            x_flat = x.reshape(B, H*W, C)
            x_rotated = torch.matmul(x_flat, self.Q)
            x = x_rotated.reshape(B, H, W, C)
        
        B, H, W, _ = x.shape
        
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # ipdb.set_trace()
        q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))
        if self.qkT_v:
            from per_tensor_channel_group import quantize_activation_per_token_absmax, quantize_weight_per_channel_absmax
            attn = quantize_activation_per_token_absmax(attn, n_bits=self.n_bits_act)
            v = quantize_weight_per_channel_absmax(v.permute(0, 2, 1), n_bits=self.n_bits_act).permute(0, 2, 1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, module.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = module.proj(x)
        return x

class EncoderAttentionProcessQuarotCenterQ(EncoderAttentionProcessorQuarot):
    
    attention_score = defaultdict(list)
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent Attention class."""
        super().__init__(*args, **kwargs)
        # Quantization attributes
   

    def process(self, x: torch.Tensor) :
  
        B, H, W, C = x.shape
        if hasattr(self, 'Q') and self.Q is not None:
            self.Q = self.Q.to( dtype=x.dtype)
            
            x_flat = x.reshape(B, H*W, C)
            x_rotated = torch.matmul(x_flat, self.Q)
            x = x_rotated.reshape(B, H, W, C)
        # qkv with shape (3, B, nHead, H * W, C)
        x = x.reshape(B, H*W, C)
        bias = self.qkv.bias[None, None, ...]
        x_mean = x.mean(1, keepdim=True)
        x_hat= (x - x_mean)
        # qkv     = self.qkv(x)       # shape: (B, H*W, 3*num_heads*dim)        
        qkv_hat = self.qkv_w_hat(x_hat, self.q_scales)
        qkv_mean = self.qkv(x_mean) - bias


        qkv_hat = qkv_hat.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv_mean = qkv_mean.reshape(B, 1, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # import ipdb; ipdb.set_trace()
        q_hat, k_hat, v_hat = qkv_hat.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        q_mean, _, v_mean = qkv_mean.reshape(3, B * self.num_heads, 1, -1).unbind(0)
        if ImageEncoderViTObserver.debug:
            qkv = qkv.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
            # Assert that q_ori = q + q_mean (and similarly for k, v)
            assert torch.allclose(q, q_hat+q_mean, rtol=1e-4, atol=1e-4), "q_ori != q + q_mean"
            # assert torch.allclose(k, k_hat+k_mean, rtol=1e-4, atol=1e-4), "k_ori != k + k_mean"
            assert torch.allclose(v, v_hat+v_mean, rtol=1e-4, atol=1e-4), "v_ori != v + v_mean"
            attn_ori = (q * self.scale) @ k.transpose(-2, -1)
        # q_hat = q_hat+q_mean
        # k_hat = k_hat+k_mean
        

        # Compute attention
      
        attn = (q_hat * self.scale) @ k_hat.transpose(-2, -1)
        attn_mean = (q_mean * self.scale) @ k_hat.transpose(-2, -1)
        attn = attn + attn_mean

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q_hat+q_mean, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )
            if ImageEncoderViTObserver.debug:
                attn_ori = add_decomposed_rel_pos(
                    attn_ori, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
                )

        attn = attn.softmax(dim=-1)
        
        if self.qkT_v:
            attn, indices = quantize_activation_low_high_density_activation_index(attn, n_bits=self.n_bits_act,percent=self.percent, quantizehigh=True, )
            v=quantize_activation_low_high_density_activation_index(v_hat+v_mean, n_bits=self.n_bits_act, percent=self.percent, quantizehigh=True, indices=indices)[0]
        else:
            v= v_hat+v_mean
        if ImageEncoderViTObserver.debug:
            attn_ori = attn_ori.softmax(dim=-1)
        output = (
            (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        )
        output = self.proj(output)

        return output, attn, attn_mean, q_hat, k_hat, v_hat, q_mean, v_mean 

    @staticmethod
    def clear_dict():
        """Clear the attention score dictionary."""
        AttentionObserver.attention_score = defaultdict(list)