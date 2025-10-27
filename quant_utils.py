import math
import torch
import torch.nn as nn
from functools import  partial
import torch.nn.functional as F
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
    """
    Quantize activations per token (per row) using absmax scaling.

    Args:
        t: Tensor to quantize (any shape)
        n_bits: Number of bits for quantization

    Returns:
        Quantized tensor with same shape as input
    """
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

@torch.no_grad()
def quantize_activation_per_channel_absmax(t, n_bits=8):
    """
    Quantize activations per channel (along dim 1) using absmax scaling.

    Args:
        t: Tensor to quantize, shape (..., N, C)
        n_bits: Number of bits for quantization

    Returns:
        Quantized tensor with same shape as input
    """
    scales = t.abs().max(dim=1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
    """
    Quantize entire tensor using single absmax scale.

    Args:
        t: Tensor to quantize
        n_bits: Number of bits for quantization

    Returns:
        Quantized tensor with same shape as input
    """
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t

    

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

    def clear_dict(self):
        self.stat = {}




class DecoderSignProcessor(AttentionProcessor):
    """
    Decoder attention processor that uses sign statistics for quantization.

    Hook Expectations:
    ------------------
    stat_linear(X, Y, name, linear_name):
        X: Input to linear layer
           - Shape: (B, N, C) where B=batch, N=num_tokens, C=channels
           - May be a tuple (X,) - will be unpacked automatically
        Y: Output from linear layer (after nn.Linear forward)
           - For q_proj: (B, N, C) - query projections
           - For k_proj: (B, N, C) - key projections
           - For v_proj: (B, N, C) - value projections
        name: Attention module name (e.g., "model.mask_decoder.transformer.layers.0.self_attn")
        linear_name: Specific linear layer name (e.g., "q_proj", "k_proj", "v_proj")

    stat_attn(X, Y, name, n_heads):
        NOT USED in this processor (implicitly defined but empty)
    """

    def __init__(self, strategy_name):
        super().__init__(strategy_name)
        self.stat = {}

    def stat_linear(self, X, Y:torch.Tensor, name, linear_name):
        # attn_name = name 
        sign = torch.sign(torch.sign(Y).mean(-2, keepdim=True))
        if 'k' in linear_name:
            if name not in self.stat:
                self.stat[name] =  defaultdict() 
                self.stat[name][linear_name] = sign
            else:
                self.stat[name][linear_name] += sign


        
    def process(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, name, n_bits):
        sign = self.stat[name]['k_proj'].sign().reshape(-1, 16)[None, None, ...].permute(0,2,1,3)

        Q.mul_(sign)
        K.mul_(sign)
        Q =  quantize_activation_per_token_absmax(Q.permute(0,2,1,3), n_bits=4)
        K =  quantize_activation_per_token_absmax(K.permute(0,2,1,3), n_bits=4)
        return Q, K, V






class DecoderDoNothingProcessor(AttentionProcessor):
    """
    Baseline decoder processor that performs quantization without calibration.

    Hook Expectations:
    ------------------
    stat_linear(X, Y, name, linear_name):
        X: Input to linear layer
           - Shape: (B, N, C) where B=batch, N=num_tokens, C=channels
        Y: Output from linear layer
           - Shape: (B, N, C)
        NOT USED - no statistics collected

    stat_attn(X, Y, name, n_heads):
        X: Input to attention module
           - Shape: (B, N, C) for decoder attention
        Y: Output from attention module
           - Shape: (B, N, C)
        NOT USED - no statistics collected
    """

    def __init__(self, strategy_name):
        super().__init__(strategy_name)
        self.stat = {}

    def stat_linear(self, X, Y:torch.Tensor, name, linear_name):
        pass

    def stat_attn(self, X, Y:torch.Tensor, name, linear_name):
        pass

    def process(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, name, n_bits):
        # breakpoint()
        Q =  quantize_activation_per_token_absmax(Q.permute(0,2,1,3), n_bits=4)
        K =  quantize_activation_per_token_absmax(K.permute(0,2,1,3), n_bits=4)
        
        return Q.permute(0,2,1,3), K.permute(0,2,1,3) ,V


class DecoderChannelScaleProcessor(AttentionProcessor):
    """
    Decoder processor that applies channel-wise scaling based on high/low attention patterns.

    Hook Expectations:
    ------------------
    stat_linear(X, Y, name, linear_name):
        X: Input to linear layer
           - Shape: (B, N, C) where B=batch, N=num_tokens, C=channels
        Y: Output from linear layer (stores full tensor for later analysis)
           - For q_proj: (B, N, C) - query projections
           - For k_proj: (B, N, C) - key projections
           - For v_proj: (B, N, C) - value projections
        name: Attention module name
        linear_name: Specific linear layer ("q_proj", "k_proj", "v_proj")

        STORES: Full Y tensor for each projection to analyze in stat_attn

    stat_attn(X, Y, name, n_heads):
        X: Input to attention module
           - Shape: (B, N, C)
        Y: Output from attention module
           - Shape: (B, N, C)

        USES: Stored q_proj, k_proj, v_proj tensors from stat_linear
        COMPUTES:
          - Attention scores to identify high/low attention tokens
          - Channel-wise difference between high and low attention key features
          - Sign statistics for centering
        STORES:
          - 'diff': Per-head channel differences between high/low attention keys
          - 'sign': Aggregated sign statistics across tokens
          - 'order': Channel indices sorted by diff (for reordering)
    """

    def __init__(self, strategy_name):
        super().__init__(strategy_name)
        self.stat = {}

    def stat_linear(self, X, Y:torch.Tensor, name, linear_name):
        if name not in self.stat:
            self.stat[name] =  defaultdict()
        self.stat[name][linear_name] = Y

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2) 

    def stat_attn(self, X, Y:torch.Tensor, name, n_heads):
        if 'final' in name or 'cross' in name:
            q = self.stat[name]['q_proj']
            k = self.stat[name]['k_proj']
            v = self.stat[name]['v_proj']
            sign = torch.sign(torch.sign(k).mean(-2, keepdim=True))

            q = self._separate_heads(q, n_heads)
            k = self._separate_heads(k, n_heads)
            v = self._separate_heads(v, n_heads)
            
            # Attention
            _, _, _, c_per_head = q.shape
            attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
            attn = attn / math.sqrt(c_per_head)
            attn = torch.softmax(attn, dim=-1)
            attn = attn.mean(1).mean(1) #[B, T]
            mask = torch.where(attn > attn.mean(-1,keepdim=True),1,0)
            k_high=k.permute(2,0,1,3)[mask.squeeze().nonzero()].squeeze().reshape(1, -1, q.shape[-1]*n_heads)
            k_low= k.permute(2,0,1,3)[(1-mask).squeeze().nonzero()].squeeze().reshape(1, -1, q.shape[-1]*n_heads)
            k_high_mean = k_high.mean(-2)
            k_low_mean = k_low.mean(-2)
            diff = torch.abs(k_high_mean  - k_low_mean)

            diff = diff.reshape(n_heads, -1)
            if 'diff' not in self.stat[name].keys():
                # self.stat[name]['diff'] = torch.abs(dif.reshape(n_heads, -1))
                self.stat[name]['diff'] = diff
                self.stat[name]['sign'] = sign
            else:
                self.stat[name]['diff'] += diff
                self.stat[name]['sign'] += sign 

            self.stat[name]['order'] = torch.argsort(self.stat[name]['diff'], descending=False)

            

    def process(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, name):
        # diff = self.stat[name]['diff']
        order = self.stat[name]['order']
        sign = torch.sign(self.stat[name]['sign'])[None, ...].reshape(-1, 8, 1,Q.shape[-1]) 
        
        Q = Q.mul_(sign).permute(0,2,1,3)
        K = K.mul_(sign).permute(0,2,1,3)

        Q = torch.gather(Q, index=order[None, None,...].expand(Q.shape), dim=-1 )
        K = torch.gather(K, index=order[None, None,...].expand(K.shape), dim=-1 )

        
        scales_1= torch.linspace(1.0, 1.0, steps=Q.shape[-1]//2)[None, None, None,...].to(K.device)
        scales_2 = torch.linspace(1.0, 1.25, steps=Q.shape[-1]//2)[None, None, None,...].to(K.device)
        scales = torch.cat([scales_1, scales_2], dim=-1)

   
        # breakpoint()
        K.mul_(1/scales)
        Q.mul_(scales)
        return Q.permute(0,2,1,3), K.permute(0,2,1,3), V

        
class DecoderRecenterProcessor(AttentionProcessor):
    """
    Decoder processor that recenters activations based on high/low attention statistics.

    Hook Expectations:
    ------------------
    stat_linear(X, Y, name, linear_name):
        X: Input to linear layer
           - Shape: (B, N, C) where B=batch, N=num_tokens, C=channels
        Y: Output from linear layer (stores full tensor for later analysis)
           - For q_proj: (B, N, C) - query projections
           - For k_proj: (B, N, C) - key projections
           - For v_proj: (B, N, C) - value projections
        name: Attention module name (only processes 'final' or 'cross' attention)
        linear_name: Specific linear layer ("q_proj", "k_proj", "v_proj")

        STORES: Full Y tensor for each projection

    stat_attn(X, Y, name, n_heads):
        X: Input to attention module
           - Shape: (B, N, C)
        Y: Output from attention module
           - Shape: (B, N, C)

        USES: Stored q_proj, k_proj, v_proj tensors from stat_linear
        COMPUTES:
          - Attention scores to split tokens into high/low attention groups
          - Mean absolute values of q and k projections
          - Difference between high and low attention key features
          - Sign statistics combined with query means
        STORES:
          - 'diff': Channel differences for high/low attention keys
          - 'sign': Combined sign and query mean statistics
          - 'q': Accumulated query absolute means
          - 'k': Accumulated key absolute means
          - 'order': Channel indices sorted by diff (descending)
    """

    def __init__(self, strategy_name):
        super().__init__(strategy_name)
        self.stat = {}

    def stat_linear(self, X, Y:torch.Tensor, name, linear_name):
        if name not in self.stat:
            self.stat[name] =  defaultdict()
        self.stat[name][linear_name] = Y

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2) 

    def stat_attn(self, X, Y:torch.Tensor, name, n_heads):
        if 'final' in name or 'cross' in name:
            q = self.stat[name]['q_proj']
            k = self.stat[name]['k_proj']
            v = self.stat[name]['v_proj']
            sign = torch.sign(torch.sign(k).mean(-2, keepdim=True))

            k_mean = torch.abs(k).mean(-2)[0]
            q_mean = torch.abs(q).mean(-2)[0]


            q = self._separate_heads(q, n_heads)
            k = self._separate_heads(k, n_heads)
            v = self._separate_heads(v, n_heads)
            _, _, _, c_per_head = q.shape
            attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
            attn = attn / math.sqrt(c_per_head)
            attn = torch.softmax(attn, dim=-1)
            attn = attn.mean(1).mean(1) #[B, T]
            mask = torch.where(attn > attn.mean(-1,keepdim=True),1,0)


            k_high=k.permute(2,0,1,3)[mask.squeeze().nonzero()].squeeze().reshape(1, -1, q.shape[-1]*n_heads)
            k_low= k.permute(2,0,1,3)[(1-mask).squeeze().nonzero()].squeeze().reshape(1, -1, q.shape[-1]*n_heads)
            k_high_mean = k_high.mean(-2)[0]
            k_low_mean = k_low.mean(-2)[0]
            diff = torch.abs(k_high_mean  - k_low_mean)
            if 'diff' not in self.stat[name].keys():
                # self.stat[name]['diff'] = torch.abs(dif.reshape(n_heads, -1))
                self.stat[name]['diff'] = diff  
                self.stat[name]['sign'] = sign + q_mean
                self.stat[name]['q'] = q_mean
                self.stat[name]['k'] = k_mean 
            else:
                self.stat[name]['diff'] += diff
                self.stat[name]['sign'] += (sign  + q_mean)
                self.stat[name]['q'] += q_mean
                self.stat[name]['k'] += k_mean 

            # breakpoint()
            self.stat[name]['order'] = torch.argsort(self.stat[name]['diff'].reshape(n_heads, -1), descending=True)
            # self.stat[name]['order'] = torch.argsort(self.stat[name]['q'].reshape(n_heads, -1), descending=True)
            # self.stat[name]['order'] = torch.argsort(self.stat[name]['k'].reshape(n_heads, -1), descending=True)
            # self.stat[name]['topk_diff'] = torch.topk(self.stat[name]['diff'].reshape(n_heads,-1), largest=True, k=2)[1]
            # self.stat[name]['topk_q_max'] = torch.topk(self.stat[name]['q'].reshape(n_heads,-1), largest=True, k=2)[1]
            # self.stat[name]['topk_k_max'] = torch.topk(self.stat[name]['k'].reshape(n_heads,-1), largest=True, k=2)[1]
            

            

    def process(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, name, n_bits):

        Q = Q.permute(0,2,1,3)
        K = K.permute(0,2,1,3)

        K = K - K.mean(1, keepdim=True)
        Q =  quantize_activation_per_token_absmax(Q, n_bits=4)
        K =  quantize_activation_per_token_absmax(K, n_bits=4)
        return Q.permute(0,2,1,3), K.permute(0,2,1,3), V


# ============================================================================
# Image Encoder Processor
# ============================================================================


class EncoderAttentionProcessor(AttentionProcessor):
    """
    Baseline processor for image encoder attention layers (ViT-based).

    Hook Expectations:
    ------------------
    stat_linear(X, Y, name, linear_name):
        X: Input to linear layer
           - For 'qkv' projection: (B, H, W, C) reshaped from image patches
             where H, W are spatial dimensions (e.g., 64x64 for 1024x1024 image)
           - For 'proj' (output projection): (B, H, W, C)
        Y: Output from linear layer
           - For 'qkv': (B, H, W, 3*C) - combined Q, K, V projections
           - For 'proj': (B, H, W, C) - output projection
        name: Encoder block name (e.g., "image_encoder.blocks.0.attn")
        linear_name: "qkv" or "proj"

        NOT USED in base processor - no statistics collected

    stat_attn(X, Y, name, n_heads):
        X: Input to encoder attention block
           - Shape: (B, H, W, C) spatial feature maps
           - Example: (1, 64, 64, 1280) for ViT-L
        Y: Output from encoder attention block
           - Shape: (B, H, W, C) same as input

        NOT USED in base processor - no statistics collected

    Note: Encoder attention is different from decoder:
      - Input is 4D (B, H, W, C) not 3D (B, N, C)
      - QKV are computed together via single 'qkv' linear layer
      - May use windowed attention with window_partition/unpartition
      - Uses relative positional encodings
    """

    def __init__(self, strategy_name:str='base', n_bits=8):
        super().__init__(strategy_name)
        self.n_bits = n_bits
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
        B, H, W, C = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x)  # (B, H, W, 3*C)
        qkv = qkv.permute(0, 3, 1, 2).contiguous()  # (B, 3*C, H, W)
        qkv = qkv.view(B, 3, module.num_heads, -1, H, W)  # (B, 3, num_heads, C_per_head, H, W)
        qkv = qkv.permute(1, 0, 2, 4, 5, 3).contiguous()  # (3, B, num_heads, H, W, C_per_head)

        # Flatten spatial dimensions: (3, B, num_heads, H*W, C_per_head)
        qkv = qkv.view(3, B, module.num_heads, H * W, -1)
        # Merge batch and heads: (3, B*num_heads, H*W, C_per_head)
        qkv = qkv.view(3, B * module.num_heads, H * W, -1)

        q, k, v = qkv[0], qkv[1], qkv[2]
        q = quantize_activation_per_token_absmax(q, self.n_bits)
        k = quantize_activation_per_token_absmax(k, self.n_bits)
        v = quantize_activation_per_channel_absmax(v, self.n_bits)
        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        attn = quantize_activation_per_token_absmax(attn, self.n_bits)

        out = attn @ v  # (B*num_heads, H*W, C_per_head)
        out = out.view(B, module.num_heads, H, W, -1)
        out = out.permute(0, 2, 3, 1, 4).contiguous()
        out = out.view(B, H, W, -1)
        x = module.proj(out)

        return x



class EncoderHighLowAttentionProcessor(EncoderAttentionProcessor):
    """
    Encoder processor that collects energy score statistics from attention inputs.

    Hook Expectations:
    ------------------
    stat_linear(X, Y, name, linear_name):
        X: Input to linear layer
           - For 'qkv': (B, H, W, C) spatial feature maps
           - For 'proj': (B, H, W, C)
        Y: Output from linear layer
           - For 'qkv': (B, H, W, 3*C)
           - For 'proj': (B, H, W, C)
        name: Encoder block name
        linear_name: "qkv" or "proj"

        NOT USED - no linear layer statistics collected

    stat_attn(X, Y, name, n_heads):
        X: Input to encoder attention block
           - Shape: (B, H, W, C) spatial feature maps
           - May be tuple (X,) - automatically unpacked
           - Example shapes:
             * Global attention: (1, 64, 64, 1280) for full image
             * Windowed attention: (num_windows, window_size, window_size, C)
        Y: Output from encoder attention block
           - Shape: (B, H, W, C)

        COMPUTES:
          - Reshapes X from (B, H, W, C) to (B, H*W, C)
          - Normalizes features: X_norm = X / ||X||_2
          - Energy score: ELU(X_norm @ X_norm^T - 0.9)
            Shape: (B, H*W, H*W) similarity matrix
        STORES:
          - 'energy_score_mean': Mean energy score per sample
          - 'energy_score_std': Std deviation of energy scores
          - 'energy_score_max': Maximum energy score
          - 'energy_score_min': Minimum energy score
          - 'energy_score_raw': Full energy score matrices (for correlation analysis)

    Note: Energy scores measure token-to-token similarity in the input features.
          Higher correlation between attention and energy indicates attention
          follows feature similarity patterns.
    """


    def __init__(self, strategy_name='recentered', n_bits=8):
        super().__init__(strategy_name)
        self.n_bits = n_bits 
        self.stat = {}

    def stat_linear(self, X, Y: torch.Tensor, name, linear_name):
        pass

    def stat_attn(self, X, Y: torch.Tensor, name, n_heads):
        pass
        # x= X[0]
        # B, H, W, C =  x.shape
        # x = x.view(B, -1, C)

        # B, H, W, _ = x.shape
        # # qkv with shape (3, B, nHead, H * W, C)
        # qkv = module.qkv(x).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
        # # q, k, v with shape (B * nHead, H * W, C)
        # q, k, v = qkv.reshape(3, B * module.num_heads, H * W, -1).unbind(0)
        # attn = (q * module.scale) @ k.transpose(-2, -1)

        # if module.use_rel_pos:
        #     attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

        # attn = attn.softmax(dim=-1)
        
        

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


    
    def cal_energy(self,  X:torch.Tensor, margin=0.9):
        """
        Calculate the energy of the input tensor.
        """
        X = F.normalize(X, dim=-1, p=2)
        score_map = F.elu(X @ X.transpose(-1, -2) - margin)
        scores = score_map.mean(-1)
        return scores

    def quantize_activation_per_token_absmax(self, X, mask):
        scales = X.abs().max(dim=-1, keepdim=True)[0]
        q_max = 2 ** (self.n_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        X.div_(scales).round_().mul_(scales)
        return X

    def quantize_activation_per_channel_absmax(self, X, mask):
        scales = X.abs().max(dim=1, keepdim=True)[0]
        q_max_8 = 2 ** (8 - 1) - 1
        q_max_4 = 2 ** (self.n_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max_8)
        X.div_(scales).round_().mul_(scales)
        return X



    def process(self,  x:torch.Tensor, module, module_name:str=None):
        B, H, W, C = x.shape
        energy = self.cal_energy(x, 0.9)
        mask = torch.where(energy < 0.8, 1, 0)
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = module.qkv(x)  # (B, H, W, 3*C)
        qkv = qkv.permute(0, 3, 1, 2).contiguous()  # (B, 3*C, H, W)
        qkv = qkv.view(B, 3, module.num_heads, -1, H, W)  # (B, 3, num_heads, C_per_head, H, W)
        qkv = qkv.permute(1, 0, 2, 4, 5, 3).contiguous()  # (3, B, num_heads, H, W, C_per_head)

        # Flatten spatial dimensions: (3, B, num_heads, H*W, C_per_head)
        qkv = qkv.view(3, B, module.num_heads, H * W, -1)
        # Merge batch and heads: (3, B*num_heads, H*W, C_per_head)
        qkv = qkv.view(3, B * module.num_heads, H * W, -1)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # q = self.quantize_activation_per_token_absmax(q, self.n_bits, mask)
        # k = self.quantize_activation_per_token_absmax(k, self.n_bits, mask)
        # v = self.quantize_activation_per_channel_absmax(v, self.n_bits, mask)

        attn = (q * module.scale) @ k.transpose(-2, -1)

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        # print('got here')
        attn = self.quantize_activation_per_token_absmax(attn, self.n_bits, mask)

        out = attn @ v  # (B*num_heads, H*W, C_per_head)
        out = out.view(B, module.num_heads, H, W, -1)
        out = out.permute(0, 2, 3, 1, 4).contiguous()
        out = out.view(B, H, W, -1)
        x = module.proj(out)

        return x
   


class EncoderRecenterAttentionProcessor(AttentionProcessor):
    """
    Encoder processor that recenters QKV projections by subtracting spatial means.

    Hook Expectations:
    ------------------
    stat_linear(X, Y, name, linear_name):
        X: Input to linear layer
           - For 'qkv': (B, H, W, C) spatial feature maps
           - For 'proj': (B, H, W, C)
        Y: Output from linear layer
           - For 'qkv': (B, H, W, 3*C)
           - For 'proj': (B, H, W, C)
        name: Encoder block name
        linear_name: "qkv" or "proj"

        NOT USED - no statistics collected

    stat_attn(X, Y, name, n_heads):
        X: Input to encoder attention block
           - Shape: (B, H, W, C) spatial feature maps
        Y: Output from encoder attention block
           - Shape: (B, H, W, C)

        NOT USED - no statistics collected

    Processing Strategy:
    -------------------
    In the process() method, this processor:
      1. Computes spatial mean: x_mean = x.mean(H).mean(W)
      2. Subtracts mean: x_hat = x - x_mean
      3. Projects: qkv_hat = QKV(x_hat), qkv_mean = QKV(x_mean) - bias
      4. Computes attention with recentered keys:
         attn = (q_hat @ k_hat^T) + (q_mean @ k_hat^T)
      5. Applies attention to recentered values: out = attn @ (v_hat + v_mean)

    This helps with quantization by centering activations around zero.
    """

    def __init__(self, strategy_name='recentered'):
        super().__init__(strategy_name)
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
        B, H, W, C = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        bias = module.qkv.bias[None, None, ...]
        x_mean = x.mean(1, keepdim=True).mean(2, keepdim=True)
        x_hat = x - x_mean

        # Process x_hat
        qkv_hat = module.qkv(x_hat)  # (B, H, W, 3*C)
        qkv_hat = qkv_hat.permute(0, 3, 1, 2).contiguous()  # (B, 3*C, H, W)
        qkv_hat = qkv_hat.view(B, 3, module.num_heads, -1, H, W)
        qkv_hat = qkv_hat.permute(1, 0, 2, 4, 5, 3).contiguous()
        qkv_hat = qkv_hat.view(3, B, module.num_heads, H * W, -1)
        qkv_hat = qkv_hat.view(3, B * module.num_heads, H * W, -1)

        # Process x_mean
        qkv_mean = module.qkv(x_mean) - bias  # (B, 1, 1, 3*C)
        qkv_mean = qkv_mean.permute(0, 3, 1, 2).contiguous()  # (B, 3*C, 1, 1)
        qkv_mean = qkv_mean.view(B, 3, module.num_heads, -1, 1, 1)
        qkv_mean = qkv_mean.permute(1, 0, 2, 4, 5, 3).contiguous()
        qkv_mean = qkv_mean.view(3, B, module.num_heads, 1, -1)
        qkv_mean = qkv_mean.view(3, B * module.num_heads, 1, -1)

        q_hat, k_hat, v_hat = qkv_hat[0], qkv_hat[1], qkv_hat[2]
        q_mean, k_mean, v_mean = qkv_mean[0], qkv_mean[1], qkv_mean[2]

        attn = (q_hat * module.scale) @ k_hat.transpose(-2, -1)
        attn_mean = (q_mean * module.scale) @ k_hat.transpose(-2, -1)
        attn = attn + attn_mean

        if module.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q_hat + q_mean, module.rel_pos_h, module.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        out = (attn @ v_hat) + v_mean  # (B*num_heads, H*W, C_per_head)
        out = out.view(B, module.num_heads, H, W, -1)
        out = out.permute(0, 2, 3, 1, 4).contiguous()
        out = out.view(B, H, W, -1)
        x = module.proj(out)

        return x