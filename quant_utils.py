import math
import torch
import torch.nn as nn
from functools import  partial
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




class DecoderSignProcessor(AttentionProcessor):

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