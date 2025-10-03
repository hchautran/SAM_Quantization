import math
import torch
import torch.nn as nn
from functools import  partial
import os
import logging
import time
from train.utils.dataloader import get_im_gt_name_dict, Resize
from abc import abstractmethod
from data_utils import OnlineDataset
from torchvision import transforms
from segment_anything.modeling.transformer import TwoWayAttentionBlock, TwoWayTransformer, Attention
from torch.utils.data import DataLoader
from data_utils import OnlineDataset
from segment_anything import SamPredictor, sam_model_registry
from matplotlib import pyplot as plt
from functools import partial
from accelerate import Accelerator
import train.utils.misc as misc
from tqdm.auto import tqdm
from utils import show_mask_image
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
            if isinstance(component, Attention):
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

    def calibrate(self, predictor: SamPredictor, modules, num_samples=32):
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




class SignProcessor(ProcessStrategy):

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






class DoNothingProcessor(ProcessStrategy):

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


class AttnBasedProcessor(ProcessStrategy):

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

        
        




class AttnBasedProcessor(ProcessStrategy):

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
            # breakpoint()

            q = self.stat[name]['q_proj']
            k = self.stat[name]['k_proj']
            v = self.stat[name]['v_proj']
            sign = torch.sign(torch.sign(k).mean(-2, keepdim=True))

            #cal q_max 
            # breakpoint()
            k_mean = torch.abs(k).mean(-2)[0]

            #cal k_max 
            q_mean = torch.abs(q).mean(-2)[0]


            #cal diff

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
        # diff = self.stat[name]['diff']
        # order = self.stat[name]['order']
        # sign = torch.sign(self.stat[name]['sign'])[None, ...].reshape(-1, 8, 1,Q.shape[-1]) 
        
        Q = Q.permute(0,2,1,3)
        K = K.permute(0,2,1,3)

        # Q = torch.gather(Q, index=order[None, None,...].expand(Q.shape), dim=-1 )
        # K = torch.gather(K, index=order[None, None,...].expand(K.shape), dim=-1 )

        
        # # scales_1= torch.linspace(1.0, 1.0, steps=Q.shape[-1]//2)[None, None, None,...].to(K.device)
        # # scales_2 = torch.linspace(1.0, 1.25, steps=Q.shape[-1]//2)[None, None, None,...].to(K.device)
        # # scales = torch.cat([scales_1, scales_2], dim=-1)
        # # breakpoint()
        # # K.mul_(1/scales)
        # # Q.mul_(scales)
        # Q_shape = Q.shape
        # K_shape = K.shape
        # breakpoint()
        # Q = Q.reshape(-1, K_shape[-1])
        # K = K.reshape(-1, K_shape[-1])
        # topk_indices = torch.tensor([0,1])

        # breakpoint()
        # q_backup = Q[:, :, :,topk_indices].detach().clone()
        # k_backup = K[:, :, :,topk_indices].detach().clone()
        K = K - K.mean(1, keepdim=True)
        Q =  quantize_activation_per_token_absmax(Q, n_bits=4)
        K =  quantize_activation_per_token_absmax(K, n_bits=4)
        # breakpoint()
        # Q[:, :, :,topk_indices] = q_backup
        # K[:, :, :,topk_indices] = k_backup
        return Q.permute(0,2,1,3), K.permute(0,2,1,3), V
        # return  Q.reshape(Q_shape), K.reshape(K_shape) ,V


# ============================================================================
# Image Encoder Processor
# ============================================================================


class ImageEncoderProcessor(ProcessStrategy):
    """
    Processor for calibrating and processing image encoder attention layers.

    This processor is designed specifically for the ViT-based image encoder
    in SAM, which has a different architecture than the mask decoder.
    """

    def __init__(self, strategy_name):
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
        if name not in self.stat:
            self.stat[name] = defaultdict()
        self.stat[name][linear_name] = Y

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

    def stat_attn(self, X, Y: torch.Tensor, name, n_heads):
        pass


    def process(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, name, n_bits):
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
        # Center K around mean
        # K = K - K.mean(1, keepdim=True)

        # # Quantize Q and K
        # Q = quantize_activation_per_token_absmax(Q, n_bits=n_bits)
        # K = quantize_activation_per_token_absmax(K, n_bits=n_bits)

        return Q, K, V

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
            if isinstance(component, Attention):
                # Hook the QKV linear layer
                for linear_name, m in component.named_modules():
                    if isinstance(m, nn.Linear) and linear_name == 'qkv':
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


