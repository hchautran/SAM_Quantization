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
from quant_utils import AttentionProcessor

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