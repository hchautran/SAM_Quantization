# %%
import os
import math
from IPython.core.pylabtools import figsize
from cv2.detail import AffineBasedEstimator
from matplotlib.typing import MarkerType

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_image_predictor import SAM2ImagePredictor
from observer import ObserverBase
from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_unpartition,
    window_partition
)
from sam2.sam2_video_predictor_legacy import SAM2VideoPredictor
from typing import Union, Tuple
from sam2.modeling.backbones.hieradet import MultiScaleAttention, MultiScaleBlock
from tkinter.constants import E
import numpy as np

def do_pool(x:torch.Tensor, pool:nn.Module, norm:nn.Module = None) :
    if pool is None:
        return x

    # (B, H , W, C) -> (B, C, H, W)
    x = x.permute(0,3,1,2)
    x = pool(x)
    # (B, C , H', W') -> (B, W', H', C)
    x = x.permute(0,2,3,1)
    if norm:
        x = norm(x)
    return x

def to_numpy(x:torch.Tensor):
    return x.detach().cpu().numpy()


def get_histc(x:torch.Tensor, bins=200, min=-3.0, max=3.0):
    x = x.flatten()
    return torch.histc(x, bins=bins, min=min, max=max)

class Distribution():
    def __init__(self, n_channels:int, dist_name:str ,mean:np.ndarray, max:np.ndarray, min:np.ndarray, p75:np.ndarray, p25:np.ndarray, p99:np.ndarray, p1:np.ndarray):
        self.n_channels = n_channels
        self.mean = mean
        self.max = max
        self.min = min
        self.p75 = p75
        self.p25 = p25
        self.p99 = p99
        self.p1 = p1
        self.dist_name = dist_name


    def plot_channel_distribution(self, ax):
        x = np.arange(self.n_channels)
        ax.plot(x, self.mean, color='black', linewidth=-1, marker='_')
        ax.plot(x, self.min, color='skyblue', linewidth=-1, marker='_')
        ax.plot(x, self.max, color='skyblue', linewidth=-1, label='Min/Max', marker='_')
        ax.scatter(x, self.p1, color='red', linewidth=1, marker='_')
        ax.scatter(x, self.p99, color='red', linewidth=1, label='1/99 Percentile', marker='_')
        ax.scatter(x, self.p25, color='orange', linewidth=1, marker='_')
        ax.scatter(x, self.p75, color='orange', linewidth=1, label='25/75 Percentile', marker='_')
        ax.set_title(self.dist_name)
        ax.set_ylim([-10.0, 10.0])
        ax.set_xlabel('Hidden dimension index')
        ax.set_ylabel('Activation value')

    def box_plot_channel_distribution(self,ax):
        pass


def get_activation_distribution(activations:torch.Tensor, title:str) -> Distribution:
# Compute percentiles along the sample axis (axis=-1)


    activations = np.reshape(to_numpy(activations), (-1, activations.shape[-1]))

    min_val= np.min(activations, axis=0)
    max_val = np.max(activations, axis=0)
    mean= np.mean(activations, axis=0)
    p1 = np.percentile(activations, 1, axis=0)
    p99 = np.percentile(activations, 99, axis=0)
    p25 = np.percentile(activations, 25, axis=0)
    p75 = np.percentile(activations, 75, axis=0)

    return Distribution(
        n_channels=activations.shape[-1],
        dist_name=title,
        mean=mean,
        min=min_val,
        max=max_val,
        p1=p1,
        p99=p99,
        p25=p25,
        p75=p75
    )
# %%


class EnergyObserver(ObserverBase):

    def __init__(self, module_list):
        super().__init__(module_list)

    def cal_energy(self, X:torch.Tensor, margin=0.5):
        """
        Calculate the energy of the input tensor.
        """
        X = F.normalize(X.reshape(-1, X.shape[-1]), dim=-1, p=2)
        score_map = F.elu(X @ X.transpose(-1, -2) - margin)
        scores = score_map.mean(-1)
        return scores

    def register_energy_hook(self, model:nn.Module):
        self.dictionary['energy'] = defaultdict(list)

        def pre_hook(module, input, name):
            # calculate energy
            if input[0].shape[-2] <= 64:
                self.dictionary['energy'][name].append(self.cal_energy(input[0]).cpu().detach().numpy())

        self.register_hooks(model, pre_hook=pre_hook)

    def show_video_energy_map(self, layer_idx, frame_idx):
        names = list(ObserverBase.dictionary['energy'].keys())
        energies = list(ObserverBase.dictionary['energy'][names[layer_idx]])
        width = int(math.sqrt(energies[frame_idx].shape[0]))
        energies_map= energies[frame_idx].reshape(width, -1)
        plt.imshow(energies_map)
        plt.colorbar()
        plt.show()


# %%

class AttnObserver(MultiScaleAttention):
    observer_state = defaultdict(list)

    @staticmethod
    def clear_dict():
        AttnObserver.observer_state = defaultdict(list)

    @staticmethod
    def show_energy_map(energies_map:np.ndarray):
        plt.imshow(energies_map)
        plt.colorbar()
        plt.show()

    @staticmethod
    def cal_density(X:torch.Tensor, margin:float=0.5,independent_heads=True):
        if not independent_heads:
            X = X.mean(2, keepdim=True)

        X = X.permute(0, 2, 1, 3)
        X = F.normalize(X, p=2, dim=-1)
        score_map = F.elu(X @ X.transpose(-1, -2) - margin)
        scores = score_map.mean(-1)
        return scores


        # Implement energy calculation logic here


    def forward(self, x:torch.Tensor)->Tuple[torch.Tensor]:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)


        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )

        AttnObserver.observer_state['q'].append(to_numpy(q))
        AttnObserver.observer_state['k'].append(to_numpy(k))
        AttnObserver.observer_state['v'].append(to_numpy(v))
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x, k, v



class BlockObserver(MultiScaleBlock):
    observer_state = defaultdict(list)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @staticmethod
    def clear_dict():
        BlockObserver.observer_state = defaultdict(list)

    def cal_energy(self, X:torch.Tensor, margin=0.0):
        """
        Calculate the energy of the input tensor.
        """
        X = X.squeeze()
        if len(X.shape) == 4:
            B, W, H, C = X.shape
            X = X.reshape(B, W*H, C)
            X = F.normalize(X, dim=-1, p=2)
        else:
            X = F.normalize(X.reshape(-1, X.shape[-1]), dim=-1, p=2)

        score_map = F.elu(X @ X.transpose(-1, -2) - margin)
        scores = score_map.mean(-1)
        return scores

    @staticmethod
    def show_energy_map(energies_map:np.ndarray):
        B,  W, H, nHeads = energies_map.shape
        energies_map = energies_map.squeeze()
        if nHeads == 1:
            fig, axs = plt.subplots(nrows=1, ncols=nHeads, squeeze=True, figsize=(8, 8*nHeads))
            print(energies_map.shape)
            pos = axs.imshow(energies_map)
            fig.colorbar(pos, ax=axs)
        else:
            fig, axs = plt.subplots(nrows=1, ncols=nHeads+1, squeeze=True, figsize=(8*nHeads + 1, 8))
            total_e = energies_map.sum(-1)
            for i in range(nHeads):
                axs[i].imshow(energies_map[:,:,i])
            axs[-1].imshow(total_e)
        plt.show()



    def _to_numpy(self, x:torch.Tensor):
        return x.cpu().detach().numpy()


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)
            BlockObserver.observer_state['attn_input'].append(x)
            scores = self.cal_energy(x).reshape(-1, self.window_size, self.window_size)[..., None]
            BlockObserver.observer_state['attn_input_E_AP'].append(self._to_numpy(scores))
            scores = window_unpartition(scores, window_size, pad_hw, (H, W))
            BlockObserver.observer_state['attn_input_E'].append(self._to_numpy(scores))


        # Window Attention + Q Pooling (if stage change)


        x, k, v = self.attn(x)
        if window_size > 0:
            k_scores = AttnObserver.cal_density(k, independent_heads=False).reshape(k.shape[0], self.window_size, self.window_size, -1)
            v_scores = AttnObserver.cal_density(v, independent_heads=False).reshape(k.shape[0], self.window_size, self.window_size, -1)
            k_scores = window_unpartition(k_scores, window_size, pad_hw, (H, W))
            v_scores = window_unpartition(v_scores, window_size, pad_hw, (H, W))

            AttnObserver.observer_state['k_E'].append(to_numpy(k_scores))
            AttnObserver.observer_state['v_E'].append(to_numpy(v_scores))


        if self.q_stride:

            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        BlockObserver.observer_state['attn_output'].append(self._to_numpy(x))
        x = shortcut + self.drop_path(x)
        # MLP
        BlockObserver.observer_state['mlp_input'].append(self._to_numpy(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        BlockObserver.observer_state['mlp_output'].append(self._to_numpy(x))
        return x


def apply_energy_observer(model:nn.Module):
    for _, module in model.named_modules():
        if isinstance(module, MultiScaleBlock):
            module.__class__ = BlockObserver
        if isinstance(module, MultiScaleAttention):
            module.__class__ = AttnObserver



# %%
#

#build model
checkpoint = './checkpoints/sam2.1_hiera_small.pt'
model_cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'
# video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)
sam2 = build_sam2(model_cfg, checkpoint)
predictor = SAM2ImagePredictor(sam2)
# inference video
observer = EnergyObserver(module_list=(MultiScaleBlock,))

apply_energy_observer(predictor.model)

# observer.register_energy_hook(predictor.model)
# observer.inference_video(predictor=predictor, show_video=True)
observer.inference_image(predictor, show_image=False, image_dir='./notebooks/images/cars.jpg')
# observer.clear_hook()

# %%
# observer.show_video_energy_map(layer_idx=35, frame_idx=0)
# observer.clear_dict()
# %%
def get_activation_boxplot(activations:torch.Tensor, title:str, ax):
    data = to_numpy(activations.reshape(-1, activations.shape[-1]))

    # Convert data to a pandas DataFrame for easier handling
    channel_names = [f"{i+1}" for i in  range(activations.shape[-1])]
    df = pd.DataFrame(data, columns=channel_names)

    # Step 2: Plot box plots for each channel
    df.boxplot(column=channel_names, ax=ax)
    ax.grid(True)

def get_fft(activation:torch.Tensor):
    activation = activation.reshape(-1, activation.shape[-1])
    activation_fft = torch.fft.fft(activation, norm='ortho', dim=0)
    freqs = torch.fft.fftfreq(activation.shape[-1])
    return activation_fft, freqs

def get_ifft(activation:torch.Tensor):
    if len(activation.shape) > 2:
        activation = activation.reshape(-1, activation.shape[-1])
    activation_fft = torch.fft.ifft(activation, norm='ortho')
    return activation_fft



n_bins = 200
min = -10.0
max = 10.0
# mid = torch.zeros(200,)
# low = torch.zeros(200,)
# high = torch.zeros(200,)
for i in range(len(BlockObserver.observer_state['attn_input'])):
    attention_input = BlockObserver.observer_state['attn_input'][i]
    score_map = BlockObserver.observer_state['attn_input_E_AP'][i]
    # print(score_map.mean(-1))
    q = np.array([25, 75])
    score_map = score_map.squeeze().reshape(score_map.shape[0],  -1)
    score_mean = score_map.mean(-1)
    # print(score_mean)

    # score_type = np.percentile(score_map, q, axis=-1, )
    idx = score_mean.argsort(-1)

    high_input = attention_input[idx[-10:]]
    low_input = attention_input[idx[:-10]]
    high_input_fft, freqs = get_fft(high_input)
    low_input_fft, freqs = get_fft(low_input)

    print(high_input.shape, low_input.shape)
    print(high_input_fft.shape, low_input_fft.shape)
    print('--'*100)
    # axes[0]
    fig, axes = plt.subplots(2,2, figsize=(18, 6), squeeze=True, sharey=True)

    high_dist = get_activation_distribution(high_input, 'high').plot_channel_distribution(axes[0][0])
    low_dist = get_activation_distribution(low_input, 'low').plot_channel_distribution(axes[1][0])
    high_dist_fft =get_activation_distribution(high_input_fft.real, 'high_fft').plot_channel_distribution(axes[0][1])

    low_dist_fft = get_activation_distribution(low_input_fft.real, 'low_fft').plot_channel_distribution(axes[1][1])
    
    # get_activation_boxplot(high_input, 'high', axes[0][0])
    # get_activation_boxplot(low_input,  'low', axes[0][1])
    # get_activation_boxplot(high_input_fft.real, 'high_fft', axes[1][0])
    # get_activation_boxplot(low_input_fft.real, 'low_fft', axes[1][1])
    # fig.xlabel("Channels")
    # fig.ylabel("Values" )



    # n_bins = 200
    # bins = torch.linspace(min, max, n_bins)
    # fig = plt.figure(figsize=(10, 10))
    # high_hist = get_histc(high_input, bins=n_bins, min=min, max=max)
    # low_hist = get_histc(low_input, bins=n_bins, min=min, max=max)
    # high_hist = high_hist / high_hist.sum()
    # low_hist = low_hist / low_hist.sum()

    # plt.bar(bins, to_numpy(high_hist), width=(bins[1] - bins[0]).item(), alpha=0.5, label='high')
    # plt.bar(bins, to_numpy(low_hist), width=(bins[1] - bins[0]).item(), alpha=0.5, label='low')
    # plt.legend()
    # plt.show()

    # BlockObserver.show_energy_map(score_map)
# %%

# for i in range(len(AttnObserver.observer_state['k_E'])):
    # print(i)
    # print('q',AttnObserver.observer_state['k_E'][i].shape)

    # k_map = AttnObserver.observer_state['k_E'][i]
    # v_map = AttnObserver.observer_state['v_E'][i].squeeze()
    # BlockObserver.show_energy_map(k_map)


# AttnObserver.clear_dict()
# BlockObserver.clear_dict()

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Generate or load data
# Simulate data for multiple channels (e.g., 4 channels with 100 samples each)
data = to_numpy(high_input.reshape(-1, high_input.shape[-1]))

# Convert data to a pandas DataFrame for easier handling
channel_names = [f"{i+1}" for i in  range(high_input.shape[-1])]
df = pd.DataFrame(data, columns=channel_names)

# Step 2: Plot box plots for each channel
plt.figure(figsize=(10, 6))
df.boxplot(column=channel_names)
plt.title("Box Plot for Each Channel")
plt.xlabel("Channels")
plt.ylabel("Values")
plt.grid(True)
plt.show()



# %%
BlockObserver.clear_dict()
AttnObserver.clear_dict()

# %%
