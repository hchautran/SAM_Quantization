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

module_name_dict = {
    'layers.0.self_attn.q_proj': 'p2p_q_0',
    'layers.0.self_attn.k_proj': 'p2p_k_0', 
    'layers.0.self_attn.v_proj' : 'p2p_v_0', 
    'layers.0.cross_attn_token_to_image.q_proj': 'p2i_q_0', 
    'layers.0.cross_attn_token_to_image.k_proj' : 'p2i_k_0', 
    'layers.0.cross_attn_token_to_image.v_proj' : 'p2i_v_0', 
    'layers.0.cross_attn_image_to_token.q_proj' : 'i2p_q_0', 
    'layers.0.cross_attn_image_to_token.k_proj' : 'i2p_k_0', 
    'layers.0.cross_attn_image_to_token.v_proj' : 'i2p_v_0', 
    'layers.1.self_attn.q_proj' : 'p2p_q_1', 
    'layers.1.self_attn.k_proj' : 'p2p_k_1', 
    'layers.1.self_attn.v_proj' : 'p2p_v_1', 
    'layers.1.cross_attn_token_to_image.q_proj' : 'p2i_q_1', 
    'layers.1.cross_attn_token_to_image.k_proj' : 'p2i_k_1', 
    'layers.1.cross_attn_token_to_image.v_proj' : 'p2i_v_1', 
    'layers.1.cross_attn_image_to_token.q_proj' : 'i2p_q_1', 
    'layers.1.cross_attn_image_to_token.k_proj' : 'i2p_k_1', 
    'layers.1.cross_attn_image_to_token.v_proj' : 'i2p_v_1', 
    'final_attn_token_to_image.q_proj' : 'p2i_q_final', 
    'final_attn_token_to_image.k_proj' : 'p2i_k_final', 
    'final_attn_token_to_image.v_proj' : 'p21_v_final', 
}

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


class W8A8Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        act_quant="per_token",
        quantize_output=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=8)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=8)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        q_x = self.act_quant(x)
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.output_quant(y)
        return q_y

    @staticmethod
    def from_float(
        module, weight_quant="per_channel", act_quant="per_token", quantize_output=False
    ):
        assert isinstance(module, torch.nn.Linear)
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            act_quant=act_quant,
            quantize_output=quantize_output,
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8
            )  # use 8-bit integer for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, act_quant={self.act_quant_name}, output_quant={self.output_quant_name})"


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
    def calibrate(self, predictor:SamPredictor,  modules, num_samples=32):
        # model.eval()

        def stat_hook(module, X, Y:torch.Tensor, name, linear_name):
            if isinstance(X, tuple):
                X = X[0]

            self.stat_tensor(X, Y, name, linear_name)

        hooks = []

        for name, component in predictor.model.named_modules():
            if isinstance(component, Attention):
                for linear_name, m in component.named_modules():
                    if isinstance(m, nn.Linear):
                        hooks.append(
                            m.register_forward_hook(partial(stat_hook, name=name, linear_name=linear_name))
                        )

                
        logger =setup_logger('./calib_logs', self.strategy_name)
        logger.info(f'______Using: {self.strategy_name}_______')

        #load data
        # model = self.accelerator.prepare(model) 
        for k in range(len(self.dataloaders)):
            dataloader = self.accelerator.prepare(self.dataloaders[k])
            print('valid_dataloader len:', len(dataloader))
            # logger.info(f"\nCalibarating {self.datasets[k]['name']}:")
            progress_bar = tqdm(total=num_samples, desc=f"Calibrating")
            for i, data_val in enumerate(dataloader):
                if i == num_samples: break 
                _, inputs_val, labels_val, _, labels_ori, ori_image = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label'], data_val['ori_im']
                # breakpoint()
                
                imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
                # breakpoint()
                predictor.set_image(imgs.squeeze())
                labels_boxes = misc.masks_to_boxes(labels_val[:,0,:,:]).cpu().numpy()
                masks, scores, logits = predictor.predict(
                    # model, 
                    box=labels_boxes, 
                    hq_token_only=False
                )
                progress_bar.update(1)
                if False:

                    plt.figure(figsize=(10, 10))
                    plt.imshow(imgs.squeeze())

                    
                    if len(masks) > 0:
                        show_mask_image(masks[0], plt.gca(), random_color=False)
                    
                    box = labels_boxes[0]
                    x0, y0 = box[0], box[1]
                    w, h = box[2] - box[0], box[3] - box[1]
                    plt.gca().add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
                    
                        
                    plt.title(f'Example {i} - Score: {scores[0]:.3f}')
                    plt.savefig(f'./sample_{i}.png')
                    plt.axis('off')
                    plt.show()


                #TODO: forward and record input/output stats


        for h in hooks:
            h.remove()

    def clear_dict(self):
        self.stat = {}




class SignProcessor(ProcessStrategy):

    def __init__(self, strategy_name):
        super().__init__(strategy_name)
        self.stat = {} 

    def stat_tensor(self, X, Y:torch.Tensor, name, linear_name):
        # attn_name = name 
        sign = torch.sign(torch.sign(Y).mean(-2, keepdim=True))
        if 'k' in linear_name:
            if name not in self.stat:
                self.stat[name] =  defaultdict() 
                self.stat[name][linear_name] = sign
            else:
                self.stat[name][linear_name] += sign


        
    def process(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, name):
        # name = module_name_dict[name]

        Q.mul_(self.stat[name]['k_proj'].sign())
        K.mul_(self.stat[name]['k_proj'].sign())
        return Q, K, V



class MyProcessor(ProcessStrategy):

    def __init__(self, strategy_name):
        super().__init__(strategy_name)
        self.stat = {} 

    def stat_tensor(self, X, Y:torch.Tensor, name, linear_name):
        # attn_name = name 
        sign = torch.sign(torch.sign(Y).mean(-2, keepdim=True))
        if 'k' in linear_name:
            if name not in self.stat:
                self.stat[name] =  defaultdict() 
                self.stat[name][linear_name] = sign
            else:
                self.stat[name][linear_name] += sign


        
    def process(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, name):
        # name = module_name_dict[name]

        Q.mul_(self.stat[name]['k_proj'].sign())
        K.mul_(self.stat[name]['k_proj'].sign())
        return Q, K, V


class DoNothingProcessor(ProcessStrategy):

    def __init__(self, strategy_name):
        super().__init__(strategy_name)
        self.stat = {} 

    def stat_tensor(self, X, Y:torch.Tensor, name, linear_name):
        pass
        

    def process(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, name):
        pass





