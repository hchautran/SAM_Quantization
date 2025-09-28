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
    def stat_attn(self, X, Y, name, n_heads):
        pass


        
    @abstractmethod
    def calibrate(self, predictor:SamPredictor,  modules, num_samples=32):
        # model.eval()

        def stat_linear_hook(module, X, Y:torch.Tensor, name, linear_name):
            if isinstance(X, tuple):
                X = X[0]
            self.stat_linear(X, Y, name, linear_name)

        def stat_attn_hook(module, X, Y:torch.Tensor, name, n_heads ):
            self.stat_attn(X, Y,  name, n_heads)


        linear_hooks = []
        attn_hooks = []
        mlp_hooks = []

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


        for h in linear_hooks:
            h.remove()
            
        for h in attn_hooks:
            h.remove()

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


        
    def process(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, name):
        sign = self.stat[name]['k_proj'].sign().reshape(-1, 16)[None, None, ...].permute(0,2,1,3)

        Q.mul_(sign)
        K.mul_(sign)
        return Q, K, V



class MyProcessor(ProcessStrategy):

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


        
    def process(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, name):
        # name = module_name_dict[name]
        Q.mul_(self.stat[name]['k_proj'].sign())
        K.mul_(self.stat[name]['k_proj'].sign())
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
            q = self.stat[name]['q_proj']
            k = self.stat[name]['k_proj']
            v = self.stat[name]['v_proj']
            sign = torch.sign(torch.sign(k).mean(-2, keepdim=True))

            #cal q_max 
            # breakpoint()
            k_mean = torch.abs(k).mean(-2)

            #cal k_max 
            q_mean = torch.abs(q).mean(-2)


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
            k_high_mean = k_high.mean(-2)
            k_low_mean = k_low.mean(-2)
            diff = torch.abs(k_high_mean  - k_low_mean)
            if 'diff' not in self.stat[name].keys():
                # self.stat[name]['diff'] = torch.abs(dif.reshape(n_heads, -1))
                self.stat[name]['diff'] = diff
                self.stat[name]['sign'] = sign
                self.stat[name]['q'] = q_mean
                self.stat[name]['k'] = k_mean 
            else:
                self.stat[name]['diff'] += diff
                self.stat[name]['sign'] += sign 
                self.stat[name]['q'] += q_mean
                self.stat[name]['k'] += k_mean 

            # breakpoint()
            self.stat[name]['order'] = torch.argsort(self.stat[name]['diff'].reshape(n_heads, -1), descending=True)
            self.stat[name]['order_q'] = torch.argsort(self.stat[name]['q'].reshape(n_heads, -1), descending=True)
            self.stat[name]['order_k'] = torch.argsort(self.stat[name]['k'].reshape(n_heads, -1), descending=True)
            # self.stat[name]['topk_diff'] = torch.topk(self.stat[name]['diff'].reshape(n_heads,-1), largest=True, k=2)[1]
            # self.stat[name]['topk_q_max'] = torch.topk(self.stat[name]['q'].reshape(n_heads,-1), largest=True, k=2)[1]
            # self.stat[name]['topk_k_max'] = torch.topk(self.stat[name]['k'].reshape(n_heads,-1), largest=True, k=2)[1]
            

            

    def process(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, name, n_bits):
        # diff = self.stat[name]['diff']
        order = self.stat[name]['order']
        sign = torch.sign(self.stat[name]['sign'])[None, ...].reshape(-1, 8, 1,Q.shape[-1]) 
        
        Q = Q.permute(0,2,1,3)
        K = K.permute(0,2,1,3)

        Q = torch.gather(Q, index=order[None, None,...].expand(Q.shape), dim=-1 )
        K = torch.gather(K, index=order[None, None,...].expand(K.shape), dim=-1 )

        
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
        topk_indices = torch.tensor([0,1])

        # breakpoint()
        q_backup = Q[:, :, :,topk_indices].detach().clone()
        k_backup = K[:, :, :,topk_indices].detach().clone()
        Q =  quantize_activation_per_token_absmax(Q, n_bits=4)
        K =  quantize_activation_per_token_absmax(K, n_bits=4)
        Q[:, :, :,topk_indices] = q_backup
        K[:, :, :,topk_indices] = k_backup
        return Q.permute(0,2,1,3), K.permute(0,2,1,3), V
        # return  Q.reshape(Q_shape), K.reshape(K_shape) ,V

        
