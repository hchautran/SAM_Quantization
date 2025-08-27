import math
import torch
import sys
import os
import transformers
import argparse
import ipdb
# Add the RTN_quantization directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
rtn_quantization_dir = os.path.join(parent_dir, 'RTN_quantization')
sys.path.append(rtn_quantization_dir)
from per_tensor_channel_group import (
    quantize_activation_per_token_absmax,
    quantize_activation_per_tensor_absmax,
    quantize_activation_per_group_absmax_token_dim,
    quantize_weight_per_channel_absmax,
    quantize_weight_per_tensor_absmax,
    quantize_weight_per_group_absmax_input_features,
)
from hadamard_utils import matmul_hadU_cuda, fast_hadamard_transform
from functools import partial 
import numpy as np





def parser_gen():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=123, help='Random Seed for HuggingFace and PyTorch')
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--bsz', type=int, default=32,
                        help='Batch-size for PPL evaluation (default:32)')
    parser.add_argument('--model_type', type=str, default="vit_l",)
    
    parser.add_argument('--hidden_size_image_en', type=int, default=1024)
    parser.add_argument('--hidden_size_mask_de', type=int, default=256,
                        help='Hidden Size for Mask Decoder (default: 256) [b,l,h]~[]')
    parser.add_argument('--num_attention_head_image_en', type = int, default=64,
                        help='Number of Attention Heads for Image Encoder (default: 8)')
    parser.add_argument('--num_attention_head_mask_de', type = int, default=32,
                        help='Number of Attention Heads for Mask Decoder (default: 8)')
    


    # Rotation Arguments
    parser.add_argument('--rotate', action=argparse.BooleanOptionalAction, default=True, 
                        help='''Rotate the moodel. This will include online rotation for down-projection and
                        out-projection. Note that this does not apply rotation to the K/Q and they will be rotated
                        if we want to quantize the Keys''')
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    parser.add_argument('--rotation_seed', type=int, default=-1,
                        help='Random Seed for generating random matrix!!')
    parser.add_argument('--fp32_had', action=argparse.BooleanOptionalAction, default=False,
                        help='Apply Hadamard rotation in FP32 (default: False)')
    parser.add_argument('--int8_down_proj', action=argparse.BooleanOptionalAction, default=False,
                        help='Use INT8 for Down Projection! If this set, both weights and activations of this layer will be in INT8')
    
    
    # # KV-Cache Quantization Arguments
    # parser.add_argument('--v_bits', type=int, default=16,
    #                     help='''Number of bits for V-cache quantization. 
    #                     Note that quantizing the V-cache does not need any other rotation''')
    # parser.add_argument('--v_groupsize', type=int, default=-1)
    # parser.add_argument('--v_asym', action=argparse.BooleanOptionalAction, default=False,
    #                     help='ASymmetric V-cache quantization')
    # parser.add_argument('--v_clip_ratio', type=float, default=1.0,
    #     help='Clip ratio for v-cache quantization. new_max = max * clip_ratio')
    
    # parser.add_argument('--k_bits', type=int, default=16,
    #                     help='''Number of bits for K-cache quantization. 
    #                     Note that quantizing the K-cache needs another rotation for the keys/queries''')
    # parser.add_argument('--k_groupsize', type=int, default=-1)
    # parser.add_argument('--k_asym', action=argparse.BooleanOptionalAction, default=False, 
    #                     help='ASymmetric K-cache quantization')
    # parser.add_argument('--k_pre_rope', action=argparse.BooleanOptionalAction, default=False, 
    #                     help='Pre-RoPE quantization for K-cache (not Supported yet!)')
    # parser.add_argument('--k_clip_ratio', type=float, default=1.0,
    #     help='Clip ratio for k-cache quantization. new_max = max * clip_ratio')

    
    # WandB Arguments
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)

    #Experiments Arguments
    args = parser.parse_args()
    
    # quant_type = f'w{args.w_bits}a{args.a_bits}_{args.rotate_mode}'
   
    return args

def hadamard_matrix(nh, dh):
    """
    Generates a Hadamard matrix of size (nh * dh) where nh and dh are powers of 2.

    Parameters:
    nh (int): Size of the Hadamard matrix for the first dimension (must be a power of 2).
    dh (int): Size of the Hadamard matrix for the second dimension (must be a power of 2).

    Returns:
    np.ndarray: Hadamard matrix of size (nh * dh, dh).
    """
    # Generate Hadamard matrix of size nh
    H_nh = np.array([[1]])
    while H_nh.shape[0] < nh:
        H_nh = np.block([[H_nh, H_nh], [H_nh, -H_nh]])

    I_dh = np.eye(dh)

    return np.kron(H_nh, I_dh)
def hadamard_matrix_transposed(nh, dh):
    """
    Generates the matrix I_{nh} X H_{dh} where nh and dh are powers of 2.

    Parameters:
    nh (int): Size of the identity matrix for the first dimension (must be a power of 2).
    dh (int): Size of the Hadamard matrix for the second dimension (must be a power of 2).

    Returns:
    np.ndarray: The matrix I_{nh} X H_{dh}.
    """
    # Generate Hadamard matrix of size dh
    H_dh = np.array([[1]])
    while H_dh.shape[0] < dh:
        H_dh = np.block([[H_dh, H_dh], [H_dh, -H_dh]])

    # Create identity matrix of size nh
    I_nh = np.eye(nh)

    # Return the matrix I_nh X H_dh
    return np.kron(I_nh, H_dh)
class ActQuantizer(torch.nn.Module):
    '''
        A class for quantizing the activations. We support per-token, per-tensor, 
        and per-group quantization for the activations.
    '''

    def __init__(self, act_quant="per_token", group_size=None, n_bits=8):
        super(ActQuantizer, self).__init__()
        self.bits = n_bits
        self.group_size = group_size
        
        # Set the activation quantization method
        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=self.bits)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=self.bits)
        elif act_quant == "per_group_token":
            self.act_quant_name = "per_group_token"
            assert group_size is not None, "group_size must be provided for per_group_token quantization"
            self.act_quant = partial(
                quantize_activation_per_group_absmax_token_dim, 
                group_size=self.group_size, 
                n_bits=self.bits
            )
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")


    def forward(self, x):
        if self.bits == 16:
            return x
        return self.act_quant(x)

    def quantize(self, x):
        # For compatibility - same as forward for these quantization methods
        return self.forward(x)

    def configure(self, bits, act_quant="per_token", group_size=None):
        """
        Reconfigure the quantizer with new parameters
        """
        self.bits = bits
        self.group_size = group_size
        
        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=self.bits)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=self.bits)
        elif act_quant == "per_group_token":
            self.act_quant_name = "per_group_token"
            assert group_size is not None, "group_size must be provided for per_group_token quantization"
            self.act_quant = partial(
                quantize_activation_per_group_absmax_token_dim, 
                group_size=self.group_size, 
                n_bits=self.bits
            )
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")


class ActQuantWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation quantization.
        We extract the FP features in the forward pass and quantize the rest using
        the specified quantization method.
        If a rotation Q is provided, the weight matrix will be rotated,
        a pre-forward hook will be registered to rotate the activation before quantization.
    '''

    def __init__(self, module: torch.nn.Linear, act_quant="per_token",weight_quant ="per_channel", n_bit =8, group_size=None):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.group_size = group_size
        self.n_bits = n_bit
        # Set the activation quantization method
        
        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = partial(quantize_activation_per_token_absmax, n_bits=self.n_bits)
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = partial(quantize_activation_per_tensor_absmax, n_bits=self.n_bits)
        elif act_quant == "per_group_token":
            self.act_quant_name = "per_group_token"
            self.act_quant = partial(quantize_activation_per_group_absmax_token_dim, group_size=self.group_size, n_bits=self.n_bits)
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False
        self.quantize_output = False
        self.quantize_input = False
        
        if weight_quant == "per_channel":
            self.weight = quantize_weight_per_channel_absmax(self.weight, n_bits=self.n_bits)
        elif weight_quant == "per_tensor":  
            self.weight = quantize_weight_per_tensor_absmax(self.weight, n_bits=self.n_bits)
        elif weight_quant == "per_group":
            self.weight = quantize_weight_per_group_absmax_input_features(self.weight, n_bits=self.n_bits, group_size=group_size)

    def extra_repr(self) -> str:
        return f"Activation Quantization: {self.act_quant_name}"

    def forward(self, x):
        x_dtype = x.dtype
        
        # Rotate, if needed
        if self.online_full_had:
            if self.fp32_had:  # Full Hadamard in FP32
                x = matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else:  # Full Hadamard in FP16
                x = matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            
            dims = len(init_shape)
        
            if dims == 4:  # 4D tensor [B, H, W, C] - image encoder
                x = x.reshape(init_shape[0], init_shape[1]*init_shape[2], init_shape[3])
            
            if self.K == 1:
                x = fast_hadamard_transform.hadamard_transform(
                    x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim).transpose(1, 2),
                    scale=1 / math.sqrt(init_shape[-1] // self.had_dim)
                ).transpose(1, 2)
                # Had_I = torch.tensor(hadamard_matrix(init_shape[-1] // self.had_dim, self.had_dim), device=x.device, dtype=x.dtype)
                # torch.set_printoptions(threshold=torch.inf)
                # x = x @ Had_I / math.sqrt(init_shape[-1] // self.had_dim)
            else:
                x = (self.had_K.to(x.dtype) @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)) / math.sqrt(
                    init_shape[-1] // self.had_dim
                )

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)
        if self.quantize_input :
            x = self.act_quant(x)

        # Pass through the wrapped module
        x = self.module(x).to(x_dtype)
        
        if self.quantize_output:
            x = self.act_quant(x)
        return x


def add_actquant(module, name='', layers=[torch.nn.Linear,
                                        ActQuantWrapper,
                                        transformers.models.falcon.modeling_falcon.FalconLinear],rtn_ro_config=None):
    if rtn_ro_config is None:
        if isinstance(module, ActQuantWrapper):
            return
        for attr in dir(module):
            tmp = getattr(module, attr)
            if type(tmp) in layers:
                setattr(module, attr, ActQuantWrapper(tmp))
            if type(tmp) == torch.nn.Sequential:
                replaced = []
                for i, child in enumerate(tmp.children()):
                    if type(child) in layers:
                        replaced.append(ActQuantWrapper(child))
                    else:
                        replaced.append(child)
                setattr(module, attr, torch.nn.Sequential(*replaced))
            if type(tmp) == torch.nn.ModuleList:
                replaced = []
                for i, child in enumerate(tmp.children()):
                    if type(child) in layers:
                        replaced.append(ActQuantWrapper(child))
                    else:
                        replaced.append(child)
                setattr(module, attr, torch.nn.ModuleList(replaced))
        for name1, child in module.named_children():
            add_actquant(child, name + '.' + name1 if name != '' else name1, layers,rtn_ro_config)
    else:
        if isinstance(module, ActQuantWrapper):
            return
        n_bits = rtn_ro_config.n_bits
        weight_quant = rtn_ro_config.weight_quant
        act_quant = rtn_ro_config.act_quant
        group_size = rtn_ro_config.group_size
        quantize_output = rtn_ro_config.quantize_output
        quantize_input  = rtn_ro_config.quantize_input
        for attr in dir(module):
            tmp = getattr(module, attr)
            if type(tmp) in layers:
                setattr(module, attr, ActQuantWrapper(
                    tmp, 
                    act_quant=act_quant,
                    weight_quant=weight_quant, 
                    n_bit=n_bits, 
                    group_size=group_size
                ))
                # Set quantize_output flag if specified
                if quantize_output:
                    getattr(module, attr).quantize_output = True
                if quantize_input:
                    getattr(module, attr).quantize_input = True
                    
            if type(tmp) == torch.nn.Sequential:
                replaced = []
                for i, child in enumerate(tmp.children()):
                    if type(child) in layers:
                        wrapper = ActQuantWrapper(
                            child,
                            act_quant=act_quant,
                            weight_quant=weight_quant,
                            n_bit=n_bits,
                            group_size=group_size
                        )
                        if quantize_output:
                            wrapper.quantize_output = True
                        if quantize_input:
                            wrapper.quantize_input = True
                        replaced.append(wrapper)
                    else:
                        replaced.append(child)
                setattr(module, attr, torch.nn.Sequential(*replaced))
                
            if type(tmp) == torch.nn.ModuleList:
                replaced = []
                for i, child in enumerate(tmp.children()):
                    if type(child) in layers:
                        wrapper = ActQuantWrapper(
                            child,
                            act_quant=act_quant,
                            weight_quant=weight_quant,
                            n_bit=n_bits,
                            group_size=group_size
                        )
                        if quantize_output:
                            wrapper.quantize_output = True
                        if quantize_input:
                            wrapper.quantize_input = True
                        replaced.append(wrapper)
                    else:
                        replaced.append(child)
                setattr(module, attr, torch.nn.ModuleList(replaced))
                
        for name1, child in module.named_children():
            add_actquant(child, name + '.' + name1 if name != '' else name1, layers, rtn_ro_config)
        
def find_qlayers(module, layers=[torch.nn.Linear,
                                ActQuantWrapper], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            print(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )
            # logging.info(
            #     f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
            #     f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            # )