import math
import torch
from per_tensor_channel_group import (
    quantize_activation_per_token_absmax,
    quantize_activation_per_tensor_absmax,
    quantize_activation_per_group_absmax_token_dim,
)
from hadamard_utils import matmul_hadU_cuda, fast_hadamard_transform





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
def find_params_per_token_groupwise(self, x):
    """
    Find quantization parameters for token-wise group quantization.
    
    This method is deprecated as quantization is now performed directly in the forward pass.
    
    Parameters:
        x (torch.Tensor): The input tensor to find quantization parameters for.
        
    Returns:
        None: This method is a placeholder and does not return any value.
    """
    # No longer needed - quantization is done directly in forward pass
    pass

def find_params(self, x):
    """
    Find quantization parameters for the input tensor.
    
    This method is deprecated as quantization is now performed directly in the forward pass.
    
    Parameters:
        x (torch.Tensor): The input tensor to find quantization parameters for.
        
    Returns:
        None: This method is a placeholder and does not return any value.
    """
    # No longer needed - quantization is done directly in forward pass
    pass

    def __repr__(self):
        return f"ActQuantizer(bits={self.bits}, method={self.act_quant_name}, group_size={self.group_size})"




class ActQuantWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation quantization.
        We extract the FP features in the forward pass and quantize the rest using
        the specified quantization method.
        If a rotation Q is provided, the weight matrix will be rotated,
        a pre-forward hook will be registered to rotate the activation before quantization.
    '''

    def __init__(self, module: torch.nn.Linear, act_quant="per_token", group_size=None):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.group_size = group_size

        # Set the activation quantization method
        if act_quant == "per_token":
            self.act_quant_name = "per_token"
            self.act_quant = quantize_activation_per_token_absmax
        elif act_quant == "per_tensor":
            self.act_quant_name = "per_tensor"
            self.act_quant = quantize_activation_per_tensor_absmax
        elif act_quant == "per_group_token":
            self.act_quant_name = "per_group_token"
            self.act_quant = lambda x: quantize_activation_per_group_absmax_token_dim(
                x, group_size=self.group_size
            )
        else:
            raise ValueError(f"Invalid act_quant: {act_quant}")

        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False
        self.quatize_output = True  
        

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
            if self.K == 1:
                x = fast_hadamard_transform.hadamard_transform(
                    x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim).transpose(1, 2),
                    scale=1 / math.sqrt(init_shape[-1] // self.had_dim)
                ).transpose(1, 2)
            else:
                x = (self.had_K.to(x.dtype) @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)) / math.sqrt(
                    init_shape[-1] // self.had_dim
                )

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        # Quantize the input
        x = self.act_quant(x)

        # Pass through the wrapped module
        x = self.module(x).to(x_dtype)
        
        if self.quatize_output:
            x = self.act_quant(x)
            
        return x
    
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
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )