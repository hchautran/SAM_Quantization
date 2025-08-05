
import torch
from segment_anything.modeling.image_encoder import Attention as OriginalAttention
from segment_anything.modeling.image_encoder import Block as OriginalBlock
from typing import Optional, Tuple, Type
import torch.nn.functional as F
import ipdb

def get_rope_function_name(model):
    # SAM models don't use rotary position embeddings
    # They use standard attention with positional encodings added to inputs
    # Return None to indicate no RoPE function exists
    return None
def window_partition(x: torch.Tensor, window_size: int) :
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]) :
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x
def get_layers(model):
    return model.mask_decoder.transformer.layers
def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    # Collect modules to replace
    modules_to_replace = []

    for name, module in root.named_modules():
        
        # if ("mask_decoder" in name and "norm2" in name) or ("image_encoder" in name and "norm" in name):
        if "image_encoder" in name and "norm" in name :

            if isinstance(module, type_to_replace):
                modules_to_replace.append((name, module))

    # Perform replacements
    for name, module in modules_to_replace:
        new_module = None
        if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
            new_module = new_module_factory(module, int(name))
        else:  # layernorm_fusion.fuse_modules case where layernorms are fused
            new_module = new_module_factory(module)
        if new_module is not None:
            # Use `setattr` to replace the module
            parent_module, attr_name = _get_parent_module_and_attr(root, name)
            setattr(parent_module, attr_name, new_module)


def _get_parent_module_and_attr(root: torch.nn.Module, full_name: str):
    """Helper function to get the parent module and attribute name for a given module."""
    parts = full_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)
    

class CustomAttention(OriginalAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = None
        
    def _take_Q(self, Q: Optional[torch.Tensor] = None):
        if Q is not None:
            self.Q = Q
            
    def forward(self, x: torch.Tensor) :
        
        # Apply Q matrix multiplication if provided
        if self.Q is not None:
            B, H, W, C = x.shape
            self.Q = self.Q.to( dtype=x.dtype)
            
            x_flat = x.reshape(B, H*W, C)
            x_rotated = torch.matmul(x_flat, self.Q)
            x = x_rotated.reshape(B, H, W, C)
        
        B, H, W, _ = x.shape
        
        # qkv with shape (3, B, nHead, H * W, C)
        
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # ipdb.set_trace()
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x
class CustomBlock(OriginalBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q= None
    def _take_Q(self, Q: Optional[torch.Tensor] = None):
        if Q is not None:
            self.Q = Q
    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        
        
        x = (shortcut.double() @ self.Q).to(torch.float32) + x        
        # x = (( x.double() @ self.Q.T)@self.Q).to(torch.float32) + self.mlp(self.norm2(x)) --> this line is equivalent to the next line
        x = x + self.mlp(self.norm2(x))
        return x
class CustomBlock_3(OriginalBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q= None
    def _take_Q(self, Q: Optional[torch.Tensor] = None):
        if Q is not None:
            self.Q = Q
    def forward(self, x: torch.Tensor):
        
        shortcut = (x.double()@self.Q.T).to(torch.float32)
        shortcut_ = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
            
        # x= shortcut + x
        # x = x+self.mlp(self.norm2(x))

        # x = (shortcut.double() @ self.Q).to(torch.float32) + x # equivalent to the next line
        x = shortcut_ + x
        x =( x.double() @ self.Q.T ).to(torch.float32)+ self.mlp(self.norm2(x))
        
        return x  
class CustomBlock_2(OriginalBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q= None
    def _take_Q(self, Q: Optional[torch.Tensor] = None):
        if Q is not None:
            self.Q = Q
    def forward(self, x: torch.Tensor):
        
        shortcut = (x.double()@self.Q.T).to(torch.float32)
        # shortcut_ = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        
        x = (shortcut.double() @ self.Q).to(torch.float32) + x   
        # x = shortcut_ +x     
        # x = (( x.double() @ self.Q.T)@self.Q).to(torch.float32) + self.mlp(self.norm2(x)) # this line is equivalent to the next line
        x = x + self.mlp(self.norm2(x))
        return x  
