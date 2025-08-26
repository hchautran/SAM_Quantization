import torch
import torch.nn as nn
import qgemm  # Use existing extension
from .int4_linear import Int4Linear



def replace_linear_with_int4(model, exclude_modules=None):
    """Replace nn.Linear with Int4Linear"""
    if exclude_modules is None:
        exclude_modules = []
    
    for name, module in model.named_children():
        if any(exclude in name for exclude in exclude_modules):
            continue
            
        if isinstance(module, nn.Linear):
            # Create replacement
            int4_layer = Int4Linear(
                module.in_features, 
                module.out_features, 
                bias=module.bias is not None
            )
            
            # Copy weights
            with torch.no_grad():
                int4_layer.quantize_weights(module.weight.to(torch.float16))
                if module.bias is not None:
                    int4_layer.bias.copy_(module.bias)
            
            setattr(model, name, int4_layer)
        else:
            replace_linear_with_int4(module, exclude_modules)

def quantize_sam_model(sam_model):
    """Quantize SAM with INT4"""
    exclude_modules = ["pos_embed", "cls_token", "patch_embed"]
    replace_linear_with_int4(sam_model, exclude_modules)
    return sam_model