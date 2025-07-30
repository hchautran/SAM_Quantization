import torch
import torch.nn as nn
from typing import Dict, Optional
import utils
import model_utils

def save_checkpoint_original_format(model, save_path: str, original_model):
    """Save rotated model checkpoint in original format."""
    print("Converting rotated model to original format...")
    
    current_state_dict = model.state_dict()
    original_state_dict = original_model.state_dict()
    converted_state_dict = {}
    
    # Get module mappings
    original_modules = dict(original_model.named_modules())
    current_modules = dict(model.named_modules())
    
    # Convert each original parameter
    for param_name in original_state_dict.keys():
        converted_value = convert_parameter(
            param_name, current_state_dict, original_modules, current_modules
        )
        if converted_value is not None:
            converted_state_dict[param_name] = converted_value
    
    torch.save(converted_state_dict, save_path)
    print(f"Checkpoint saved to {save_path}")
    print(f"Converted {len(converted_state_dict)}/{len(original_state_dict)} parameters")

def convert_parameter(param_name: str, current_state_dict: Dict, 
                     original_modules: Dict, current_modules: Dict) -> Optional[torch.Tensor]:
    """Convert parameter from rotated to original format."""
    
    module_path = '.'.join(param_name.split('.')[:-1])
    param_attr = param_name.split('.')[-1]
    
    original_module = original_modules.get(module_path)
    current_module = current_modules.get(module_path)
    
    if not original_module:
        return None
    
    # Case 1: Direct match (same type)
    if param_name in current_state_dict and type(original_module) == type(current_module):
        return current_state_dict[param_name]
    
    # Case 2: ActQuantWrapper -> Linear
    if isinstance(current_module, utils.ActQuantWrapper) and isinstance(original_module, nn.Linear):
        # Copy from current's .module to original's direct location
        # current: image_encoder.blocks.0.attn.qkv.module.weight
        # target: image_encoder.blocks.0.attn.qkv.weight
        actquant_param = f"{module_path}.module.{param_attr}"
        return current_state_dict.get(actquant_param)
    
    # Case 3: RMSN -> LayerNorm
    if isinstance(current_module, model_utils.RMSN) and isinstance(original_module, nn.LayerNorm):
        if param_attr == "weight":
            return torch.ones(current_module.mean_dim, 
                            dtype=current_state_dict[f"{module_path}.weight"].dtype,
                            device=current_state_dict[f"{module_path}.weight"].device)
        elif param_attr == "bias":
            return torch.zeros(current_module.mean_dim,
                             dtype=current_state_dict[f"{module_path}.weight"].dtype,
                             device=current_state_dict[f"{module_path}.weight"].device)
    
    # Case 4: Fallback
    return current_state_dict.get(param_name)