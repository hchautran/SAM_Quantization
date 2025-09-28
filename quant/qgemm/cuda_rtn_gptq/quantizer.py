import torch
import torch.nn as nn
import qgemm  # Use existing extension
from int4_linear import Int4Linear
import os 
import ipdb


def replace_linear_with_int4(model,exclude_modules=None):
    """Replace nn.Linear with Int4Linear and quantize weights"""
    if exclude_modules is None:
        exclude_modules = []
    
    for name, module in model.named_children():
        if any(exclude in name for exclude in exclude_modules):
            continue
            
        if isinstance(module, nn.Linear):
            print(f"Replacing {name}: {module}")
            
            # Create Int4Linear replacement
            int4_layer = Int4Linear(
                module.in_features, 
                module.out_features, 
                bias=module.bias is not None
            )

            int4_layer = int4_layer.cuda()
            # Quantize weights
            with torch.no_grad():
                weight_fp16 = module.weight.to(torch.float16).cuda()
                int4_layer.quantize_weights(weight_fp16)
                
                if module.bias is not None:
                    int4_layer.bias.data = module.bias.to(torch.float16).cuda()
                    
            setattr(model, name, int4_layer)
            print(f"  -> Replaced with Int4Linear (moved to GPU)")
        else:
            replace_linear_with_int4(module, exclude_modules)
def save_cuda_quantized_model(model, save_dir="./quantized_models", model_name="sam_int4_full"):
   
    os.makedirs(save_dir, exist_ok=True)
    model_filename = f"{model_name}_.pth"
    model_path = os.path.join(save_dir, model_filename)
    
    # Save the entire model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model': model,  # Save the entire model architecture
    }, model_path)
    
def replace_linear_with_int4_gptq(model,quantizers,exclude_modules=None,parent_name=""):
    """Replace nn.Linear with Int4Linear and quantize weights"""
    if exclude_modules is None:
        exclude_modules = []
    
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        
        if name in exclude_modules:
            continue
        
        if isinstance(module, nn.Linear):
            print(full_name)
         
            if full_name not in quantizers.keys():
                continue

            # Create Int4Linear replacement
            int4_layer = Int4Linear(
                module.in_features, 
                module.out_features, 
                bias=module.bias is not None
            )
            scale= quantizers[full_name].scale
            weight_data = module.weight.data.to(torch.float16).cuda()
            scale = scale.to(torch.float16).cuda()
            
            int4_layer.scale = scale
            int4_layer.qweight = qgemm.sym_quant(weight_data, scale)
            if module.bias is not None:
                int4_layer.bias.data = module.bias.to(torch.float16).cuda()
            setattr(model, name, int4_layer)
            print(f"Replaced {full_name} with Int4Linear")
        else:
            replace_linear_with_int4_gptq(module,quantizers, exclude_modules,full_name)
            

        