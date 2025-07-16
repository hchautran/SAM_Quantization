import torch
import torch.nn as nn
from per_tensor_channel_group import W8A8Linear
from Smooth_quantization import smooth_ln_fcs, smooth_ln_fcs_llama_like
from calibration import get_act_scales_sam

# Import transformer decoder layers for smooth_lm function
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralRMSNorm,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralRMSNorm,
)
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer

def replace_linear_with_target_and_quantize(module, 
                               target_class, module_name_to_exclude, 
                               weight_quant="per_channel", act_quant="per_token", 
                               quantize_output=False, group_size=None):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
        any([x == name for x in module_name_to_exclude]):
            
            # Use from_float method instead of manual creation
            new_module = target_class.from_float(
                child, 
                weight_quant=weight_quant, 
                act_quant=act_quant, 
                quantize_output=quantize_output,
                group_size=group_size
            )
            setattr(module, name, new_module) # replace the module with the new module
            
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize(child, 
                     target_class, module_name_to_exclude, 
                     weight_quant, act_quant, quantize_output, group_size)


    

@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, FalconDecoderLayer):
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + ".self_attention.query_key_value"]
            fc1_input_scales = scales[name + ".mlp.dense_h_to_4h"]
            fc1 = module.mlp.dense_h_to_4h
    
            if (
                not module.config.new_decoder_architecture
                and module.config.parallel_attn
            ):
                attn_ln = module.input_layernorm
                smooth_ln_fcs(attn_ln, [qkv, fc1], qkv_input_scales, alpha)
            else:
                attn_ln = (
                    module.ln_attn
                    if module.config.new_decoder_architecture
                    else module.input_layernorm
                )
                ffn_ln = (
                    module.ln_mlp
                    if module.config.new_decoder_architecture
                    else module.post_attention_layernorm
                )
                smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)
                smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        elif isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer)):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.mlp.gate_proj, module.mlp.up_proj]
            fcs_input_scales = scales[name + ".mlp.gate_proj"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)
        elif isinstance(module, MixtralDecoderLayer):
            attn_ln = module.input_layernorm  # attention forward norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]

            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs_llama_like(attn_ln, qkv, qkv_input_scales, alpha)

            ffn_ln = module.post_attention_layernorm  # feed forward norm
            fcs = [module.block_sparse_moe.gate]
            for expert in module.block_sparse_moe.experts:
                fcs.append(expert.w1)
                fcs.append(expert.w3)
            fcs_input_scales = scales[name + ".block_sparse_moe.gate"]

            smooth_ln_fcs_llama_like(ffn_ln, fcs, fcs_input_scales, alpha)   

@torch.no_grad()
def smooth_sam(model, batched_input, alpha=0.5, num_samples=512):
    """
    Apply SmoothQuant to SAM model focusing on image encoder and prompt encoder.
    
    Args:
        model: SAM model instance
        batched_input: List of input dictionaries for calibration
        alpha: Smoothing balance factor (0.0 to 1.0)
        num_samples: Number of samples for calibration
    """
    # Get activation scales from calibration
    print("Collecting activation scales for SAM...")
    act_scales = get_act_scales_sam(model, batched_input, num_samples)
    
    print("Applying SmoothQuant smoothing to SAM...")
    
    # Target image encoder blocks specifically (based on SAM structure)
    if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'blocks'):
        for i, block in enumerate(model.image_encoder.blocks):
            block_name = f"image_encoder.blocks.{i}"
            try:
                # Attention smoothing: norm1 -> attn.qkv
                if hasattr(block, 'norm1') and hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                    qkv_name = f"{block_name}.attn.qkv"
                    if qkv_name in act_scales:
                        qkv_input_scales = act_scales[qkv_name]
                        smooth_ln_fcs(block.norm1, [block.attn.qkv], qkv_input_scales, alpha)
                
                # MLP smoothing: norm2 -> mlp (first layer in MLPBlock)
                if hasattr(block, 'norm2') and hasattr(block, 'mlp'):
                    # MLPBlock structure - need to find the first linear layer
                    mlp_first_layer = None
                    mlp_name = None
                    
                    # Look for common MLP layer names
                    for attr_name in ['fc1', 'lin1', 'dense', 'linear']:
                        if hasattr(block.mlp, attr_name):
                            mlp_first_layer = getattr(block.mlp, attr_name)
                            mlp_name = f"{block_name}.mlp.{attr_name}"
                            break
                    
                    # If no standard names found, get the first Linear layer
                    if mlp_first_layer is None:
                        for name, module in block.mlp.named_children():
                            if isinstance(module, nn.Linear):
                                mlp_first_layer = module
                                mlp_name = f"{block_name}.mlp.{name}"
                                break
                    
                    if mlp_first_layer is not None and mlp_name in act_scales:
                        mlp_input_scales = act_scales[mlp_name]
                        smooth_ln_fcs(block.norm2, [mlp_first_layer], mlp_input_scales, alpha)
                        print(f"Applied MLP smoothing to {block_name}")
                        
            except Exception as e:
                print(f"Error smoothing block {i}: {e}")
                continue
    
    # Handle prompt encoder layers
    if hasattr(model, 'prompt_encoder'):
        for name, module in model.prompt_encoder.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"prompt_encoder.{name}"
                if full_name in act_scales:
                    # Look for preceding LayerNorm in the same parent module
                    parent_path = name.split('.')[:-1]
                    parent_module = model.prompt_encoder
                    for part in parent_path:
                        parent_module = getattr(parent_module, part)
                    
                    # Find LayerNorm in the same parent
                    for child_name, child_module in parent_module.named_children():
                        if isinstance(child_module, nn.LayerNorm):
                            try:
                                linear_input_scales = act_scales[full_name]
                                smooth_ln_fcs(child_module, [module], linear_input_scales, alpha)
                                print(f"Applied prompt encoder smoothing: {full_name}")
                                break
                            except Exception as e:
                                print(f"Error smoothing prompt encoder {full_name}: {e}")
                                break
    
    print("SmoothQuant smoothing completed for SAM model")
    return model


@torch.no_grad()
def smooth_sam_targeted(model, batched_input, alpha=0.5, num_samples=512):
    """
    More targeted SAM smoothing focusing specifically on known ViT patterns.
    """
    # Get activation scales
    act_scales = get_act_scales_sam(model, batched_input, num_samples)
    
    print("Applying targeted SmoothQuant to SAM...")
    
    # Target image encoder blocks specifically
    if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'blocks'):
        for i, block in enumerate(model.image_encoder.blocks):
            block_name = f"image_encoder.blocks.{i}"
            
            try:
                # Attention smoothing
                if hasattr(block, 'norm1') and hasattr(block.attn, 'qkv'):
                    qkv_name = f"{block_name}.attn.qkv"
                    if qkv_name in act_scales:
                        smooth_ln_fcs(block.norm1, [block.attn.qkv], act_scales[qkv_name], alpha)
                        print(f"Smoothed attention in block {i}")
                
                # MLP smoothing
                if hasattr(block, 'norm2') and hasattr(block.mlp, 'lin1'):
                    mlp_name = f"{block_name}.mlp.lin1"
                    if mlp_name in act_scales:
                        smooth_ln_fcs(block.norm2, [block.mlp.lin1], act_scales[mlp_name], alpha)
                        print(f"Smoothed MLP in block {i}")
                        
            except Exception as e:
                print(f"Error smoothing block {i}: {e}")
                continue
    
    # Target prompt encoder if it has LayerNorm + Linear patterns
    if hasattr(model, 'prompt_encoder'):
        for name, module in model.prompt_encoder.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"prompt_encoder.{name}"
                if full_name in act_scales:
                    # Try to find a preceding LayerNorm
                    # This is more heuristic and might need adjustment based on actual SAM structure
                    try:
                        print(f"Found prompt encoder linear layer: {full_name}")
                    except Exception as e:
                        continue
    
    return model




####################################### example usage #################################################
class DummyModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.emb = torch.nn.Embedding(1, 1)
    # Try with bias
    self.linear_1 = nn.Linear(1, 1)
    # Try without bias
    self.linear_2 = nn.Linear(1, 1, bias=False)
    # Lm prediction head
    self.lm_head = nn.Linear(1, 1, bias=False)

  def forward(self, x):
    x = self.emb(x)
    x = self.linear_1(x)
    x = self.linear_2(x)
    x = self.lm_head(x)
    return x

if __name__ == "__main__":
    model = DummyModel()  # Fixed: Added parentheses to instantiate
    print("Before quantization:")
    print(model)
    print("\nLinear layers before:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  {name}: {type(module)}")
    
    replace_linear_with_target_and_quantize(
        model, 
        W8A8Linear, 
        module_name_to_exclude=["emb", "lm_head"],
        weight_quant="per_channel",
        act_quant="per_token"
    )
    
    print("\nAfter quantization:")
    print(model)
    print("\nLinear layers after:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, W8A8Linear)):
            print(f"  {name}: {type(module)}")
            
    # Test forward pass
    test_input = torch.randint(0, 1, (1,))
    output = model(test_input)
    print(f"\nTest forward pass successful: {output.shape}")