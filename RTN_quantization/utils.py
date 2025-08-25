import torch
import torch.nn as nn
from Smooth import smooth_ln_fcs
from matplotlib import pyplot as plt


import os
import sys
import numpy as np
# Add the sam-hq directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up to SAM_Quantization
sam_hq_path = os.path.join(project_root, "sam-hq")

sys.path.insert(0, sam_hq_path)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment_anything import sam_model_registry, SamPredictor

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
                               target_class,n_bit, module_name_to_exclude, 
                               weight_quant="per_channel", act_quant="per_token", 
                               quantize_output=False, group_size=None):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not \
        any([x == name for x in module_name_to_exclude]):
            
            # Use from_float method instead of manual creation
            input_feature = child.in_features
            output_feature = child.out_features
            target_class.in_features= input_feature
            target_class.out_features = output_feature
            target_class.n_bits = n_bit
            new_module = target_class.from_float(
                child, 
                n_bits=n_bit,
                weight_quant=weight_quant, 
                act_quant=act_quant, 
                quantize_output=quantize_output,
                group_size=group_size
            )
            setattr(module, name, new_module) # replace the module with the new module
            
        else:
            # Recursively call the function for nested modules
            replace_linear_with_target_and_quantize(child, 
                     target_class,n_bit, module_name_to_exclude, 
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
def smooth_sam(model, act_scales, do_smooth_decoder=True, alpha=0.5, num_samples=512):
    """
    Apply SmoothQuant to SAM model focusing on image encoder and prompt encoder.
    
    Args:
        model: SAM model instance
        batched_input: List of input dictionaries for calibration
        alpha: Smoothing balance factor (0.0 to 1.0)
        num_samples: Number of samples for calibration
    """
    if hasattr(model, 'transformer'):
        print("applying smooth to the decoder")
        transformer = model.transformer
        for i, layer in enumerate(transformer.layers):  # Ensure this aligns correctly
            layer_name = f"mask_decoder.transformer.layers.{i}"
            mlp_name = f"{layer_name}.mlp.lin1"
            if mlp_name in act_scales:
                smooth_ln_fcs(layer.norm2, [layer.mlp.lin1], act_scales[mlp_name], alpha)
                print(f"Applied MLP smoothing (norm2 -> mlp.lin1) to {layer_name}")
    
    # Target image encoder blocks specifically (based on SAM structure)
    if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'blocks'):
        for i, block in enumerate(model.image_encoder.blocks):
            block_name = f"image_encoder.blocks.{i}"
            # Attention smoothing: norm1 -> attn.qkv
            if hasattr(block, 'norm1') and hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                qkv_name = f"{block_name}.attn.qkv"
                if qkv_name in act_scales:
                    qkv_input_scales = act_scales[qkv_name]
                    smooth_ln_fcs(block.norm1, [block.attn.qkv], qkv_input_scales, alpha)
                    print(f"Applied attention smoothing to {block_name}")

            if hasattr(block, 'norm2') and hasattr(block, 'mlp'):
                mlp_first_layer = None
                mlp_name = None
                for attr_name in ['fc1', 'lin1', 'dense', 'linear']:
                    if hasattr(block.mlp, attr_name):
                        mlp_first_layer = getattr(block.mlp, attr_name)
                        mlp_name = f"{block_name}.mlp.{attr_name}"
                        break      
                if mlp_first_layer is not None and mlp_name in act_scales:
                    mlp_input_scales = act_scales[mlp_name]
                    smooth_ln_fcs(block.norm2, [mlp_first_layer], mlp_input_scales, alpha)
                    print(f"Applied MLP smoothing to {block_name}")

    
    # Handle prompt encoder layers
    if hasattr(model, 'prompt_encoder'):
        for name, module in model.prompt_encoder.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"prompt_encoder.{name}"
                if full_name in act_scales:
                    parent_path = name.split('.')[:-1]
                    parent_module = model.prompt_encoder
                    for part in parent_path:
                        parent_module = getattr(parent_module, part)
                    for child_name, child_module in parent_module.named_children():
                        if isinstance(child_module, nn.LayerNorm):
                            linear_input_scales = act_scales[full_name]
                            smooth_ln_fcs(child_module, [module], linear_input_scales, alpha)
                            print(f"Applied prompt encoder smoothing: {full_name}")
                            break
                            
    # Handle mask decoder layers
    
    if do_smooth_decoder and  hasattr(model, 'mask_decoder'):
        print("Processing mask decoder layers...")
        
        # Handle transformer layers in mask decoder
        if hasattr(model.mask_decoder, 'transformer'):
            transformer = model.mask_decoder.transformer
        
            for i, layer in enumerate(transformer.layers):
                layer_name = f"mask_decoder.transformer.layers.{i}"
                ## TODO: Find the way to smooth residual layers  
                # norm1 -> cross_attn_token_to_image
                # if hasattr(layer, 'norm1') and hasattr(layer, 'cross_attn_token_to_image'):
                #     cross_q_name = f"{layer_name}.cross_attn_token_to_image.q_proj"
                #     if cross_q_name in act_scales:
                #         cross_layers = [
                #             layer.cross_attn_token_to_image.q_proj,
                #             layer.cross_attn_token_to_image.k_proj,
                #             layer.cross_attn_token_to_image.v_proj
                #         ]
                #         smooth_ln_fcs(layer.norm1, cross_layers, act_scales[cross_q_name], alpha)
                #         print(f"Applied smoothing (norm1 -> cross_attn_token_to_image) to {layer_name}")
                
                # norm2 -> mlp.lin1
                if hasattr(layer, 'norm2') and hasattr(layer, 'mlp') and hasattr(layer.mlp, 'lin1'):
                    mlp_name = f"{layer_name}.mlp.lin1"
                    if mlp_name in act_scales:
                        smooth_ln_fcs(layer.norm2, [layer.mlp.lin1], act_scales[mlp_name], alpha)
                        print(f"Applied MLP smoothing (norm2 -> mlp.lin1) to {layer_name}")
                
                # norm3 -> cross_attn_image_to_token
                # if hasattr(layer, 'norm3') and hasattr(layer, 'cross_attn_image_to_token'):
                #     img_to_token_q_name = f"{layer_name}.cross_attn_image_to_token.q_proj"
                #     if img_to_token_q_name in act_scales:
                #         img_to_token_layers = [
                #             layer.cross_attn_image_to_token.q_proj,
                #             layer.cross_attn_image_to_token.k_proj,
                #             layer.cross_attn_image_to_token.v_proj
                #         ]
                #         smooth_ln_fcs(layer.norm3, img_to_token_layers, act_scales[img_to_token_q_name], alpha)
                #         print(f"Applied smoothing (norm3 -> cross_attn_image_to_token) to {layer_name}")
                        
                # norm4 of final layer -> final_attn_token_to_image
                # Check if this is the last layer (i == len(transformer.layers) - 1)
                # if i == len(transformer.layers) - 1:
                #     if hasattr(layer, 'norm4') and hasattr(transformer, 'final_attn_token_to_image'):
                #         final_attn_name = "mask_decoder.transformer.final_attn_token_to_image.q_proj"
                #         if final_attn_name in act_scales:
                #             final_attn_layers = [
                #                 transformer.final_attn_token_to_image.q_proj,
                #                 transformer.final_attn_token_to_image.k_proj,
                #                 transformer.final_attn_token_to_image.v_proj
                #             ]
                #             smooth_ln_fcs(layer.norm4, final_attn_layers, act_scales[final_attn_name], alpha)
                #             print(f"Applied final attention smoothing (norm4 of final layer -> final_attn_token_to_image)")
                                    
    print("SmoothQuant smoothing completed for SAM model")
    return model


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point, input_label, plt.gca())
        
        print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.png',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis('off')
    plt.savefig(filename +'.png',bbox_inches='tight',pad_inches=-0.1)
    plt.close()
    

# SAM Model Testing Section
def sam_smoothing_test(sam_checkpoint, smoothed_sam_checkpoint, model_type, device):
    """Test SAM model smoothing and weight comparison"""
    
    sam_checkpoint = sam_checkpoint 
    print(sam_checkpoint)
    smoothed_sam_checkpoint = smoothed_sam_checkpoint 
    model_type = model_type
    device = device
    
    original_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    original_sam.to(device=device)
    original_sam.eval()
    print("Sam model loaded successfully!")
    
    smoothed_sam = sam_model_registry[model_type](checkpoint=smoothed_sam_checkpoint)  # Load base architecture    
    smoothed_sam.to(device=device)
    smoothed_sam.eval()
    print("Smoothed SAM model loaded successfully!")
    
    # # Load pre-calculated activation scales
    # print("Loading pre-calculated activation scales...")
    # act_scales = torch.load("sam_activation_scales.pt")
    # print(f"Loaded activation scales for {len(act_scales)} layers:")
    # for name, scales in act_scales.items():
    #     print(f"{name}: shape {scales.shape}, max {scales.max():.4f}, min {scales.min():.4f}")
    
    print("\n" + "="*80)
    print("WEIGHT COMPARISON: ORIGINAL vs SMOOTHED")
    print("="*80)
    
    for layer_idx in range(2):
        print(f"\n--- LAYER {layer_idx} ---")
        
    
        orig_block = original_sam.image_encoder.blocks[layer_idx]
        smoothed_block = smoothed_sam.image_encoder.blocks[layer_idx]
        

        print(f"\nNorm1 weights:")
        print(f"Original norm1 weight: {orig_block.norm1.weight[:5].detach().cpu()}")
        print(f"Smoothed norm1 weight: {smoothed_block.norm1.weight[:5].detach().cpu()}")
        print(f"Norm1 weight change: {torch.norm(smoothed_block.norm1.weight - orig_block.norm1.weight).item():.6f}")
        
        print(f"\nAttention QKV weights (first 5 input features, first 5 output features):")
        print(f"Original attn.qkv weight: {orig_block.attn.qkv.weight[:5, :5].detach().cpu()}")
        print(f"Smoothed attn.qkv weight: {smoothed_block.attn.qkv.weight[:5, :5].detach().cpu()}")
        print(f"QKV weight change: {torch.norm(smoothed_block.attn.qkv.weight - orig_block.attn.qkv.weight).item():.6f}")
        
        print(f"\nNorm2 weights:")
        print(f"Original norm2 weight: {orig_block.norm2.weight[:5].detach().cpu()}")
        print(f"Smoothed norm2 weight: {smoothed_block.norm2.weight[:5].detach().cpu()}")
        print(f"Norm2 weight change: {torch.norm(smoothed_block.norm2.weight - orig_block.norm2.weight).item():.6f}")
        
        mlp_first_layer = None
        mlp_attr_name = None
        for attr_name in ['fc1', 'lin1', 'dense', 'linear']:
            if hasattr(smoothed_block.mlp, attr_name):
                mlp_first_layer = getattr(smoothed_block.mlp, attr_name)
                mlp_attr_name = attr_name
                break
        
        if mlp_first_layer is None:
            for name, module in smoothed_block.mlp.named_children():
                if isinstance(module, nn.Linear):
                    mlp_first_layer = module
                    mlp_attr_name = name
                    break
        
        if mlp_first_layer is not None:
            orig_mlp_layer = getattr(orig_block.mlp, mlp_attr_name)
            print(f"\nMLP first layer ({mlp_attr_name}) weights (first 5x5):")
            print(f"Original mlp.{mlp_attr_name} weight: {orig_mlp_layer.weight[:5, :5].detach().cpu()}")
            print(f"Smoothed mlp.{mlp_attr_name} weight: {mlp_first_layer.weight[:5, :5].detach().cpu()}")
            print(f"MLP weight change: {torch.norm(mlp_first_layer.weight - orig_mlp_layer.weight).item():.6f}")
        else:
            print(f"\nCould not find MLP first layer in block {layer_idx}")
    
    # Add mask decoder transformer layers comparison
    print("\n" + "="*80)
    print("MASK DECODER TRANSFORMER LAYERS COMPARISON")
    print("="*80)
    
    if hasattr(original_sam, 'mask_decoder') and hasattr(original_sam.mask_decoder, 'transformer'):
        orig_transformer = original_sam.mask_decoder.transformer
        smoothed_transformer = smoothed_sam.mask_decoder.transformer
        
        if hasattr(orig_transformer, 'layers') and hasattr(smoothed_transformer, 'layers'):
            num_layers = min(len(orig_transformer.layers), 2)  # Compare first 2 layers
            
            for layer_idx in range(num_layers):
                print(f"\n--- TRANSFORMER LAYER {layer_idx} ---")
                
                orig_layer = orig_transformer.layers[layer_idx]
                smoothed_layer = smoothed_transformer.layers[layer_idx]
                
                # Compare norm2 weights
                if hasattr(orig_layer, 'norm2') and hasattr(smoothed_layer, 'norm2'):
                    print(f"\nNorm2 weights:")
                    print(f"Original norm2 weight: {orig_layer.norm2.weight[:5].detach().cpu()}")
                    print(f"Smoothed norm2 weight: {smoothed_layer.norm2.weight[:5].detach().cpu()}")
                    print(f"Norm2 weight change: {torch.norm(smoothed_layer.norm2.weight - orig_layer.norm2.weight).item():.6f}")
                
                # Compare mlp.lin1 weights
                if (hasattr(orig_layer, 'mlp') and hasattr(orig_layer.mlp, 'lin1') and 
                    hasattr(smoothed_layer, 'mlp') and hasattr(smoothed_layer.mlp, 'lin1')):
                    print(f"\nMLP lin1 weights (first 5x5):")
                    print(f"Original mlp.lin1 weight: {orig_layer.mlp.lin1.weight[:5, :5].detach().cpu()}")
                    print(f"Smoothed mlp.lin1 weight: {smoothed_layer.mlp.lin1.weight[:5, :5].detach().cpu()}")
                    print(f"MLP lin1 weight change: {torch.norm(smoothed_layer.mlp.lin1.weight - orig_layer.mlp.lin1.weight).item():.6f}")
                else:
                    print(f"\nMLP lin1 not found in transformer layer {layer_idx}")
        else:
            print("\nTransformer layers not found in mask decoder")
    else:
        print("\nMask decoder transformer not found")
    
    print("\n" + "="*80)

