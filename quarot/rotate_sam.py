import utils
import torch
import model_utils
import transformers
import rotation_utils
import hadamard_utils
import checkpoint_utils
from segment_anything import sam_model_registry, SamPredictor
import copy
import os

def print_model_structure(model, title="Model Structure"):
    print(f"\n{title}")
    print("=" * len(title))
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")
    print("=" * len(title))

def main():
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sam_model_checkpoint = "/media/caduser/MyBook/chau/chi/SAM_Quantization/pretrained_checkpoint/sam_hq_vit_l.pth"
    sam_model_type = "vit_l"

    print(f"Loading SAM model: {sam_model_type}")
    sam_model = sam_model_registry[sam_model_type](checkpoint=sam_model_checkpoint)
    origin_model = copy.deepcopy(sam_model) 
    sam_model.to(device)
    sam_model.eval()
    
    
    

    # block0 = sam_model.image_encoder.blocks[0]
    # image_encoder_hidden_size = block0.attn.qkv.weight.shape[1]  # Input dimension
    # image_encoder_num_heads = block0.attn.num_heads
    # image_encoder_head_dim = image_encoder_hidden_size // image_encoder_num_heads
    # print(f"Image Encoder - Hidden Size: {image_encoder_hidden_size}, Num Heads: {image_encoder_num_heads}, Head Dim: {image_encoder_head_dim}")
    # layer0 = sam_model.mask_decoder.transformer.layers[0]
    # mask_decoder_hidden_size = layer0.self_attn.q_proj.weight.shape[1]
    # mask_decoder_num_heads = layer0.self_attn.num_heads
    # mask_decoder_head_dim = mask_decoder_hidden_size // mask_decoder_num_heads
    # print(f"Mask Decoder - Hidden Size: {mask_decoder_hidden_size}, Num Heads: {mask_decoder_num_heads}, Head Dim: {mask_decoder_head_dim}")

    
   
    if args.rotate:
        print("Starting model rotation...")
      
        # Use SAM-specific functions
        rotation_utils.fuse_layer_norms_sam(sam_model)
        rotation_utils.rotate_model(sam_model, args)  # Use SAM-specific rotation
        utils.cleanup_memory(verbos=True)
        
        # utils.add_actquant(sam_model) #Add Activation Wrapper to the model
        # qlayers = utils.find_qlayers(sam_model)
        # for name in qlayers:
        #     if "lin2" in name: #ffn layers
        #         if "mask_decoder" in name:  
        #             intermidiate_size = sam_model.mask_decoder.transformer.layers[0].mlp.lin2.weight.shape[1]
        #         elif "image_encoder" in name:
        #             intermidiate_size = sam_model.image_encoder.blocks[0].mlp.lin2.weight.shape[1]
        #         had_K, K = hadamard_utils.get_hadK(intermidiate_size)
        #         qlayers[name].online_full_had = True
        #         qlayers[name].had_K = had_K
        #         qlayers[name].K = K
        #         qlayers[name].fp32_had = args.fp32_had
        #     if 'mask_decoder' in name and "out_proj" in name:  # attention layer just decoder only as it include position embeddings
        #         had_K, K = hadamard_utils.get_hadK(args.num_attention_head_mask_de)
        #         qlayers[name].online_partial_had = True
        #         qlayers[name].had_K = had_K
        #         qlayers[name].K = K
        #         qlayers[name].had_dim = args.hidden_size_mask_de // args.num_attention_head_mask_de
        #         qlayers[name].fp32_had = args.fp32_had
                
                
        
        # if args.k_bits < 16:
        #     if args.k_pre_rope:
        #     raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        # else:
        rope_function_name = model_utils.get_rope_function_name(sam_model)
        
        # Only apply QK rotation wrapper if the model has RoPE function
        if rope_function_name is not None:
            layers = model_utils.get_layers(sam_model)
            k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                    "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                            layer.self_attn, 
                            rope_function_name, 
                            args,
                            **k_quant_config)
        else:
            print("SAM model doesn't use RoPE - skipping QK rotation wrapper")
        
        # Save checkpoint in original format
        save_path = "/media/caduser/MyBook/chau/chi/SAM_Quantization/pretrained_checkpoint/sam_hq_vit_l_rotation.pth"
        
        origin_model.to(device)
        checkpoint_utils.save_checkpoint_original_format(sam_model, save_path, original_model=origin_model)
        
    print_model_structure(sam_model, title="Model Structure After Rotation")  

if __name__ == '__main__':
    main()