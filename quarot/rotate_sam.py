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
import cv2
import ipdb
import numpy as np
import matplotlib.pyplot as plt
def print_model_structure(model, title="Model Structure"):
    print(f"\n{title}")
    print("=" * len(title))
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")
    print("=" * len(title))
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
def main():
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)
    
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sam_model_checkpoint = "/media/caduser/MyBook/chau/chi/SAM_Quantization/pretrained_checkpoint/sam_hq_vit_l.pth"
    sam_model_type = args.model_type

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
        Q_image_encoder = rotation_utils.get_orthogonal_matrix(args.hidden_size_image_en,args.rotate_mode,device = args.device,seed=args.seed)
        Q_mask_decoder = rotation_utils.get_orthogonal_matrix(args.hidden_size_mask_de,args.rotate_mode,device = args.device,seed=args.seed+1)
        # Use SAM-specific functions
        rotation_utils.fuse_layer_norms_sam(sam_model)
        
        rotation_utils.rotate_model(sam_model,Q_image_encoder, Q_mask_decoder , args)  # Use SAM-specific rotation
        utils.cleanup_memory(verbos=True)
        
        utils.add_actquant(sam_model) #Add Activation Wrapper to the model
        qlayers = utils.find_qlayers(sam_model)
        for name in qlayers:
            
            if ("lin2" in name) and ("image_encoder" in name): #ffn layers
                if "mask_decoder" in name:  
                    intermidiate_size = sam_model.mask_decoder.transformer.layers[0].mlp.lin2.weight.shape[1]
                elif "image_encoder" in name:
                    intermidiate_size = sam_model.image_encoder.blocks[0].mlp.lin2.weight.shape[1]
                had_K, K = hadamard_utils.get_hadK(intermidiate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
            if 'image_encoder' in name and "proj"  in name:  # attention layer just decoder only as it include position embeddings
                had_K, K = hadamard_utils.get_hadK(args.num_attention_head_mask_de)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = args.hidden_size_image_en // args.num_attention_head_image_en
                qlayers[name].fp32_had = args.fp32_had
                
                
        
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
        
        
    print_model_structure(sam_model, title="Model Structure After Rotation")    
    # exit()
    sam_model.to(device)
    predictor = SamPredictor(sam_model)    
    # ipdb.set_trace()
    for i in range(8):
        print("image:   ",i)
        # hq_token_only: False means use hq output to correct SAM output. 
        #                True means use hq output only. 
        #                Default: False
        hq_token_only = False 
        # To achieve best visualization effect, for images contain multiple objects (like typical coco images), we suggest to set hq_token_only=False
        # For images contain single object, we suggest to set hq_token_only = True
        # For quantiative evaluation on COCO/YTVOS/DAVIS/UVO/LVIS etc., we set hq_token_only = False

        image = cv2.imread('/media/caduser/MyBook/chau/chi/SAM_Quantization/sam-hq/demo/input_imgs/example'+str(i)+'.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        if i==0:
            input_box = np.array([[4,13,1007,1023]])
            input_point, input_label = None, None
        elif i==1:
            input_box = np.array([[306, 132, 925, 893]])
            input_point, input_label = None, None
            hq_token_only = True
        elif i==2:
            input_point = np.array([[495,518],[217,140]])
            input_label = np.ones(input_point.shape[0])
            input_box = None
            hq_token_only = True
        elif i==3:
            input_point = np.array([[221,482],[498,633],[750,379]])
            input_label = np.ones(input_point.shape[0])
            input_box = None
        elif i==4:
            input_box = np.array([[64,76,940,919]])
            input_point, input_label = None, None
            hq_token_only = True
        elif i==5:
            input_point = np.array([[373,363], [452, 575]])
            input_label = np.ones(input_point.shape[0])
            input_box = None
        elif i==6:
            input_box = np.array([[181, 196, 757, 495]])
            input_point, input_label = None, None
        elif i==7:
            # multi box input
            input_box = torch.tensor([[45,260,515,470], [310,228,424,296]],device=predictor.device)
            transformed_box = predictor.transform.apply_boxes_torch(input_box, image.shape[:2])
            input_point, input_label = None, None

        batch_box = False if input_box is None else len(input_box)>1 
        result_path = '/media/caduser/MyBook/chau/chi/SAM_Quantization/demo/hq_sam_result_rotated_hadamard/'
        os.makedirs(result_path, exist_ok=True)

        if not batch_box: 
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box = input_box,
                multimask_output=False,
                hq_token_only=hq_token_only, 
            )
            show_res(masks,scores,input_point, input_label, input_box, result_path + 'example'+str(i), image)
        # ipdb.set_trace()
        else:
            masks, scores, logits = predictor.predict_torch(
                point_coords=input_point,
                point_labels=input_label,
                boxes=transformed_box,
                multimask_output=False,
                hq_token_only=hq_token_only,
            )
            masks = masks.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()
            input_box = input_box.cpu().numpy()
            show_res_multi(masks, scores, input_point, input_label, input_box, result_path + 'example'+str(i), image)
   
        #compare logits
        
        # Save checkpoint in original format
        # save_path = "/media/caduser/MyBook/chau/chi/SAM_Quantization/pretrained_checkpoint/sam_hq_vit_l_rotation.pth"
        
        # origin_model.to(device)
        # checkpoint_utils.save_checkpoint_original_format(sam_model, save_path, original_model=origin_model)
        
    # print_model_structure(sam_model, title="Model Structure After Rotation")  

if __name__ == '__main__':
    main()