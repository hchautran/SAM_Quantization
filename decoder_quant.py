import torch
import torch.nn.functional as F
from segment_anything.modeling.transformer import TwoWayAttentionBlock, TwoWayTransformer, Attention
from quant.utils.observer import ObserverBase 
import torch.nn as nn
from collections import defaultdict
from typing import Type, Tuple, Optional
import math
from segment_anything import SamPredictor
import numpy as np
import numpy as np
import cv2
import matplotlib.pyplot  as plt
from typing import Optional
import pandas as pd 

def to_numpy(x:torch.Tensor):
    return x.detach().cpu().numpy()

@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.contiguous().view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t
@torch.no_grad()
def quantize_specific_heads(t, head_indices, num_heads, n_bits=8):
    """
    Quantize specific attention heads using per-token absmax quantization.

    Args:
        t: Input tensor of shape (Bz, T, C) where C = num_heads * head_dim
        head_indices: List or tensor of head indices to quantize
        head_dim: Dimension of each attention head
        n_bits: Number of bits for quantization

    Returns:
        Quantized tensor with same shape as input
    """
    Bz, T, C = t.shape
    head_dim = C // num_heads

    # Reshape to separate heads: (Bz, T, num_heads, head_dim)
    t_heads = t.view(Bz, T, num_heads, head_dim)

    # Convert head_indices to tensor if it's a list
    if isinstance(head_indices, list):
        head_indices = torch.tensor(head_indices, device=t.device)

    # Quantize specified heads
    for head_idx in head_indices:
        if head_idx < num_heads:
            # Extract specific head: (Bz, T, head_dim)
            head_data = t_heads[:, :, head_idx, :]

            # Reshape for quantization: (Bz * T, head_dim)
            head_data_2d = head_data.contiguous().view(-1, head_dim)

            # Apply per-token absmax quantization
            scales = head_data_2d.abs().max(dim=-1, keepdim=True)[0]
            q_max = 2 ** (n_bits - 1) - 1
            scales.clamp_(min=1e-5).div_(q_max)
            head_data_2d.div_(scales).round_().mul_(scales)

            # Reshape back and update the original tensor
            t_heads[:, :, head_idx, :] = head_data_2d.view(Bz, T, head_dim)

    # Reshape back to original shape
    return t_heads.view(Bz, T, C)
    
class TwoWayTransformerObserver(TwoWayTransformer):
    attention_score = defaultdict(list) 
    
    
    def forward(self,
        image_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        point_embedding: torch.Tensor
    ):
        bs, c, h, w = image_embedding.shape 
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding
        TwoWayTransformerObserver.attention_score['pre_p'] = queries
        TwoWayTransformerObserver.attention_score['pre_i'] = keys

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            
            queries, keys, p2p_attn,  p2p_q, p2p_k, p2p_v ,p2i_attn, p2i_q, p2i_k, p2i_v ,i2p_attn,  i2p_q, i2p_k, i2p_v = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

            TwoWayTransformerObserver.attention_score['p2p_q'].append(p2p_q)
            TwoWayTransformerObserver.attention_score['p2p_k'].append(p2p_k)
            TwoWayTransformerObserver.attention_score['p2p_v'].append(p2p_v)

            TwoWayTransformerObserver.attention_score['i2p_q'].append(i2p_q)
            TwoWayTransformerObserver.attention_score['i2p_k'].append(i2p_k)
            TwoWayTransformerObserver.attention_score['i2p_v'].append(i2p_v)

            TwoWayTransformerObserver.attention_score['p2i_q'].append(p2i_q)
            TwoWayTransformerObserver.attention_score['p2i_k'].append(p2i_k)
            TwoWayTransformerObserver.attention_score['p2i_v'].append(p2i_v)


            TwoWayTransformerObserver.attention_score['p2p_attn'].append(p2p_attn)
            TwoWayTransformerObserver.attention_score['i2p_attn'].append(i2p_attn)
            TwoWayTransformerObserver.attention_score['p2i_attn'].append(p2i_attn)

        # Apply the final attenion layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out, final_attn, final_q, final_k, final_v = self.final_attn_token_to_image(q=q, k=k, v=keys)
        # TwoWayTransformerObserver.attention_score['final_attn'] = final_attn
        # TwoWayTransformerObserver.attention_score['final_q'] = final_q
        # TwoWayTransformerObserver.attention_score['final_k'] = final_k
        # TwoWayTransformerObserver.attention_score['final_v'] = final_v
        TwoWayTransformerObserver.attention_score['final_attn'].append(final_attn)
        TwoWayTransformerObserver.attention_score['final_q'].append(final_q)
        TwoWayTransformerObserver.attention_score['final_k'].append(final_k)
        TwoWayTransformerObserver.attention_score['final_v'].append(final_v)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys
    
    def clear_dict():
        TwoWayTransformerObserver.attention_score = defaultdict(list)


class TwoWayAttentionBlockObserver(TwoWayAttentionBlock):
    attention_dict={}

    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        query_pe: torch.Tensor, 
        key_pe: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries, p2p_attn, p2p_q, p2p_k, p2p_v = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out, p2p_attn, p2p_q, p2p_k, p2p_v = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out, p2i_attn, p2i_q, p2i_k, p2i_v  = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out, i2p_attn, i2p_q, i2p_k, i2p_v  = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys, p2p_attn,  p2p_q, p2p_k, p2p_v ,p2i_attn, p2i_q, p2i_k, p2i_v ,i2p_attn,  i2p_q, i2p_k, i2p_v


class AttentionObserver(Attention):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out, attn, q, k, v
list_he=[0,1,2,3,4,5,6,7] # 5 7
str_list="_"
for i in range(len(list_he)):
    str_list += str(list_he[i])+"_"
class AttentionObserver_q(Attention):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        print("chiiiiiiiiiii",self.num_heads)
        
        q= quantize_specific_heads(q, head_indices=list_he, num_heads=self.num_heads, n_bits=2)
        k= quantize_specific_heads(k, head_indices=list_he, num_heads=self.num_heads, n_bits=2)
        # q=quantize_activation_per_token_absmax(q, n_bits=2)
        # k=quantize_activation_per_token_absmax(k, n_bits=2)
        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        # import ipdb; ipdb.set_trace()
        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out, attn, q, k, v
def mask_decoder_monkey_patch(model):
    for name, module in model.named_modules():
        if isinstance(module, Attention) and "self_attn" not in name:
            module.__class__ = AttentionObserver
        if isinstance(module, Attention) and "self_attn" in name:
            module.__class__ = AttentionObserver_q
        # if isinstance(module, Attention) :
        #     module.__class__ = AttentionObserver_q
        if isinstance(module, TwoWayAttentionBlock):
            module.__class__ = TwoWayAttentionBlockObserver
        if isinstance(module, TwoWayTransformer):
            module.__class__ = TwoWayTransformerObserver 



def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    
def show_mask_image(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def inference_with_sam_model(
    sam_model,
    image: np.ndarray,
    input_point: Optional[np.ndarray] = None,
    input_label: Optional[np.ndarray] = None,
    input_box: Optional[np.ndarray] = None,
    hq_token_only: bool = False
):
    """
    Run inference directly with Sam model (not SamPredictor wrapper)
    """
    # Make sure the entire model is on a single device
    device = next(sam_model.parameters()).device
    sam_model = sam_model.to(device)
    
    # Prepare image tensor
    input_image = torch.as_tensor(image).to(device).permute(2, 0, 1).contiguous()
    original_size = image.shape[:2]
    
    # Prepare batched input for Sam model
    batched_input = []
    dict_input = {
        'image': input_image,
        'original_size': original_size
    }
    
    # Add prompts if provided
    if input_point is not None and input_label is not None:
        point_coords = torch.as_tensor(input_point).to(device)
        point_labels = torch.as_tensor(input_label).to(device)
        dict_input['point_coords'] = point_coords
        dict_input['point_labels'] = point_labels
        
    if input_box is not None:
        boxes = torch.as_tensor(input_box).to(device)
        dict_input['boxes'] = boxes
        
    batched_input.append(dict_input)
    
    # Make sure the model is in eval mode
    sam_model.eval()
    
    # Force all model parameters to correct device
    for module in sam_model.modules():
        for param in module.parameters(recurse=False):
            param.data = param.data.to(device)
        for buffer in module.buffers(recurse=False):
            buffer.data = buffer.data.to(device)

    with torch.no_grad():
        outputs = sam_model(batched_input, multimask_output=False)
        if isinstance(outputs, tuple):
            outputs, interm_embeddings = outputs
        else:
            interm_embeddings = None
            

    
    # Extract results from outputs
    if len(outputs) > 0:
        output = outputs[0]
        masks = output['masks'].detach().cpu().numpy()
        scores = output['iou_predictions'].detach().cpu().numpy()
        logits = output['low_res_logits'].detach().cpu().numpy()
    else:
        # Fallback if no outputs
        h, w = original_size
        masks = np.zeros((1, h, w), dtype=bool)
        scores = np.array([0.0])
        logits = np.zeros((1, 256, 256))
        
    return masks, scores, logits

@torch.inference_mode()
def inference_image(
    predictor,
    image_dir: str = './input_imgs/example1.png',
    show_image: bool = False,
    example_idx: int = 1,  # Which example configuration to use
):
    """
    Run inference on a single image using either SamPredictor or Sam model directly
    """
    image = cv2.imread(f'{image_dir}/example{example_idx}.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Configure based on example index
    if example_idx == 0:
        input_box = np.array([[4, 13, 1007, 1023]])
        input_point, input_label = None, None
        hq_token_only = True
    elif example_idx == 1:
        input_box = np.array([[306, 132, 925, 893]])
        input_point, input_label = None, None
        hq_token_only = True
    elif example_idx == 2:
        input_point = np.array([[495, 518], [217, 140]])
        input_label = np.ones(input_point.shape[0])
        input_box = None
        hq_token_only = True
    elif example_idx == 3:
        random_points = generate_random_points(5)
        input_point = np.array([[221, 482], [498, 633], [750, 379]])
        all_points = np.concatenate([input_point, random_points])
        
        input_label = np.ones(input_point.shape[0])
        new_labels = np.ones(random_points.shape[0])
        all_labels = np.concatenate([input_label, new_labels])
        input_label = all_labels
        input_point = all_points
        input_box = None
        hq_token_only = False
    elif example_idx == 4:
        input_box = np.array([[64, 76, 940, 919]])
        input_point, input_label = None, None
        hq_token_only = True
    else:
        # Default fallback
        input_box = np.array([[306, 132, 925, 893]])
        input_point, input_label = None, None
        hq_token_only = True
    
    # Check if predictor is SamPredictor or Sam model
    if isinstance(predictor, SamPredictor):
        # Use existing SamPredictor logic
        predictor.set_image(image)
        
        try:
            # Try to predict with hq_token_only parameter
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=False,
                hq_token_only=hq_token_only,
            )
        except TypeError as e:
            if "hq_token_only" in str(e):
                # Fall back to standard prediction without hq_token_only
                print("Warning: hq_token_only parameter not supported, using standard prediction")
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=input_box,
                    multimask_output=False,
                )
            else:
                raise e
                
    else: #Sam is not None and isinstance(predictor, Sam):
        # Use direct Sam model inference
        masks, scores, logits = inference_with_sam_model(
            sam_model=predictor,
            image=image,
            input_point=input_point,
            input_label=input_label,
            input_box=input_box,
            hq_token_only=hq_token_only
        )
    # else:
    #     raise ValueError(f"Unsupported predictor type: {type(predictor)}")
    
    if show_image:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        if len(masks) > 0:
            show_mask_image(masks[0], plt.gca(), random_color=False)

        if input_box is not None:
            box = input_box[0]
            x0, y0 = box[0], box[1]
            w, h = box[2] - box[0], box[3] - box[1]
            plt.gca().add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

        if input_point is not None and input_label is not None:
            show_points(input_point, input_label, plt.gca())

        output_path = "/home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/decoder_quant_r"
        import os
        os.makedirs(output_path, exist_ok=True)

        # Handle non-quantized model
        if str_list == "_":
            # Save mask (no quant)
            output_filename = os.path.join(output_path, f'example_{example_idx}_no_quant.png')
            # Save mask information to a file
            if len(masks) > 0:
                mask_info_file = os.path.join(output_path, f'example_{example_idx}_mask_info.npy')
                np.save(mask_info_file, masks[0])
                print(f"Saved non-quantized mask information to: {mask_info_file}")

            plt.title(f'Example {example_idx} - No Quantization - Score: {scores[0]:.3f}')

        # Handle quantized model
        else:
            # Try to load the non-quantized mask from the saved file
            non_quant_mask = None
            mask_info_file = os.path.join(output_path, f'example_{example_idx}_mask_info.npy')

            if os.path.exists(mask_info_file):

                non_quant_mask = np.load(mask_info_file)
                print(f"Loaded non-quantized mask from: {mask_info_file}")

                if non_quant_mask.dtype == bool or masks[0].dtype == bool:
                    # For boolean masks, convert to float (0 and 1) before calculating MSE
                    non_quant_float = non_quant_mask.astype(float)
                    mask_float = masks[0].astype(float)
                    mse = np.mean((non_quant_float - mask_float) ** 2)
                else:
                    # For non-boolean masks, calculate MSE directly
                    mse = np.mean((non_quant_mask - masks[0]) ** 2)

                # Calculate IoU (Intersection over Union)
                intersection = np.logical_and(non_quant_mask, masks[0])
                union = np.logical_or(non_quant_mask, masks[0])
                iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

                title = f'Example {example_idx} - Quant: {str_list} - Score: {scores[0]:.3f} - MSE: {mse:.4f} - IoU: {iou:.4f}'

            else:
                print(f"No non-quantized mask found at: {mask_info_file}")
                title = f'Example {example_idx} - Quant: {str_list} - Score: {scores[0]:.3f} - No reference mask'

            output_filename = os.path.join(output_path, f'example_{example_idx}_quant_{str_list}_self_attn_2.png')
            plt.title(title)

        plt.axis('off')
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        print(f"Image saved to: {output_filename}")

    return masks, scores, logits

def generate_random_points(num_points, image_width=1024, image_height=1024, seed=None):
    """
    Generate random points within the specified image dimensions.

    Args:
        num_points (int): Number of random points to generate
        image_width (int, optional): Width of the image. Defaults to 1024.
        image_height (int, optional): Height of the image. Defaults to 1024.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray: Array of shape (num_points, 2) containing random points
                   where each point is [x, y] coordinates
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random x and y coordinates
    x_coords = np.random.randint(0, image_width, size=num_points)
    y_coords = np.random.randint(0, image_height, size=num_points)

    # Stack them into a single array of shape (num_points, 2)
    random_points = np.column_stack((x_coords, y_coords))

    return random_points


import seaborn as  sns

def get_activation_boxplot(
    high_activations:torch.Tensor, 
    low_activations:torch.Tensor, 
    ax,
    token_wise=False,pertoken=False, max_channels=64, offset=0, show_plot=True
):
    
    if not token_wise:
        high_data = to_numpy(high_activations.reshape(-1, high_activations.shape[-1])[:, offset:offset+max_channels])
        low_data = to_numpy(low_activations.reshape(-1, low_activations.shape[-1])[:, offset:offset+max_channels])
        high_channel_names = np.repeat(np.array([f"{i+1}" for i in  range(max_channels)]), high_data.shape[0])
        low_channel_names = np.repeat(np.array([f"{i+1}" for i in  range(max_channels)]), low_data.shape[0])
        high_data = high_data.flatten(order='F')
        low_data = low_data.flatten(order='F')

        types = ['high']*(high_data.shape[0])+['low']*(low_data.shape[0])

        df = pd.DataFrame({
            'values':  np.concatenate([high_data, low_data]),
            'channel': np.concatenate([high_channel_names, low_channel_names]),
            'types':  types,
        }) 

        # Create Plotly violin plot
        sns.violinplot(
            df, 
            ax=ax,
            x='channel', 
            y='values', 
            hue='types',
            split=True,
            inner='quart'
        )
    else:
        if not pertoken:
            if len(high_activations.shape)== 3:
                Bh, Th, Ch = high_activations.shape
                Bl, Tl, Cl = low_activations.shape
                high_data = to_numpy(high_activations.permute(1,0,2).reshape(Th, Bh, Ch)[:, :, offset:offset+max_channels])
                low_data = to_numpy(low_activations.permute(1,0,2).reshape(Tl, Bl, Cl)[:, :, offset:offset+max_channels])
                # max_token = max_channels
                # high_data = to_numpy(high_activations.permute(1,0,2).reshape(Th, Bh, Ch)[offset:offset+max_token, :, :])
                # low_data = to_numpy(low_activations.permute(1,0,2).reshape(Tl, Bl, Cl)[offset:offset+max_token, :, :])
            else:
                Bh, Hh, Th,Ch = high_activations.shape
                Bl, Hl, Tl, Cl = low_activations.shape
                high_data = to_numpy(high_activations.permute(2,0,1,3).reshape(Th, Bh, Ch*Hh)[:, :, offset:offset+max_channels])
                low_data = to_numpy(low_activations.permute(2,0,1,3).reshape(Tl, Bl, Cl*Hl)[:, :, offset:offset+max_channels])
                # high_data = to_numpy(high_activations.permute(1,0,2).reshape(Th, Bh, Ch)[offset:offset+max_token, :, :])
                # low_data = to_numpy(low_activations.permute(1,0,2).reshape(Tl, Bl, Cl)[offset:offset+max_token, :, :])

            high_data = high_data.reshape(Th, -1)
            low_data = low_data.reshape(Tl, -1)
            
            high_token_names = np.repeat(np.array([f"{i+1}" for i in  range(Th)]), max_channels*Bh)
            low_token_names = np.repeat(np.array([f"{i+1}" for i in  range(Tl)]), max_channels*Bl)
            
            high_data = high_data.flatten(order='C')
            low_data = low_data.flatten(order='C')
            
            types = ['high']*(Th*max_channels*Bh)+['low']*(Tl*max_channels*Bl)
            
            df = pd.DataFrame({
                'values':  np.concatenate([high_data, low_data]),
                'token': np.concatenate([high_token_names, low_token_names]),
                'types':  types,
            }) 
        
            sns.violinplot(
                df, 
                ax=ax,
                x='types', 
                y='values', 
                hue='types',
                split=True,
                inner='quart'
            )
        else:
            if len(high_activations.shape)== 3:
                Bh, Th, Ch = high_activations.shape
                Bl, Tl, Cl = low_activations.shape
                max_token = max_channels
                high_data = to_numpy(high_activations.permute(1,0,2).reshape(Th, Bh, Ch)[offset:offset+max_token, :, :])
                low_data = to_numpy(low_activations.permute(1,0,2).reshape(Tl, Bl, Cl)[offset:offset+max_token, :, :])
            else:
                Bh, Hh, Th,Ch = high_activations.shape
                Bl, Hl, Tl, Cl = low_activations.shape
                max_token = max_channels
                high_data = to_numpy(high_activations.permute(2,0,1,3).reshape(Th, Bh, Ch*Hh)[:, :, offset:offset+max_channels])
                low_data = to_numpy(low_activations.permute(2,0,1,3).reshape(Tl, Bl, Cl*Hl)[:, :, offset:offset+max_channels])

            
            high_data = high_data.reshape(max_token, -1)
            low_data = low_data.reshape(max_token, -1)   
            
            a=high_data.shape[1]
            high_token_names = np.repeat(np.array([f"{i+1}" for i in  range(max_token)]), a)
            low_token_names = np.repeat(np.array([f"{i+1}" for i in  range(max_token)]),a)

            high_data = high_data.flatten(order='C')
            low_data = low_data.flatten(order='C')
            types = ['high']*(max_token*a)+['low']*(max_token*a)
            
            df = pd.DataFrame({
                'values':  np.concatenate([high_data, low_data]),
                'token': np.concatenate([high_token_names, low_token_names]),
                'types':  np.array(types),
            }) 
        
            sns.violinplot(
                df, 
                ax=ax,
                x='token', 
                y='values', 
                hue='types',
                split=True,
                inner='quart'
            )
        


def separate_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, n, c = x.shape
    x = x.reshape(b, n, num_heads, c // num_heads)
    return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

def recombine_heads(x: torch.Tensor) -> torch.Tensor:
    b, n_heads, n_tokens, c_per_head = x.shape
    x = x.transpose(1, 2)
    return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

def re_cal_attn(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    _, _, _, c_per_head = q.shape
    attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
    attn = attn / math.sqrt(c_per_head)
    attn = torch.softmax(attn, dim=-1)

    return attn


    