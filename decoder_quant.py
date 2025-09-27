from abc import abstractmethod
import torch
import torch.nn.functional as F
from segment_anything.modeling.transformer import TwoWayAttentionBlock, TwoWayTransformer, Attention
from quant.utils.observer import ObserverBase 
import torch.nn as nn
from tqdm.auto import tqdm
from collections import defaultdict
from typing import Type, Tuple, Optional
import math
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import numpy as np
import cv2
import matplotlib.pyplot  as plt
from typing import Optional
import pandas as pd 
from quant_utils import ProcessStrategy, quantize_activation_per_token_absmax
from utils import show_points, show_mask_image
from quant_utils import AttnBasedProcessor ,DoNothingProcessor



def to_numpy(x:torch.Tensor):
    return x.detach().cpu().numpy()





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
        print(final_k.shape)
        TwoWayTransformerObserver.attention_score['final_attn'] = final_attn
        TwoWayTransformerObserver.attention_score['final_q'] = final_q
        TwoWayTransformerObserver.attention_score['final_k'] = final_k
        TwoWayTransformerObserver.attention_score['final_v'] = final_v
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
    # def set_processor(self, processor:ProcessStrategy):
        # self.processor = processor



    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        
        # q =   quantize_activation_per_token_absmax(q, n_bits=4)
        # k =   quantize_activation_per_token_absmax(k, n_bits=8)
        # v =   quantize_activation_per_token_absmax(v, n_bits=8)


        if self.processor is not None and ('cross' in self.name or 'final' in self.name): 
            q = self._separate_heads(q, self.num_heads)
            k = self._separate_heads(k, self.num_heads)
            v = self._separate_heads(v, self.num_heads)
            q,k,v = self.processor.process(q, k, v, self.name)
            q = self._recombine_heads(q )
            k = self._recombine_heads(k )
            v = self._recombine_heads(v )

        

        q =  quantize_activation_per_token_absmax(q, n_bits=self.n_bits)
        k =  quantize_activation_per_token_absmax(k, n_bits=self.n_bits)
        v =  quantize_activation_per_token_absmax(v, n_bits=self.n_bits)

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

def mask_decoder_monkey_patch(model, processor:ProcessStrategy, n_bits=8):
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            module.__class__ = AttentionObserver
            module.processor = processor
            module.name = name 
            module.n_bits=n_bits
        if isinstance(module, TwoWayAttentionBlock):
            module.__class__ = TwoWayAttentionBlockObserver
        if isinstance(module, TwoWayTransformer):
            module.__class__ = TwoWayTransformerObserver 





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
        input_point = np.array([[221, 482], [498, 633], [750, 379]])
        input_label = np.ones(input_point.shape[0])
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
        print(image.shape)
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
            
        plt.title(f'Example {example_idx} - Score: {scores[0]:.3f}')
        plt.savefig('demo.png')
        plt.axis('off')
        plt.show()

    return masks, scores, logits




import seaborn as  sns

def get_activation_boxplot(
    high_activations:torch.Tensor, 
    low_activations:torch.Tensor, 
    ax,
    token_wise=False, 
    max_channels=64, 
    offset=0, 
    show_plot=True,
    diff:torch.Tensor = None 
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
        if diff is not None:
            diff = to_numpy(diff.squeeze())[offset:offset+max_channels]
            print(diff.shape)
            print(len(high_channel_names))
            
            df = pd.DataFrame({
                'value':  diff,
                'channel': np.array([f"{i+1}" for i in  range(max_channels)])
            }) 
            sns.barplot(
                df, 
                ax=ax,
                x='channel', 
                y='value', 
                alpha=0.5
            )

    else:
        if len(high_activations.shape)== 3:
            Bh, Th, Ch = high_activations.shape
            Bl, Tl, Cl = low_activations.shape
            high_data = to_numpy(high_activations.permute(1,0,2).reshape(Th, Bh, Ch)[:, :, offset:offset+max_channels])
            low_data = to_numpy(low_activations.permute(1,0,2).reshape(Tl, Bl, Cl)[:, :, offset:offset+max_channels])
        else:
            Bh, Hh, Th,Ch = high_activations.shape
            Bl, Hl, Tl, Cl = low_activations.shape
            high_data = to_numpy(high_activations.permute(2,0,1,3).reshape(Th, Bh, Ch*Hh)[:, :, offset:offset+max_channels])
            low_data = to_numpy(low_activations.permute(2,0,1,3).reshape(Tl, Bl, Cl*Hl)[:, :, offset:offset+max_channels])

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
        sns.set_style('darkgrid')

        sns.violinplot(
            df, 
            ax=ax,
            x='types', 
            y='values', 
            hue='types',
            split=True,
            inner='quart'
        )






if __name__ == '__main__':


    model_type = 'vit_l'
    num_calib_samples=8
    checkpoint_path= './pretrained_checkpoint/sam_hq_vit_l.pth'
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)
    processor = DoNothingProcessor('base') 
    
    # processor = AttnBasedProcessor('attn') 
    # processor.calibrate(
    #     predictor=predictor, 
    #     modules=(TwoWayTransformer),
    #     num_samples=num_calib_samples
    # )
    mask_decoder_monkey_patch(predictor.model, processor, n_bits=3)
    results = inference_image(predictor, image_dir='./input_imgs/', example_idx=3, show_image=True)
    

    