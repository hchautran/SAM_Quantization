import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional, Tuple, Callable, List
from functools import partial
from matplotlib import pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up to SAM_Quantization
sam_hq_path = os.path.join(project_root, "sam-hq")

sys.path.insert(0, sam_hq_path)

# Now import from sam-hq with correct path handling
from segment_anything import build_sam, sam_model_registry, SamPredictor
from segment_anything.modeling.image_encoder import ImageEncoderViT  
from segment_anything.modeling.mask_decoder_hq import MaskDecoderHQ
from segment_anything.modeling.mask_decoder import MaskDecoder
from train.segment_anything_training.modeling.image_encoder import ImageEncoderViT as ImageEncoderViTtrain
from RTN_quantization import per_tensor_channel_group
from quarot import utils, rotation_utils
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
class ObserverBase:
    dictionary = {}
    def __init__(self, module_list: Tuple):
        self.module_list = module_list
        self.hooks = []

    def register_hooks(self,
        model: nn.Module,
        pre_hook: Optional[Callable] = None,
        post_hook: Optional[Callable] = None,
    ):
        for name, module in model.named_modules():
            if post_hook is not None:
                if isinstance(module, self.module_list):
                    self.hooks.append(
                        module.register_forward_hook(partial(post_hook, name=name))
                    )
            if pre_hook is not None:
                if isinstance(module, self.module_list):
                    self.hooks.append(
                        module.register_forward_pre_hook(partial(pre_hook, name=name))
                    )

    def clear_hook(self):
        for hook in self.hooks:
            hook.remove()

    def clear_dict(self):
        ObserverBase.dictionary = {}

    @torch.inference_mode()
    def inference_image(
        self,
        predictor,
        image_dir: str = './input_imgs/example1.png',
        show_image: bool = False,
        example_idx: int = 1,  # Which example configuration to use
    ):
        """
        Run inference on a single image using either SamPredictor or Sam model directly
        """
        import cv2
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Configure based on example index
        if example_idx == 0:
            input_box = np.array([[4, 13, 1007, 1023]])
            input_point, input_label = None, None
            hq_token_only = False
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
            masks, scores, logits = self._inference_with_sam_model(
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
            plt.axis('off')
            plt.show()

        return masks, scores, logits

    def _inference_with_sam_model(
        self,
        sam_model: 'Sam',
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

class Distribution():
    def __init__(self, n_channels:int, dist_name:str ,mean:np.ndarray, max:np.ndarray, min:np.ndarray, p75:np.ndarray, p25:np.ndarray, p99:np.ndarray, p1:np.ndarray):
        self.n_channels = n_channels
        self.mean = mean
        self.max = max
        self.min = min
        self.p75 = p75
        self.p25 = p25
        self.p99 = p99
        self.p1 = p1
        self.dist_name = dist_name


    def plot_channel_distribution(self, ax):
        x = np.arange(self.n_channels)
        ax.plot(x, self.mean, color='black', linewidth=-1)
        ax.scatter(x, self.min, color='skyblue', linewidth=-1)
        ax.scatter(x, self.max, color='skyblue', linewidth=-1, label='Min/Max')
        ax.plot(x, self.p1, color='red', linewidth=1)
        ax.plot(x, self.p99, color='red', linewidth=1, label='1/99 Percentile')
        ax.plot(x, self.p25, color='orange', linewidth=1)
        ax.plot(x, self.p75, color='orange', linewidth=1, label='25/75 Percentile')
        ax.set_title(self.dist_name)
        ax.set_xlabel('Hidden dimension index')
        ax.set_ylabel('Activation value')

    def box_plot_channel_distribution(self,ax):
        pass



def get_activation_distribution(activations:np.ndarray, title:str) -> Distribution:
# Compute percentiles along the sample axis (axis=-1)
    activations = np.reshape(activations, (-1, activations.shape[-1]))

    min_val= np.min(activations, axis=0)
    max_val = np.max(activations, axis=0)
    mean= np.mean(activations, axis=0)
    p1 = np.percentile(activations, 1, axis=0)
    p99 = np.percentile(activations, 99, axis=0)
    p25 = np.percentile(activations, 25, axis=0)
    p75 = np.percentile(activations, 75, axis=0)

    return Distribution(
        n_channels=activations.shape[-1],
        dist_name=title,
        mean=mean,
        min=min_val,
        max=max_val,
        p1=p1,
        p99=p99,
        p25=p25,
        p75=p75
    )


def get_submodule_names(module:nn.Module)->dict:
    if isinstance(module, ImageEncoderViT) or isinstance(module, ImageEncoderViTtrain):
        return {
            'QKV': 'attn.qkv',
            'O': 'attn.proj',
            'MLP_up': 'mlp.lin1',
            'MLP_down': 'mlp.lin2'
        }

    elif isinstance(module, MaskDecoder):
        return {
            'self attn Q': 'self_attn.q_proj',
            'self attn K': 'self_attn.k_proj',
            'self attn V': 'self_attn.v_proj',
            'self attn O': 'self_attn.out_proj',
            'cross attn i2t Q': 'cross_attn_image_to_token.q_proj',
            'cross attn i2t K': 'cross_attn_image_to_token.k_proj',
            'cross attn i2t V': 'cross_attn_image_to_token.v_proj',
            'cross attn i2t O': 'cross_attn_image_to_token.out_proj',
            'cross attn t2i Q': 'cross_attn_token_to_image.q_proj',
            'cross attn t2i K': 'cross_attn_token_to_image.k_proj',
            'cross attn t2i V': 'cross_attn_token_to_image.v_proj',
            'cross attn t2i O': 'cross_attn_token_to_image.out_proj',
            'final attn t2i Q': 'final_attn_token_to_image.q_proj',
            'final attn t2i K': 'final_attn_token_to_image.k_proj',
            'final attn t2i V': 'final_attn_token_to_image.v_proj',
            'final attn t2i O': 'final_attn_token_to_image.out_proj',
            'MLP_up': 'mlp.lin1',
            'MLP_down': 'mlp.lin2',
        }


def get_model_name(module:nn.Module):
    if isinstance(module, ImageEncoderViT) or isinstance(module, ImageEncoderViTtrain):
        return 'image_encoder.blocks'
    elif isinstance(module, MaskDecoderHQ):
        return 'mask_decoder.transformer.layers'


class ActivationObserver(ObserverBase):

    def __init__(self, module_list:Tuple):
        super().__init__(module_list)
        self.IMAGE_ENCODER = 'Image Encoder'
        self.MASK_DECODER = 'Mask Decoder'
        self.bins = 200
        self.name_dict = defaultdict(list)
        self.activation_dict = {}

    def clear_dict(self):
        self.name_dict = {}
        self.activation_dict = {}

    def get_linear_name(self, module:nn.Module, layer_idxes:List[int]):
        sub_modules = get_submodule_names(module)

        model = get_model_name(module)

        if  isinstance(module, ImageEncoderViT) or isinstance(module, ImageEncoderViTtrain):
            num_layers=len(module.blocks)
            print('init num layers', num_layers)
            for layer_idx in range(num_layers):
                if layer_idx in layer_idxes:
                    for key in sub_modules.keys():
                        self.name_dict[self.IMAGE_ENCODER].append(f'{model}.{layer_idx}.{sub_modules[key]}')
        elif isinstance(module, MaskDecoderHQ):
            num_layers=len(module.transformer.layers)
            print('init num layers', num_layers)
            for layer_idx in range(num_layers):
                if layer_idx in layer_idxes:
                    for key in sub_modules.keys():
                        self.name_dict[self.MASK_DECODER].append(f'{model}.{layer_idx}.{sub_modules[key]}')
      
    def init_activation_cache(self, module:nn.Module, layer_idxes:List[int]=[0]):
        if 'activation' not in self.activation_dict.keys():
            self.activation_dict = {}

        self.get_linear_name(module, layer_idxes)
        for module_type in self.name_dict.keys():
            if len(self.name_dict[module_type]) > 0:
                for module in self.name_dict[module_type]:
                    self.activation_dict[module]=None
                    # ]=torch.zeros((self.bins,)).cpu().detach().numpy()

    def register_tensor_distribution_hook(self, model:nn.Module, use_post_hook=False, min=-0.05, max=0.05):
        def pre_hook(module, input, name):
            input_tensor = input[0]
            bins = torch.linspace(min, max, self.bins)
            hist = torch.histc(input_tensor.float(), bins=bins.shape[0], min=min, max=max)
            hist = (hist/hist.sum()).cpu().detach().numpy()
            if self.activation_dict[name] is None:
                self.activation_dict[name] = torch.zeros((self.bins,)).cpu().detach().numpy()
            self.activation_dict[name] += hist

        def post_hook(module, input, output, name):
            output_tensor = output
            bins = torch.linspace(min, max, self.bins)
            hist = torch.histc(output_tensor.float(), bins=bins.shape[0], min=min, max=max)
            hist = (hist/hist.sum()).cpu().detach().numpy()
            if self.activation_dict[name] is None:
                self.activation_dict[name] = torch.zeros((self.bins,)).cpu().detach().numpy()
            self.activation_dict[name] += hist


        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self.activation_dict.keys():
                if use_post_hook:
                    module.register_forward_hook(partial(post_hook,name=name))
                else:
                    module.register_forward_pre_hook(partial(pre_hook,name=name))


    def cal_density(self, X:torch.Tensor, margin=0.5):
        X = F.normalize(X.reshape(-1, X.shape[-1]), p=2, dim=-1)
        score_map = F.elu(X@X.transpose(-1,-2) - margin)
        scores = score_map.mean(-1)
        return scores
    def register_token_distribution_hook(self, model: nn.Module, min_val: float = -2.0, max_val: float = 2.0, use_post_hook=False):
        def pre_hook(module, input, name):
            input_tensor = input[0]  # Shape: [batch_size, seq_len, hidden_dim]
            if len(input_tensor.shape) == 3:
                batch_size, seq_len, hidden_dim = input_tensor.shape
                input_flat = input_tensor.reshape(batch_size * seq_len, hidden_dim)
            else:
                input_flat = input_tensor.reshape(-1, input_tensor.shape[-1])

            density = self.cal_density(input_flat)  # Shape: [batch_size * seq_len]
            sorted_idx = torch.argsort(density, dim=-1, descending=False)

            total_tokens = input_flat.shape[0]
            split_point = min(200, total_tokens // 2)  # Now min() works correctly

            low_d_inputs = input_flat[sorted_idx[:split_point]]
            high_d_inputs = input_flat[sorted_idx[split_point:]]
           
            bins = torch.linspace(min_val, max_val, self.bins)

            hist_high= torch.histc(high_d_inputs.float(), bins=bins.shape[0], min=min_val, max=max_val)
            hist_low = torch.histc(low_d_inputs.float(), bins=bins.shape[0], min=min_val, max=max_val)
            hist_high = (hist_high/hist_high.sum()).cpu().detach().numpy()
            hist_low= (hist_low/hist_high.sum()).cpu().detach().numpy()

            if self.activation_dict[name] is None:
                self.activation_dict[name] = {}
                self.activation_dict[name]['high'] = torch.zeros((self.bins,)).cpu().detach().numpy()
                self.activation_dict[name]['low'] = torch.zeros((self.bins,)).cpu().detach().numpy()
           
            self.activation_dict[name]['high'] += hist_high
            self.activation_dict[name]['low'] += hist_low

        def post_hook(module, input, output, name):
            output_tensor = output  # Shape: [batch_size, seq_len, hidden_dim]
            if len(output_tensor.shape) == 3:
                batch_size, seq_len, hidden_dim = output_tensor.shape
                output_flat = output_tensor.reshape(batch_size * seq_len, hidden_dim)
            else:
                output_flat = output_tensor.reshape(-1, output_tensor.shape[-1])

            density = self.cal_density(output_flat)
            sorted_idx = torch.argsort(density, dim=-1, descending=False)
            
            # Dynamically determine split point
            total_tokens = output_flat.shape[0]
            split_point = min(100, total_tokens // 2)

            bins = torch.linspace(min_val, max_val, self.bins)

            low_d_outputs = output_flat[sorted_idx[:split_point]]
            high_d_outputs = output_flat[sorted_idx[split_point:]]

            hist_high= torch.histc(high_d_outputs.float(), bins=bins.shape[0], min=min_val, max=max_val)
            hist_low = torch.histc(low_d_outputs.float(), bins=bins.shape[0], min=min_val, max=max_val)

            hist_high = (hist_high/hist_high.sum()).cpu().detach().numpy()
            hist_low= (hist_low/hist_high.sum()).cpu().detach().numpy()

            if self.activation_dict[name] is None:
                self.activation_dict[name] = {}
                self.activation_dict[name]['high'] = torch.zeros((self.bins,)).cpu().detach().numpy()
                self.activation_dict[name]['low'] = torch.zeros((self.bins,)).cpu().detach().numpy()
            
            self.activation_dict[name]['high'] += hist_high
            self.activation_dict[name]['low'] += hist_low

        for name, module in model.named_modules():
            if (isinstance(module, nn.Linear) or isinstance(module,per_tensor_channel_group.W8A8Linear) or isinstance(module,utils.ActQuantWrapper)) and name in self.activation_dict.keys():
                if use_post_hook:
                    module.register_forward_hook(partial(post_hook, name=name))
                else:
                    module.register_forward_pre_hook(partial(pre_hook, name=name))

    def register_hessian_hook(self, model:nn.Module):

        def pre_hook(module, input, name):
            pass

        def post_hook(module, input, output, name):
            pass

        for name, module in model.named_modules():
            if (isinstance(module, nn.Linear) or isinstance(module,per_tensor_channel_group.W8A8Linear) or isinstance(module,utils.ActQuantWrapper)) and name in self.activation_dict.keys():
                # module.register_forward_pre_hook(partial(pre_hook,name=name))
                module.register_forward_hook(partial(post_hook,name=name))

    def register_channel_distribution_hook(self, model:nn.Module,Q_image_encoder, use_post_hook=False):
        
        def pre_hook(module, input, name):
            input_tensor = input[0]
            
            name_class = type(module).__name__

            if Q_image_encoder is not None and 'ActQuantWrapper' in name_class and ('lin1' in name or 'qkv' in name):
                
                B, H, W, C = input_tensor.shape
                Q = Q_image_encoder.to(dtype=input_tensor.dtype, device=input_tensor.device)
                x_flat = input_tensor.reshape(B, H*W, C)
                x_rotated = torch.matmul(x_flat, Q)
                rotated_tensor = x_rotated.reshape(B, H, W, C)
                input_tensor = rotated_tensor.cpu().detach().numpy()
            else:
                input_tensor = input_tensor.cpu().detach().numpy()
            self.activation_dict[name] = get_activation_distribution(input_tensor, title=name)


        def post_hook(module, input, output, name):
            output_tensor = output.cpu().detach().numpy()
            self.activation_dict[name]=get_activation_distribution( output_tensor, title=name )

        for name, module in model.named_modules():
            name_class= type(module).__name__
            if (isinstance(module, nn.Linear) or isinstance(module,per_tensor_channel_group.W8A8Linear) or 'ActQuantWrapper' in name_class ) and name in self.activation_dict.keys():
                
                if use_post_hook:
                    module.register_forward_hook(partial(post_hook,name=name))
                else:
                    module.register_forward_pre_hook(partial(pre_hook,name=name))

    def get_activation_distribution(self, name):
        return get_activation_distribution(activations=self.activation_dict[name], title=name)




def get_tensor_distribution(checkpoint_path, model_type='vit_l', min_val=-2.0, max_val=2.0, use_post_hook=False):
    observer = ActivationObserver(module_list=(nn.Linear,))
    
    # Build SAM model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)

    observer.init_activation_cache(sam.image_encoder, layer_idxes=[0, 6, 12, 18])
    observer.register_tensor_distribution_hook(sam, min_val=min_val, max_val=max_val, use_post_hook=use_post_hook)
    observer.inference_image(predictor=predictor, show_image=False)

    names = list(observer.activation_dict.keys())
    print(f"Found {len(names)} layers")
    
    if len(names) > 0:
        rows = max(1, (len(names) + 3) // 4)
        fig, axes = plt.subplots(rows, 4, figsize=(20, 4 * rows))
        
        # Handle single row case
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif len(names) == 1:
            axes = np.array([[axes]])
        
        for i, name in enumerate(names):
            row, col = i // 4, i % 4
            x_values = np.linspace(min_val, max_val, observer.bins)
            
            if observer.activation_dict[name] is not None:
                hist = observer.activation_dict[name].squeeze()
                axes[row, col].step(x_values, hist, where='mid', alpha=0.8, linewidth=2)
                axes[row, col].fill_between(x_values, hist, step='mid', alpha=0.3)
                axes[row, col].set_title(name.replace('image_encoder.blocks.', 'Block '), fontsize=10)
                axes[row, col].set_xlabel('Activation Value')
                axes[row, col].set_ylabel('Probability Density')
                axes[row, col].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(names), rows * 4):
            row, col = i // 4, i % 4
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'Tensor Distribution Analysis - {model_type}', fontsize=16)
        plt.tight_layout()

        output_dir = os.path.join(project_root, 'demo', 'distribution')
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f'tensor_distribution_{model_type}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
        plt.show()
    
    observer.clear_hook()
    observer.clear_dict()

def get_channel_distribution(checkpoint_path, model_type='vit_l'):
    observer = ActivationObserver(module_list=(nn.Linear,))

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)

    observer.init_activation_cache(sam.image_encoder, layer_idxes=[1, 6, 12, 18, 23])
    observer.register_channel_distribution_hook(sam)
    observer.inference_image(predictor=predictor, show_image=False)

    names = list(observer.activation_dict.keys())
    if len(names) > 0:
        fig, axes = plt.subplots(len(names), 1, figsize=(15, 4 * len(names)))
        if len(names) == 1:
            axes = [axes]
            
        for i, name in enumerate(names):
            print(f"Plotting distribution for {name}")
            if observer.activation_dict[name] is not None:
                observer.activation_dict[name].plot_channel_distribution(axes[i])
                axes[i].legend()
        
        plt.suptitle(f'Channel Distribution Analysis - {model_type}', fontsize=16)
        plt.tight_layout()

        output_dir = os.path.join(project_root, 'demo', 'distribution')
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f'channel_distribution_{model_type}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
        plt.show()
    else:
        print("No distributions to plot!")
    
    observer.clear_hook()
    observer.clear_dict()

def get_tensor_density_distribution(checkpoint_path, model_type='vit_l', min_val=-2.0, max_val=2.0, layer_idxes=[0]):
    observer = ActivationObserver(module_list=(nn.Linear,))

    # Build SAM model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)
    
    # Initialize and run inference
    observer.init_activation_cache(sam.image_encoder, layer_idxes=layer_idxes)
    observer.register_token_distribution_hook(sam, min_val, max_val)
    observer.inference_image(predictor=predictor, show_image=False)
    
    names = list(observer.activation_dict.keys())
    print(f"Found {len(names)} layers for density distribution")
    
    if len(names) > 0:
        rows = max(1, (len(names) + 3) // 4)
        fig, axes = plt.subplots(rows, 4, figsize=(20, 4 * rows))
        
        # Handle subplot array shape
        if rows == 1:
            if len(names) == 1:
                axes = np.array([[axes]])
            else:
                axes = axes.reshape(1, -1)
            
        for i, name in enumerate(names):
            row, col = i // 4, i % 4
            
            if observer.activation_dict[name] is not None and isinstance(observer.activation_dict[name], dict):
                high_hist = observer.activation_dict[name]['high']
                low_hist = observer.activation_dict[name]['low']
                
                # Create x-axis values for plotting
                x_values = np.linspace(min_val, max_val, len(high_hist))
                
                # Plot using step plot
                axes[row, col].step(x_values, high_hist, where='mid', alpha=0.8, label='High Density', color='red', linewidth=2)
                axes[row, col].step(x_values, low_hist, where='mid', alpha=0.8, label='Low Density', color='blue', linewidth=2)
                axes[row, col].fill_between(x_values, high_hist, step='mid', alpha=0.3, color='red')
                axes[row, col].fill_between(x_values, low_hist, step='mid', alpha=0.3, color='blue')
                
                axes[row, col].set_title(name.replace('image_encoder.blocks.', 'Block '), fontsize=10)
                axes[row, col].set_xlabel('Activation Value')
                axes[row, col].set_ylabel('Probability Density')
                axes[row, col].legend()
                axes[row, col].grid(True, alpha=0.3)
            else:
                axes[row, col].text(0.5, 0.5, f'No data\n{name}', transform=axes[row, col].transAxes, ha='center')
        
        # Hide empty subplots
        for i in range(len(names), rows * 4):
            row, col = i // 4, i % 4
            axes[row, col].set_visible(False)
        
        plt.suptitle(f'Token Density Distribution Analysis - {model_type}', fontsize=16)
        plt.tight_layout()
        
        output_dir = os.path.join(project_root, 'demo', 'distribution')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save and show plot
        filename = os.path.join(output_dir, f'density_distribution_{model_type}_layers_{"_".join(map(str, layer_idxes))}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
        plt.show()
    else:
        print("No density distributions to plot!")
    
    observer.clear_hook()
    observer.clear_dict()

def get_channel_distribution_modify(sam,model_type, act,rot_args=None):
    if rot_args is not None:
        Q_image_encoder = rotation_utils.get_orthogonal_matrix(rot_args.hidden_size_image_en,rot_args.rotate_mode,device = rot_args.device,seed=rot_args.seed)
    else:
        Q_image_encoder = None   
    observer = ActivationObserver(module_list=(nn.Linear,))
    observer.init_activation_cache(sam.image_encoder, layer_idxes=[1, 6, 12, 18])
    observer.register_channel_distribution_hook(sam,Q_image_encoder)
    observer.inference_image(predictor=sam, show_image=False)

    names = list(observer.activation_dict.keys())
    if len(names) > 0:
        fig, axes = plt.subplots(len(names), 1, figsize=(15, 4 * len(names)))
        if len(names) == 1:
            axes = [axes]
            
        for i, name in enumerate(names):
            print(f"Plotting distribution for {name}")
            if observer.activation_dict[name] is not None:
                observer.activation_dict[name].plot_channel_distribution(axes[i])
                axes[i].legend()
        
        plt.suptitle(f'Channel Distribution Analysis - {model_type}', fontsize=16)
        plt.tight_layout()

        output_dir = os.path.join(project_root, 'demo', 'distribution')
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f'channel_distribution_{model_type}_{act}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
        plt.show()
    else:
        print("No distributions to plot!")
    
    observer.clear_hook()
    observer.clear_dict()

# %%
if __name__ == '__main__':

    checkpoint_path = "/media/caduser/MyBook/chau/chi/SAM_Quantization/pretrained_checkpoint/sam_hq_vit_l.pth"
    model_type = "vit_l"
    
    # get_tensor_density_distribution(
    #     checkpoint_path=checkpoint_path,
    #     model_type=model_type,
    #     min_val=-2.0,
    #     max_val=2.0,
    #     layer_idxes=[6, 12]
    # )
    get_channel_distribution(checkpoint_path=checkpoint_path, model_type= model_type)
