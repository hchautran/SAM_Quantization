import copy
# Standard library imports
from collections import defaultdict
from typing import Optional, Tuple

# Third-party imports
import torch
import torch.nn as nn

# SAM model imports
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling.image_encoder import (
    Attention,
    Block,
    ImageEncoderViT,
)
from seginw.segment_anything.modeling.image_encoder import Attention as Attention__
from seginw.segment_anything.modeling.image_encoder import Block as Block__
from seginw.segment_anything.modeling.image_encoder import ImageEncoderViT as ImageEncoderViT__
from train.segment_anything_training.modeling.image_encoder import Attention as Attention_
from train.segment_anything_training.modeling.image_encoder import Block as block_
from train.segment_anything_training.modeling.image_encoder import ImageEncoderViT as ImageEncoderViT_
# Local imports
from quant_utils import (
    ImageEncoderProcessor,
    quantize_activation_per_token_absmax,
)
from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
from RTN_quantization import per_tensor_channel_group
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'RTN_quantization'))
from RTN_quantization.utils import replace_linear_with_quantized, QuantizationConfig
from RTN_quantization.per_tensor_channel_group import quantize_activation_low_high_density_activation_index
# from utils import inference_image, to_numpy
from segment_anything.modeling.image_encoder import (
    window_partition,
    window_unpartition,
)

def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()



class AttentionObserver(Attention):


    attention_score = defaultdict(list)

    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent Attention class."""
        super().__init__(*args, **kwargs)
        # Quantization attributes
        self.n_bits = 8
        self.name = ""
        self.stat = None
        self.percent= 50
        self.qkT_v = False
        self.qkv_w_hat = None
        self.Q= None
    def smooth(self,alpha=0.5):
        input_qkv = self.stat['inputqkv'] # shape B , H, W , C  where C= num_heads * dim = 1024
        output_qkv = self.stat['qkv']    # shape B , H, W , 3* num_heads * dim
        act_scales = self.stat['act_scales']
        
        # Get device and dtype from qkv_w_hat
        device, dtype = self.qkv_w_hat.weight.device, self.qkv_w_hat.weight.dtype
        act_scales = act_scales.to(device=device, dtype=dtype)
        
        # Extract Q matrix weights (first 1/3 of qkv_w_hat)
        C = act_scales.numel()  # 1024
        q_weight = self.qkv_w_hat.weight[:C, :]  # Q matrix: (1024, 1024)
        
        # Calculate weight scales for Q matrix only
        weight_scales = q_weight.abs().max(dim=0)[0].clamp(min=1e-5)  # (1024,)
        
        # Calculate smoothing scales (same logic as smooth_ln_fcs)
        scales = (
            (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
            .clamp(min=1e-5)
            .to(device)
            .to(dtype)
        )
        self.q_scales = scales

        with torch.no_grad():
            self.qkv_w_hat.weight[:C, :].mul_(scales.view(1, -1))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  
        B, H, W, C = x.shape
        if hasattr(self, 'Q') and self.Q is not None:
            self.Q = self.Q.to( dtype=x.dtype)
            
            x_flat = x.reshape(B, H*W, C)
            x_rotated = torch.matmul(x_flat, self.Q)
            x = x_rotated.reshape(B, H, W, C)
        # qkv with shape (3, B, nHead, H * W, C)
        x = x.reshape(B, H*W, C)
        bias = self.qkv.bias[None, None, ...]
        x_mean = x.mean(1, keepdim=True)
        x_hat= (x - x_mean)
        # qkv     = self.qkv(x)       # shape: (B, H*W, 3*num_heads*dim)
        qkv_hat = self.qkv_w_hat(x_hat, self.q_scales)
        qkv_mean = self.qkv(x_mean) - bias


        qkv_hat = qkv_hat.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        qkv_mean = qkv_mean.reshape(B, 1, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # import ipdb; ipdb.set_trace()
        q_hat, k_hat, v_hat = qkv_hat.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        q_mean, _, v_mean = qkv_mean.reshape(3, B * self.num_heads, 1, -1).unbind(0)
        if ImageEncoderViTObserver.debug:
            qkv   = self.qkv(x)       # shape: (B, H*W, 3*num_heads*dim)
            qkv = qkv.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
            # Assert that q_ori = q + q_mean (and similarly for k, v)
            assert torch.allclose(q, q_hat+q_mean, rtol=1e-4, atol=1e-4), "q_ori != q + q_mean"
            # assert torch.allclose(k, k_hat+k_mean, rtol=1e-4, atol=1e-4), "k_ori != k + k_mean"
            assert torch.allclose(v, v_hat+v_mean, rtol=1e-4, atol=1e-4), "v_ori != v + v_mean"
            attn_mean = (q_mean * self.scale) @ k_hat.transpose(-2, -1)
            attn_ori = (q * self.scale) @ k.transpose(-2, -1)
        # q_hat = q_hat+q_mean
        # k_hat = k_hat+k_mean
        

        # Compute attention
      
        attn = (q_hat * self.scale) @ k_hat.transpose(-2, -1)
        # print('got')
        # attn = attn + attn_mean

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q_hat+q_mean, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )
            if ImageEncoderViTObserver.debug:
                # attn_ori = add_decomposed_rel_pos(
                    # attn_ori, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
                # )
                attn_ori = add_decomposed_rel_pos(
                    attn_ori, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
                )

        attn = attn.softmax(dim=-1)
        
        if self.qkT_v:

            attn, indices = quantize_activation_low_high_density_activation_index(attn, n_bits=self.n_bits,percent=self.percent, quantizehigh=True, )
            v=quantize_activation_low_high_density_activation_index(v_hat+v_mean, n_bits=self.n_bits, percent=self.percent, quantizehigh=True, indices=indices)[0]
        else:
            v= v_hat+v_mean
        if ImageEncoderViTObserver.debug:
            attn_mean= attn_mean.softmax(dim=-1)
            attn_ori = attn_ori.softmax(dim=-1)
        output = (
            (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        )
        output = self.proj(output)

        if ImageEncoderViTObserver.debug:
            return output, attn, attn_mean, attn_ori, q_hat, k_hat, v_hat, q_mean, v_mean
        return output, attn, None, None, q_hat, k_hat, v_hat, q_mean, v_mean

    @staticmethod
    def clear_dict():
        """Clear the attention score dictionary."""
        AttentionObserver.attention_score = defaultdict(list)


class BlockObserver(Block):
    """
    Observer wrapper for SAM encoder Block (Transformer blocks).

    Extends Block to return attention scores and intermediate values
    for debugging and analysis purposes.
    """

    attention_dict = {}
    debug = False

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention tracking.

        Args:
            x: Input tensor of shape (B, H, W, C)

        Returns:
            Tuple of (output, attn, q, k, v)
        """

        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        # x, attn, q, k, v = self.attn(x)
        x, attn, attn_mean, attn_ori, q_hat, k_hat, v_hat, q_mean, v_mean  = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return  x, attn, attn_mean, attn_ori, q_hat, k_hat, v_hat, q_mean, v_mean 


class ImageEncoderViTObserver(ImageEncoderViT):
    """
    Observer wrapper for ImageEncoderViT that tracks attention scores.

    This class extends ImageEncoderViT to capture and store attention scores,
    queries, keys, and values from all attention layers for analysis.
    """

    attention_score = defaultdict(list)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Forward pass with attention tracking.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Tuple of (output, interm_embeddings)
        """
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        interm_embeddings = []
        for idx, blk in enumerate(self.blocks):
            if ImageEncoderViTObserver.debug:
                ImageEncoderViTObserver.attention_score[f"block_{idx}_x"].append(to_numpy(x))
            x, attn, attn_mean, attn_ori, q_hat, k_hat, v_hat, q_mean, v_mean  = blk(x)

            # Store attention information
            if ImageEncoderViTObserver.debug:
                ImageEncoderViTObserver.attention_score[f"block_{idx}_attn"].append(to_numpy(attn))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_attn_mean"].append(to_numpy(attn_mean))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_attn_ori"].append(to_numpy(attn_ori))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_q_hat"].append(to_numpy(q_hat))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_k_hat"].append(to_numpy(k_hat))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_v_hat"].append(to_numpy(v_hat))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_q_mean"].append(to_numpy(q_mean))
                ImageEncoderViTObserver.attention_score[f"block_{idx}_v_mean"].append(to_numpy(v_mean))

            if blk.window_size == 0:
                interm_embeddings.append(x)

        x = self.neck(x.permute(0, 3, 1, 2))

        return x, interm_embeddings

    @staticmethod
    def clear_dict():
        """Clear the attention score dictionary."""
        ImageEncoderViTObserver.attention_score = defaultdict(list)
@torch.no_grad()
def quantize_weight_per_channel_absmax(w: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Quantize weights per output channel using absolute maximum scaling."""
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w
@torch.no_grad()
def quantize_activation_per_token_absmax(t: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Quantize activations per token using absolute maximum scaling."""
    t_shape = t.shape
    t = t.contiguous().view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t.view(t_shape)
class nnLinear_qkv_hat(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_weight = None
        self.quant_activation = None
        self.single_head_features = self.out_features // 3
        self.n_bits=8
    def quantize_weight(self, n_bits=8):
        # quantize Q only 
        q_weight = self.weight[:, :]
        q_weight_q = quantize_weight_per_channel_absmax(q_weight, n_bits=n_bits)
        with torch.no_grad():
            self.weight[:, :] = q_weight_q
        # self.weight = 

    def forward(self, x,q_scales):
        """
        Forward pass that applies scaling to input before Q matrix multiplication.
        
        Args:
            x: Input tensor of shape (B, seq_len, in_features) where seq_len = H*W
            
        Returns:
            Output tensor of shape (B, seq_len, out_features)
        """
        # Get dimensions
     
        
        # Split weight into Q, K, V matrices
        # out_features = 3 * head_dim * num_heads
        
            
        
        q_weight = self.weight[:self.single_head_features, :]  # Q matrix weights
        k_weight = self.weight[self.single_head_features:2*self.single_head_features, :]  # K matrix weights  
        v_weight = self.weight[2*self.single_head_features:, :]  # V matrix weights
        
       
        x_scaled = x / q_scales.view(1, 1, -1)  # Broadcasting across batch and sequence dims
        if self.quant_activation == "per_token":
            x_scaled = quantize_activation_per_token_absmax(x_scaled, n_bits=self.n_bits)
        q_output = torch.nn.functional.linear(x_scaled, q_weight)

        x = quantize_activation_per_token_absmax(x, n_bits=self.n_bits)
        # K and V use original input (no scaling)
        k_output = torch.nn.functional.linear(x, k_weight)
        v_output = torch.nn.functional.linear(x, v_weight)
        
        # Concatenate Q, K, V outputs along the feature dimension
        output = torch.cat([q_output, k_output, v_output], dim=-1)
        
        # Add bias if exists
        if self.bias is not None:
            output = output + self.bias
            
        return output
    

def image_encoder_monkey_patch(
    model,
    processor=None,
    n_bits=8,
    weight_quant="per_channel",
    act_quant="per_token",
    layers=[0,5, 17],
    debug=False,
    device="cuda",
    path_stat_dict=None,
    percent=50,
    qkT_v=False,
    quarot = False,
    rot_args=None
):
    """
    Apply monkey-patching to SAM image encoder for quantization and observation.

    Args:
        model: SAM model to patch
        processor: Processing strategy for activations
        n_bits: Number of bits for quantization
        weight_quant: Weight quantization strategy
        k_preserve: Number of channels to preserve in selective quantization
    """
    # Replace classes with observer versions using monkey patching
    layer_idx =0
    ImageEncoderViTObserver.debug = debug 
    if quarot:
        from quarot import rotation_utils
        Q_image_encoder =rotation_utils.get_orthogonal_matrix(rot_args.hidden_size_image_en,rot_args.rotate_mode,device = rot_args.device,seed=rot_args.seed)
    if path_stat_dict is not None:
        # load stat_dict -- which is processor.stat is dictionary  {key: layer name , value : {'inputqkv':..., 'qkv':..., 'act_scales':...} }
        print("Loading existing statistics from file.")
        stat_dict = torch.load(path_stat_dict)
    else:
        # save stat_dict to the file and exit()
        stat_path="/home/ubuntu/21chi.nh/Quantization/SAM_Quantization/SAM_Quantization/pretrained_checkpoint/stat_dict.pth"
        os.makedirs(os.path.dirname(stat_path), exist_ok=True)
        stat_dict = processor.stat
        torch.save(stat_dict, stat_path)
        print(f"Statistics saved to {stat_path}")
        exit()
    print("Replacing attention of image_encoder with center Q with quantization that attention ")
    for name, module in model.named_modules():
        if isinstance(module, Attention) or isinstance(module, Attention_) or isinstance(module, Attention__):

            module.__class__ = AttentionObserver
            key ='.'.join(name.split('.')[1:])
            module.stat =   stat_dict[key]  
            module.n_bits = n_bits
            module.name = name
            module.percent = percent
            module.qkT_v = qkT_v
            if quarot:
                module.Q = Q_image_encoder
            # Create qkv_w_hat manually
            module.qkv_w_hat = nnLinear_qkv_hat(
                module.qkv.in_features, 
                module.qkv.out_features, 
                bias=module.qkv.bias is not None
            ).to(device)
            module.qkv_w_hat.n_bits = n_bits
            module.qkv_w_hat.weight.data = module.qkv.weight.data.clone().to(device)
            if module.qkv.bias is not None:
                module.qkv_w_hat.bias.data = module.qkv.bias.data.clone().to(device)
            module.smooth()
            module.qkv_w_hat.quant_activation =act_quant
            if weight_quant == "per_channel":
                module.qkv_w_hat.quant_weight = weight_quant
                module.qkv_w_hat.quantize_weight(n_bits=n_bits)
        
            # import ipdb; ipdb.set_trace()
        if isinstance(module, Block) or isinstance(module, block_) or isinstance(module, Block__):
            module.__class__ = BlockObserver
        if isinstance(module, ImageEncoderViT) or isinstance(module, ImageEncoderViT_) or isinstance(module, ImageEncoderViT__):
            if  layer_idx in layers:
                module.__class__ = ImageEncoderViTObserver
            layer_idx += 1

    modules_to_exclude = [
        "pos_embed",
        "cls_token",
        "patch_embed",
        "neck",
        "fpn",
        "rel_pos_h",
        "rel_pos_w",
    ]

    # config = QuantizationConfig(
    #     n_bits_w=n_bits,
    #     n_bits_a=n_bits,
    #     weight_quant=weight_quant,
    #     act_quant=act_quant,
    #     quantize_output=False,
    # )
    # replace_linear_with_quantized(
    #     module=model.image_encoder,
    #     config=config,
    #     module_name_to_exclude=modules_to_exclude,
    # )


# ============================================================================
# Main Execution
# ============================================================================


if __name__ == "__main__":
    # Configuration
    model_type = "vit_l"
    num_calib_samples = 1
    checkpoint_path = "./pretrained_checkpoint/sam_hq_vit_l.pth"

    # Initialize model and predictor
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to("cuda")
    predictor = SamPredictor(sam)

    # Setup processor with calibration for image encoder
    processor = ImageEncoderProcessor("encoder_attn")
    processor.calibrate(
        predictor=predictor,
        modules=(Block,),
        num_samples=num_calib_samples,
    )
    # Apply quantization with monkey-patching
    image_encoder_monkey_patch(
        predictor.model,
        processor=processor,
        n_bits=4,
        weight_quant="per_channel",
        act_quant="per_token",
    )
    # Run inference
    results = inference_image(
        predictor,
        image_dir="./input_imgs/",
        example_idx=3,
        show_image=True,
    )

    # Access attention scores
    print("\nCaptured attention information:")
    for key in ImageEncoderViTObserver.attention_score.keys():
        print(f"  {key}: {len(ImageEncoderViTObserver.attention_score[key])} items")

    print("\nImage encoder quantization completed successfully!")