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
    Attention as EncoderAttention,
    Block as EncoderBlock,
    ImageEncoderViT,
)
# from seginw.segment_anything.modeling.image_encoder import Attention as EncoderAttention 
# from seginw.segment_anything.modeling.image_encoder import Block as EncoderBlockTraining 
# from seginw.segment_anything.modeling.image_encoder import ImageEncoderViT 
# from train.segment_anything_training.modeling.image_encoder import Attention as EncoderAttentionTraining
# from train.segment_anything_training.modeling.image_encoder import Block as EncoderBlockTraining 
# from train.segment_anything_training.modeling.image_encoder import ImageEncoderViT as ImageEncoderViTTraining
# Local imports
from processors.base import (
    # ImageEncoderProcessor,
    AttentionProcessor,
)
from processors.encoder import (
    # ImageEncoderProcessor,
    EncoderRecenterAttentionProcessor,
)
from utils.quant_utils import quantize_activation_per_token_absmax
from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
from RTN_quantization import per_tensor_channel_group
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'RTN_quantization'))
from quant.quant_utils import replace_linear_with_quantized, QuantizationConfig
from RTN_quantization.per_tensor_channel_group import quantize_activation_low_high_density_activation_index
# from utils import inference_image, to_numpy
from segment_anything.modeling.image_encoder import (
    window_partition,
    window_unpartition,
)
from utils.utils import inference_image

def to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()





    

class QuantizedAttention(EncoderAttention):
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent Attention class."""
        super().__init__(*args, **kwargs)

    def set_processor(self, processor:AttentionProcessor, module_name):
        self.processor = processor
        self.module_name = module_name 

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.processor.process(x, self, self.module_name)



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
    

class QuantizedAttention(EncoderAttention):
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent Attention class."""
        super().__init__(*args, **kwargs)

    def set_processor(self, processor:AttentionProcessor, module_name):
        self.processor = processor
        self.module_name = module_name 
    

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.processor.process(x, self, self.module_name)


class QuantizedAttentionQuarotCenterQ(QuantizedAttention):
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent Attention class."""
        super().__init__(*args, **kwargs)
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
    def _take_qkv_w_hat(self, args = None):
        device =  args.quarot_inf.device
        key ='.'.join(self.module_name.split('.')[1:])
        self.stat =   self.processor.stat[key]
        self.qkv_w_hat = nnLinear_qkv_hat(
                self.qkv.in_features, 
                self.qkv.out_features, 
                bias=self.qkv.bias is not None
            ).to(device)
        self.qkv_w_hat.n_bits = args.rtn_ro_config.n_bits
        self.qkv_w_hat.weight.data = self.qkv.weight.data.clone().to(device)
        self.qkv_w_hat.bias.data = self.qkv.bias.data.clone().to(device)
        self.smooth()
        self.qkv_w_hat.quant_activation =args.quantization.act_quant
        self.qkv_w_hat.quant_weight = args.quantization.weight_quant
        self.qkv_w_hat.quantize_weight(n_bits=args.rtn_ro_config.n_bits)
        self.percent = args.quantization.percent

    
def Select_quantizedAttention_encoder(args):
    if args.quantization.quanro:
        if args.quantization.centerQ:
            return QuantizedAttentionQuarotCenterQ
    return QuantizedAttention

def image_encoder_monkey_patch(
    model,
    processor=None,
    n_bits=8,
    weight_quant="per_channel",
    act_quant="per_token",
    layers=[0,5, 17],
    debug=False,
    device="cuda",
    args_yaml=None,
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
    # ImageEncoderViTObserver.debug = debug 
    
    Quantize_attention = Select_quantizedAttention_encoder(args_yaml)
   
    for name, module in model.named_modules():
        if isinstance(module, (EncoderAttention)):
            module.__class__ = Quantize_attention
            module.set_processor(processor, name)
      
        # if isinstance(module, Block) or isinstance(module, block_) or isinstance(module, Block__):
        #     module.__class__ = BlockObserver
        # if isinstance(module, ImageEncoderViT) or isinstance(module, ImageEncoderViT_) or isinstance(module, ImageEncoderViT__):
        #     if  layer_idx in layers:
        #         module.__class__ = ImageEncoderViTObserver
        #     layer_idx += 1
    
    
    if not args_yaml.quantization.quanro and n_bits < 16: # already quantized in the rotation process
        
        modules_to_exclude = ['decoder'] # Quantize Encoder only
        config = QuantizationConfig(
            n_bits_w=n_bits,
            n_bits_a=n_bits,
            weight_quant=weight_quant,
            act_quant=act_quant,
            quantize_output=False,
        )
        replace_linear_with_quantized(
            module=model.image_encoder,
            config=config,
            module_name_to_exclude=modules_to_exclude,
        )
    # move model to device
    model.to(device)
    

# ============================================================================
# Main Execution
# ============================================================================


if __name__ == "__main__":
    from quant_utils import (
    # ImageEncoderProcessor,
    AttentionProcessor,
    EncoderRecenterAttentionProcessor,
    EncoderHighLowAttentionProcessor,
    quantize_activation_per_token_absmax,
)
    # Configuration
    model_type = "vit_l"
    num_calib_samples = 1
    checkpoint_path = "./ckts/sam_hq_vit_l.pth"

    # Initialize model and predictor
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to("cuda")
    predictor = SamPredictor(sam)

    # Setup processor with calibration for image encoder
    # processor = EncoderHighLowAttentionProcessor("highlow")
    processor = EncoderRecenterAttentionProcessor("recenter")
    processor.calibrate(
        predictor=predictor,
        modules=(EncoderAttention,),
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

 