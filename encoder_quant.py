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
from quant_utils import (
    # ImageEncoderProcessor,
    AttentionProcessor,
    EncoderRecenterAttentionProcessor,
    quantize_activation_per_token_absmax,
)
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
from utils import inference_image

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




def image_encoder_monkey_patch(
    model,
    processor=None,
    n_bits=8,
    weight_quant="per_channel",
    act_quant="per_token",
    layers=[0,5, 17],
    debug=False,
    device="cuda",
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
    for name, module in model.named_modules():
        if isinstance(module, (EncoderAttention)):
            module.__class__ = QuantizedAttention
            module.set_processor(processor, name)


    modules_to_exclude = [
        "pos_embed",
        "cls_token",
        "patch_embed",
        "neck",
        "fpn",
        "rel_pos_h",
        "rel_pos_w",
        "lin1",
        "lin2",
    ]

    if n_bits < 16:
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
    checkpoint_path = "./pretrained_checkpoint/sam_hq_vit_l.pth"

    # Initialize model and predictor
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to("cuda")
    predictor = SamPredictor(sam)

    # Setup processor with calibration for image encoder
    processor = EncoderHighLowAttentionProcessor("highlow")
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

 