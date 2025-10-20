import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from encoder_quant import (
    image_encoder_monkey_patch, 
    ImageEncoderViTObserver
) 
from utils import get_activation_boxplot, to_numpy,  inference_image
from segment_anything import SamPredictor, sam_model_registry
import matplotlib.pyplot as plt
from segment_anything.modeling.image_encoder import (
    Attention,
    Block,
    ImageEncoderViT,
)
from quant_utils import ImageEncoderProcessor
from segment_anything.modeling.image_encoder import (
    window_partition,
    window_unpartition,
)
from segment_anything.modeling.image_encoder import add_decomposed_rel_pos
from quant_utils_ import EncoderAttentionProcessorSmoothMeanQ
from segment_anything.modeling.transformer import  Attention as  DecoderAttention
from train.segment_anything_training.modeling.image_encoder import Attention as EncoderAttentionTraining
from seginw.segment_anything.modeling.image_encoder import Attention as EncoderAttention

model_type = 'vit_l'
checkpoint_path= './pretrained_checkpoint/sam_hq_vit_l.pth'
sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')



processor = EncoderAttentionProcessorSmoothMeanQ('attn') 
# processor = DoNothingProcessor('attn') 
# processor = SignProcessor('attn') 
# processor.calibrate(
    # predictor=predictor, 
    # modules=(),
    # num_samples=1,
# )
predictor = SamPredictor(sam)
processor.calibrate(predictor=predictor,
                modules=(DecoderAttention, EncoderAttentionTraining, EncoderAttention),
                num_samples=num_calib_samples
            )
image_encoder_monkey_patch(predictor.model, processor, n_bits=8, debug=True)
inference_image(predictor, image_dir='./input_imgs/', example_idx=2, show_image=True)