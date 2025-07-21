import os
from PIL import Image
import matplotlib.pyplot as plt
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.memory_attention import MemoryAttention
from sam2.modeling.memory_encoder import MemoryEncoder
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from torch.profiler import profile,  ProfilerActivity, record_function
from sam2.build_sam import build_sam2_video_predictor
from observer import ObserverBase
from vis_utils import show_points, show_mask_video
from const import *
from typing import List
import torch



class Sam2MemProfiler(ObserverBase):
    def __init__(self, checkpoint:str, model_cfg:str):
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    def profile(self, video_path):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.inference_video(self.predictor, video_path, show_video=False)
        peak_mem = torch.cuda.max_memory_allocated()
        ObserverBase.dictionary['peak_mem'] = peak_mem
        print(f'PeakMem: {peak_mem/1024**3:.4f}, GiB')

        return peak_mem




if __name__ == '__main__':
    ckt_path = f'{SAM2_CKT_PATH}/{SAM_2_L}'
    config_path = f'{SAM2_CFG_PATH}/{SAM_2_L_CFG}'
    profiler = Sam2MemProfiler(checkpoint=ckt_path, model_cfg=config_path)
    print(profiler.profile(SAMPLE_VIDEO_PATH))
