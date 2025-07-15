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
from typing import List



class Sam2MemProfiler(ObserverBase):
    def __init__(self, checkpoint:str, model_cfg:str):
        self.checkpoint = checkpoint
        self.model_cfg = model_cfg
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint)

    def profile(self, video_path:str):
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], with_flops=True,profile_memory=True, record_shapes=True, with_modules=True, with_stack=True) as prof:
            self.inference_video(self.predictor, video_path, show_video=False)
        print(prof.key_averages().table(sort_by='self_cuda_memory_usage', row_limit=20))
        prof.export_chrome_trace("trace.json")



if __name__ == '__main__':
    ckt_path = '../sam2_ckts/sam2.1_hiera_small.pt'
    config_path = '../sam2/configs/sam2.1/sam2.1_hiera_s.yaml'
    video_dir = '../sam2/notebooks/videos/bedroom'
    profiler = Sam2MemProfiler(checkpoint=ckt_path, model_cfg=config_path)
    profiler.profile(video_dir)
