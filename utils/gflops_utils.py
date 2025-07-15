from ast import Mult
from re import M
import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from torch.nn.modules import module
from fvcore.nn import FlopCountAnalysis
from torch.profiler.profiler import profile
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
from observer import ObserverBase
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.memory_attention import MemoryAttention
from sam2.modeling.memory_encoder import MemoryEncoder
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.sam.transformer import TwoWayTransformer


class GFLOPSCounter(ObserverBase):
    def __init__(self, module_list: Tuple):
        super(GFLOPSCounter, self).__init__(module_list)
        self.inputs = {}
        self.results = {}

    def register_profiling_hooks(self, model):

        def hook(module:nn.Module, input, output, name):
            self.inputs[module.__class__.__name__] = input

        self.register_hooks(model, post_hook=hook)

    def profile_image_inference(self, predictor:nn.Module, image_path:str):
        self.register_profiling_hooks(predictor.model)
        self.inference_image(predictor, image_path)
        self.clear_hook()


    def profile_video_inference(self, model:nn.Module, video_path:str):
        self.register_profiling_hooks(model)
        self.inference_video(model, video_path)
        self.clear_hook()

    def clear_input(self):
        self.inputs = {}

    def count_flops(self, module:nn.Module, input):
        flops = FlopCountAnalysis(module, input)
        print(module.__class__.__name__,flops.total()/1e9)

    @torch.no_grad()
    def profile(self, model_ckt:str, model_config:str, image_path=None, video_path=None):
        if image_path:
            sam2 = build_sam2(model_config, model_ckt)
            predictor= SAM2ImagePredictor(sam2)
            self.profile_image_inference(predictor, image_path)
            for name, module in predictor.model.named_modules():
                if isinstance(module, self.module_list):
                    self.count_flops(module, self.inputs[module.__class__.__name__])
            profiler.clear_input()
        if video_path:
            predictor = build_sam2_video_predictor(config_path, ckt_path)
            self.profile_video_inference(predictor, video_path)
            for name, module in predictor.named_modules():
                if isinstance(module, self.module_list):
                    self.count_flops(module, self.inputs[module.__class__.__name__])

            profiler.clear_input()





if __name__ == "__main__":
    ckt_path = '../sam2_ckts/sam2.1_hiera_large.pt'
    config_path = '../sam2/configs/sam2.1/sam2.1_hiera_l.yaml'
    video_path= '../sam2/notebooks/videos/bedroom'
    image_path = '../sam2/notebooks/images/cars.jpg'





    profiler = GFLOPSCounter(module_list=(Hiera, MemoryEncoder, TwoWayTransformer))
    profiler.profile(ckt_path, config_path,  video_path=video_path)
