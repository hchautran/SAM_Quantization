# %%
import os
import time
import numpy as  np
from observer import ObserverBase
import torch.nn as nn
from PIL import Image
from typing import Tuple
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.memory_attention import MemoryAttention
from sam2.modeling.memory_encoder import MemoryEncoder
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.sam2_video_predictor import SAM2VideoPredictor
from vis_utils import show_points, show_mask_video
from collections import defaultdict




class LatencyObserver(ObserverBase):
    def __init__(self, module_list:Tuple):
        super(LatencyObserver, self).__init__(module_list)
        self.times = defaultdict(float)
        self.records = defaultdict(list)
        self.average=defaultdict(float)
        self.total=defaultdict(float)


    def register_latency_hooks(self, model:nn.Module):
        def pre_forward_hook(module:nn.Module, input, name ):
            module_name = module.__class__.__name__
            self.times[module_name] = time.time()

        def post_forward_hook(module:nn.Module, input, output, name):
            module_name = module.__class__.__name__
            inference_time = time.time() - self.times[module_name]
            self.records[module_name].append(inference_time)

        self.register_hooks(
            model,
            pre_forward_hook,
            post_forward_hook,
        )


    def get_results(self):
        results = defaultdict(str)
        for module in self.records.keys():
            results[module] = f'{sum(self.records[module])/len(self.records[module])*1000:.3f} ms'
        return results

    def get_profile_plot(self):
        import squarify
        import matplotlib.pyplot as plt
        import numpy as np
        sizes = [sum(self.records[module])/len(self.records[module])*1000 for module in self.records.keys()]
        labels = [module for module in self.records.keys()]
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(sizes)))
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        # Create treemap
        squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, ax=ax)
        ax.set_title('Module latency profile', fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('module_latency_profile.png', dpi=300)
        plt.show()






# %%
if __name__ == '__main__':
    from const import *
    ckt_path = f'{SAM2_CKT_PATH}/{SAM_2_L}'
    config_path = f'{SAM2_CFG_PATH}/{SAM_2_L_CFG}'
    video_dir = SAMPLE_VIDEO_PATH



    predictor = build_sam2_video_predictor(config_path, ckt_path)

    module_list = (Hiera, MemoryEncoder, MemoryAttention, MaskDecoder, PromptEncoder)
    profiler = LatencyObserver(module_list)
    profiler.register_latency_hooks(predictor)
    profiler.inference_video(predictor, video_dir)
    print(profiler.get_results())
    profiler.get_profile_plot()



    profiler.clear_hook()
    profiler.clear_dict()
