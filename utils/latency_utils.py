# %%
import torch.nn as nn
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.memory_attention import MemoryAttention
from sam2.modeling.memory_encoder import MemoryEncoder
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder

import time
from collections import defaultdict
from typing import Tuple




class ModuleProfiler:
    def __init__(self, modules:Tuple):
        self.modules=modules
        self.times = defaultdict(float)
        self.records = defaultdict(list)
        self.average=defaultdict(float)
        self.total=defaultdict(float)
        self.hooks = []


    def profile(self, model:nn.Module):
        def pre_forward_hook(module:nn.Module, input ):
            module_name = module.__class__.__name__
            self.times[module_name] = time.time()

        def post_forward_hook(module:nn.Module, input, output):
            module_name = module.__class__.__name__
            inference_time = time.time() - self.times[module_name]
            self.records[module_name].append(inference_time)

        for name, module in model.named_modules():
            if isinstance(module, self.modules):
                self.hooks.extend([
                    module.register_forward_pre_hook(pre_forward_hook),
                    module.register_forward_hook(post_forward_hook),
                ])

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



    def clear(self):
        for hook in self.hooks:
            hook.remove()



# %%
import os
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
project_dir = ''
checkpoint = '../sam2_ckts/sam2.1_hiera_small.pt'
model_cfg = '../sam2/configs/sam2.1/sam2.1_hiera_s.yaml'
video_dir = '../sam2/notebooks/videos/bedroom'

predictor = build_sam2_video_predictor(model_cfg, checkpoint)

frame_names = [
    p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in ['.jpg', '.jpeg', '.JPG', '.JPEG']
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
frame_idx = 0
plt.figure(figsize=(9,6))
plt.title(f'frame {frame_idx}')
plt.imshow(plt.imread(os.path.join(video_dir, frame_names[frame_idx])))


# %%
inference_state = predictor.init_state(video_path=video_dir)
# %%
print(type(predictor))
# %%
print(type(inference_state))
predictor.reset_state(inference_state)

# %%
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor
import numpy as  np
from vis_utils import show_points, show_mask_video

def add_prompt(
    predictor:SAM2VideoPredictor, inference_state, points, labels, ann_frame_idx:int=0, ann_obj_id:int=1, show_point:bool=False):

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # show the results on the current (interacted) frame
    if show_point:
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(points, labels, plt.gca())
        show_mask_video((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
# %%
module_list = (Hiera, MemoryEncoder, MemoryAttention, MaskDecoder, PromptEncoder)
profiler = ModuleProfiler(module_list)
profiler.profile(predictor)
add_prompt(
    predictor=predictor,
    inference_state=inference_state,
    points=np.array([[210, 350], [250, 220]], dtype=np.float32),
    labels=np.array([1, 1], np.int32),
    ann_frame_idx=0,
    ann_obj_id=1,
    show_point=True
)
profiler.get_results()


# %%
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask_video(out_mask, plt.gca(), obj_id=out_obj_id)
# %%
print(profiler.get_results())
profiler.get_profile_plot()
profiler.clear()


# %%
