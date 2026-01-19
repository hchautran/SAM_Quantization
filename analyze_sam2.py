# %%
import torch
import torch.nn as nn
from torch import Tensor
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.memory_encoder import MemoryEncoder
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.sam.prompt_encoder import PromptEncoder
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.position_encoding import apply_rotary_enc, compute_axial_cis
from sam2.modeling.sam2_utils import MLP
import torch.nn.functional as F
from collections import defaultdict
from typing import Tuple, Optional
import math



class ModuleProfiler:
    def __init__(self, modules:Tuple):
        self.modules=modules
        self.times = defaultdict(float)
        self.records = defaultdict(list)
        self.average = defaultdict(float)
        self.total = defaultdict(float)
        self.starter = torch.cuda.Event(enable_timing=True)
        self.ender = torch.cuda.Event(enable_timing=True)
        self.hooks = []


    def profile(self, model:nn.Module):
        def pre_forward_hook(module:nn.Module, input ):
            module_name = module.__class__.__name__
            self.starter.record()

        def post_forward_hook(module:nn.Module, input, output):
            module_name = module.__class__.__name__
            self.ender.record()
            torch.cuda.synchronize()
            inference_time = self.starter.elapsed_time(self.ender)
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
    
    def get_plot_for_all_frame(self):
        import matplotlib.pyplot as plt
        if not self.records:
            return
        plt.figure(figsize=(10, 6))
        for module, timings in self.records.items():
            if not timings:
                continue
            steps = list(range(len(timings)))
            values = [t * 1000 for t in timings]
            plt.plot(steps, values, label=module)
        plt.title('Module latency per call', fontsize=14, fontweight='bold')
        plt.xlabel('Call index')
        plt.ylabel('Latency (ms)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('module_latency_lineplot.png', dpi=300)
        plt.show()

    def get_profile_plot(self):
        # import squarify
        import matplotlib.pyplot as plt
        import numpy as np
        sizes = [sum(self.records[module])/len(self.records[module])*1000 for module in self.records.keys()]
        labels = [module for module in self.records.keys()]

        colors = plt.cm.Pastel1(np.linspace(0, 1, len(sizes)))

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Create treemap
        # squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8, ax=ax)
        ax.set_title('Module latency profile', fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('module_latency_profile.png', dpi=300)
        plt.show()
    



    def clear(self):
        for hook in self.hooks:
            hook.remove()


class MemoryAttentionLayerObserver(MemoryAttentionLayer):
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent Attention class."""
        super().__init__(*args, **kwargs)
    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:

        # Self-Attn, Cross-Attn
        print("memory shape", memory.shape)
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
class RoPEAttentionObserver(RoPEAttention):
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent Attention class."""
        super().__init__(*args, **kwargs)
    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class MemoryEncoderObserver(MemoryEncoder):
    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as parent Attention class."""
        super().__init__(*args, **kwargs)
    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ## Process masks
        # sigmoid, so that less domain shift from gt masks which are bool
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        ## Fuse pix_feats and downsampled masks
        # in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)
        
        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)
        return {"vision_features": x, "vision_pos_enc": [pos]}
def monkey_patch(model):
    for name, module in model.named_modules():
        if isinstance(module, (MemoryAttentionLayer)):
            module.__class__ = MemoryAttentionLayerObserver
        if isinstance(module, (RoPEAttention)):
            module.__class__ = RoPEAttentionObserver
        if isinstance(module,MemoryEncoder):
            module.__class__ = MemoryEncoderObserver
            

# %%
import os
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
checkpoint = './sam2_ckts/sam2.1_hiera_base_plus.pt'
model_cfg = 'configs/sam2.1/sam2.1_hiera_b+.yaml'

# checkpoint= "./sam2_ckts/sam2.1_hiera_large.pt"
# model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'

# checkpoint= "./sam2_ckts/sam2.1_hiera_small.pt"
# model_cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'

# checkpoint= "./sam2_ckts/sam2.1_hiera_tiny.pt"
# model_cfg = 'configs/sam2.1/sam2.1_hiera_t.yaml'

video_dir = './data/bedroom'

predictor = build_sam2_video_predictor(model_cfg, checkpoint)
print(predictor)
monkey_patch(predictor)

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
# %%
predictor.reset_state(inference_state)

# %%
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor
import numpy as  np
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

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
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
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
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
# %%
print(profiler.get_results())
profiler.get_profile_plot()
profiler.get_plot_for_all_frame()
profiler.clear()


# %%