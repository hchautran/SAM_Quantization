# %%
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Callable, List
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from functools import partial
from PIL import Image
from matplotlib import pyplot as plt
from vis_utils import show_points, show_mask_video, show_masks_image



class ObserverBase:
    dictionary = {}
    def __init__(self, module_list:Tuple):
        self.module_list = module_list
        self.hooks = []

    def register_hooks(self,
        model:nn.Module,
        pre_hook:Optional[Callable]=None,
        post_hook:Optional[Callable]=None,
    ):
        for name, module in model.named_modules():
            if post_hook is not None:
                if isinstance(module, self.module_list):
                    self.hooks.append(
                        module.register_forward_hook(partial(post_hook, name=name))
                    )
            if pre_hook is not None:
                if isinstance(module, self.module_list):
                    self.hooks.append(
                        module.register_forward_pre_hook(partial(pre_hook, name=name))
                    )

    def clear_hook(self):
        for hook in self.hooks:
            hook.remove()

    def clear_dict(self):
        ObserverBase.dictionary = {}

    def add_prompt(
        self,
        predictor:SAM2VideoPredictor,
        inference_state,
        points,
        labels,
        video_dir:Optional[str]=None,
        frame_names:Optional[List[str]]=None,
        ann_frame_idx:int=0,
        ann_obj_id:int=1,
        show_point:bool=False,
    ):

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        # show the results on the current (interacted) frame
        if (
            show_point and
            isinstance(video_dir, str) and
            isinstance(frame_names, list) and
            isinstance(frame_names[ann_frame_idx], str)
        ):
            image = Image.open(os.path.join(video_dir, frame_names[ann_frame_idx]))
            plt.figure(figsize=(9, 6))
            plt.title(f"frame {ann_frame_idx}")
            plt.imshow(image)
            show_points(points, labels, plt.gca())
            show_mask_video((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])



    @torch.inference_mode()
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def inference_video(
        self,
        predictor:SAM2VideoPredictor,
        video_dir:str='./notebooks/videos/bedroom',
        show_video:bool=False
    ):
        inference_state=predictor.init_state(video_path=video_dir)
        predictor.reset_state(inference_state)

        frame_names = self.get_frame_names(video_dir)

        self.add_prompt(
            predictor=predictor,
            inference_state=inference_state,
            points=np.array([[210, 350], [250, 220]], dtype=np.float32),
            labels=np.array([1, 1], np.int32),
            ann_frame_idx=0,
            ann_obj_id=1,
        )
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        if show_video:
            self.show_video_masks(video_dir, video_segments, frame_names)

    def get_frame_names(self, video_dir):
        frame_names = [
            p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in ['.jpg', '.jpeg', '.JPG', '.JPEG']
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        return frame_names

    def show_video_masks(self, video_dir:str, video_segments:dict, frame_names:List[str], vis_frame_stride:int=30 ):
        # render the segmentation results every few frames
        # vis_frame_stride = 30
        plt.close("all")
        for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask_video(out_mask, plt.gca(), obj_id=out_obj_id)



    @torch.inference_mode()
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def inference_image(
        self,
        predictor:SAM2ImagePredictor,
        image_dir:str='./notebooks/images/cars.jpg',
        show_image:bool=False,
    ):
        image = Image.open(image_dir)
        predictor.set_image(image)
        point_coords = np.array([[500, 375]])
        point_labels = np.array([1])

        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        if show_image:
            sorted_idx = np.argsort(scores)[::-1]
            masks = masks[sorted_idx]
            scores = scores[sorted_idx]
            logits = logits[sorted_idx]
            show_masks_image(
                image,
                masks,
                scores,
                point_coords=point_coords,
                input_labels=point_labels,
                borders=True
            )

# %%
