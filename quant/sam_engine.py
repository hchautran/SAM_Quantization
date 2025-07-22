# %%

import cv2
import torch
from engine import InferenceStrategy, SamEngine
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.sam_vis_utils import show_res_multi

class SamInferenceStrategy(InferenceStrategy):
    def __init__(self, model_type:str, sam_ckt:str, device:str):
        self.predictor = None
        self.model_type = model_type
        self.sam_ckt = sam_ckt
        self.device = device
        self.image = None

    def build_predictor(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_ckt)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)


    def set_image(self, image_dir:str):
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        self.predictor.set_image(image)

    def set_video(self, video_dir:str):
        raise NotImplementedError("Video inference is not supported for SAM")
        
    @torch.inference_mode()
    @torch.no_grad()
    def inference(self, inputs:dict):
        masks, scores, logits = self.predictor.predict(
            point_coords=inputs['point_coords'],
            point_labels=inputs['point_labels'],
            box = inputs['box'],
            multimask_output=False,
            hq_token_only=inputs['hq_token_only'], 
        )
        return masks, scores, logits

        
    def visualize(self, prompts:dict, masks:torch.Tensor, scores:torch.Tensor, result_path:str):
        show_res_multi(masks, scores, prompts['point_coords'], prompts['point_labels'], prompts['box'], result_path, self.image)
        


class HQSamEngine(SamEngine):
    def __init__(self, strategy:InferenceStrategy):
        super().__init__(strategy)
        self.strategy.build_predictor()

    def demo(self, prompts:dict, image_dir: torch.Tensor, show_image:bool= False):
        self.strategy.set_image(image_dir)
        masks, scores, logits = self.strategy.inference(prompts)
        if show_image:
            result_path = './demo.jpg' 
            self.strategy.visualize(prompts, masks, scores, result_path)
        return masks, scores, logits

    def images_inference(self, image_dir: torch.Tensor, show_image:bool= False):
        pass

# %%

if __name__ == "__main__":
    from PIL import Image
    
    model_type = 'vit_h'
    sam_ckt='../hq_ckts/sam_hq_vit_h.pth'
    device='cuda'
    prompts = {
        'point_coords': None, 
        'point_labels': None,
        'box': np.array([[4,13,1007,1023]]),
        'hq_token_only': True,
    }
    # image = Image.open('../input_imgs/example0.png')
    engine = HQSamEngine(SamInferenceStrategy(model_type=model_type, sam_ckt=sam_ckt, device=device))
    engine.demo(prompts, image_dir='../input_imgs/example1.png', show_image=True)

# %%

