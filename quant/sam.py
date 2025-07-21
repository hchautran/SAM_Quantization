import cv2
import torch
from engine import InferenceStrategy, SamEngine
from segment_anything import sam_model_registry, SamPredictor
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
