# %%

import os
import yaml
import cv2
import torch
from engine import InferenceStrategy, Engine
import numpy as np
import matplotlib.pyplot as plt
from utils.sam_vis_utils import show_res_multi

from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator
from utils.coco import CocoDetection, PostProcessSeginw
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# segment anything
from seginw.segment_anything import (
    build_sam,
    build_sam_hq,
    SamPredictor
)
import cv2
import json
import time



class SamInferenceStrategy(InferenceStrategy):
    def __init__(self, sam_config:dict):
        self.model_type = sam_config['model_type']
        self.use_sam_hq = sam_config['use_hq']
        if sam_config['use_hq']:
            self.sam_ckt = sam_config['hq_checkpoint']
        else:
            self.sam_ckt = sam_config['checkpoint']
        self.device = torch.device(sam_config['device'])
        self.predictor = None
        self.image = None

    def build_predictor(self)->SamPredictor:
        if self.use_sam_hq:
            self.predictor = SamPredictor(build_sam_hq(checkpoint=self.sam_ckt).to(self.device))
        else:
            self.predictor = SamPredictor(build_sam(checkpoint=self.sam_ckt).to(self.device))
        return self.predictor


    def set_image(self, image_dir:str):
        image = cv2.imread(image_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        self.predictor.set_image(image)

    def set_video(self, video_dir:str):
        raise NotImplementedError("Video inference is not supported for SAM")
        
    @torch.inference_mode()
    @torch.no_grad()
    def inference(self, inputs:dict, use_torch:bool=False):
        if not use_torch:
            masks, scores, logits = self.predictor.predict(**inputs)
        else:
            masks, scores, logits = self.predictor.predict_torch(**inputs)
        return masks, scores, logits

        
    def visualize(self, prompts:dict, masks:torch.Tensor, scores:torch.Tensor, result_path:str):
        show_res_multi(masks, scores, prompts['point_coords'], prompts['point_labels'], prompts['box'], result_path, self.image)
        


class SamEngine(Engine):
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
    
    def load_model(self, model_config_path: str, model_checkpoint_path: str): 
        args = SLConfig.fromfile(model_config_path)
        args.device = self.strategy.device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model

    def evaluate_seginw(self, seginw_config:dict):
        objects = os.listdir(seginw_config['data_path'])
        cfg = SLConfig.fromfile(seginw_config['config_file'])
        # build model
        model = self.load_model(seginw_config['config_file'], seginw_config['checkpoint_path'])
        model = model.to(self.strategy.device)
        model = model.eval()

        # build dataloader
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        for object in objects:
            image_dir = os.path.join(seginw_config['data_path'], object, 'valid')
            anno_path = os.path.join(seginw_config['data_path'], object, 'valid', '_annotations_min1cat.coco.json')
            dataset = CocoDetection(
                image_dir,
                anno_path, 
                transforms=transform
            )
            data_loader = DataLoader(
                dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=seginw_config['num_workers'], 
                collate_fn=collate_fn
            )

        # build post processor
            tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
            postprocessor = PostProcessSeginw(num_select=seginw_config['num_select'],coco_api=dataset.coco, tokenlizer=tokenlizer)

            # build evaluator
            evaluator = CocoGroundingEvaluator(
                dataset.coco, iou_types=("bbox","segm"), useCats=True)

            # build captions
            category_dict = dataset.coco.dataset['categories']
            cat_list = [item['name'] for item in category_dict]
            caption = " . ".join(cat_list) + ' .'
            print("Input text prompt:", caption)


            predictor = self.strategy.build_predictor()

            
            json_file = []
            start = time.time()
            progress_bar = tqdm(data_loader, desc=f"Evaluating {object}", total=len(data_loader))
            for i, (images, targets) in enumerate(data_loader):
                # get images and captions
                images = images.tensors.to(self.strategy.device)
                bs = images.shape[0]
                assert bs == 1
                input_captions = [caption] * bs

                # feed to the model
                outputs = model(images, captions=input_captions)
                orig_target_sizes = torch.stack(
                    [t["orig_size"] for t in targets], dim=0).to(images.device)
                results = postprocessor(outputs, orig_target_sizes)
                self.strategy.set_image(image_dir=f'{image_dir}/{targets[0]["file_path"]}')

                input_boxes = results[0]['boxes'].cpu()     
                transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, self.strategy.image.shape[:2]).to(self.strategy.device)
                prompts = {
                    'point_coords': None, 
                    'point_labels': None,
                    'boxes': transformed_boxes,
                    'hq_token_only': True,
                }
                masks, _, _ = self.strategy.inference(prompts, use_torch=True)
                results[0]['masks'] = masks.cpu().numpy()

                cocogrounding_res = {
                    target["image_id"]: output for target, output in zip(targets, results)}
                
                save_items = evaluator.update(cocogrounding_res)

                if seginw_config['save_json']:
                    new_items = list()
                    for item in save_items:
                        new_item = dict()
                        new_item['image_id'] = item['image_id']
                        new_item['category_id'] = item['category_id']
                        new_item['segmentation'] = item['segmentation']
                        new_item['score'] = item['score']
                        new_items.append(new_item)

                    json_file = json_file + new_items

                if (i+1) % 30 == 0:
                    used_time = time.time() - start
                    eta = len(data_loader) / (i+1e-5) * used_time - used_time
                    print(
                        f"processed {i}/{len(data_loader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s")
                progress_bar.update(1)




            evaluator.synchronize_between_processes()
            evaluator.accumulate()
            evaluator.summarize()
            print("Final results:", evaluator.coco_eval["segm"].stats.tolist())

            if seginw_config['save_json']:
                if self.strategy.use_sam_hq:
                    os.makedirs('seginw_output/sam_hq/', exist_ok=True)
                    save_path = 'seginw_output/sam_hq/seginw-'+anno_path.split('/')[-3]+'_val.json'
                else:
                    os.makedirs('seginw_output/sam/', exist_ok=True)
                    save_path = 'seginw_output/sam/seginw-'+anno_path.split('/')[-3]+'_val.json'
                with open(save_path,'w') as f:
                    json.dump(json_file,f)
                print(save_path)
                

# %%

if __name__ == "__main__":
    import yaml

    with open('config/base_h.yaml', 'r') as f:
        config = yaml.safe_load(f)

    engine = SamEngine(SamInferenceStrategy(config['model']))
    engine.evaluate_seginw(config['seginw'])
    
    prompts = {
        'point_coords': None, 
        'point_labels': None,
        'box': np.array([[4,13,1007,1023]]),
        'hq_token_only': True,
    }
    # image = Image.open('../input_imgs/example0.png')
    engine.demo(prompts, image_dir='../input_imgs/example1.png', show_image=True)

# %%

