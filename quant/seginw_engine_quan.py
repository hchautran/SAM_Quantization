# %%

import os
import yaml
import cv2
import torch
from engine import InferenceStrategy, Engine
import numpy as np
from train.utils.dataloader import get_im_gt_name_dict, create_dataloaders, Resize
import matplotlib.pyplot as plt
from utils.sam_vis_utils import show_res_multi
from typing import Union, Optional, Dict
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator
from utils.coco import CocoDetection, PostProcessSeginw
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from seginw.segment_anything import (
    build_sam,
    build_sam_hq,
    build_sam_hq_vit_l,
    SamPredictor
)
import cv2
import json
import time
from omegaconf import OmegaConf

import os.path as osp
import argparse
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.registry import  DATA_SAMPLERS, FUNCTIONS, TRANSFORMS
from mmengine.dataset import worker_init_fn as default_worker_init_fn
from functools import partial
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils import digit_version
import copy

import mmdet
from mmdet import datasets, models, evaluation
from mmdet.datasets import transforms
from mmdet.registry import DATASETS,  TRANSFORMS as MMDET_TRANSFORMS
from mmdet.utils import setup_cache_size_limit_of_dynamo
# Register mmdet transforms in mmengine registry
for name, transform_cls in MMDET_TRANSFORMS.module_dict.items():
    if not TRANSFORMS.get(name):
        TRANSFORMS.register_module(module=transform_cls, name=name, force=True)

from configmmdet.utils_ import parse_args_test, parse_args_train
import logging
import ipdb
import sys
import importlib
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
quarot_path = os.path.join(project_root, 'quarot')
qgemm_dir = os.path.join(project_root, 'quant', 'qgemm')
rtn_cuda_dir = os.path.join(qgemm_dir, 'cuda_rtn_gptq') 
sys.path.insert(0, project_root)  
sys.path.insert(0, quarot_path) 
sys.path.insert(0, qgemm_dir)  
sys.path.insert(0, rtn_cuda_dir) 

from distribution_sam import get_channel_distribution_modify
import RTN_quantization.utils as rtn_utils
from RTN_quantization import per_tensor_channel_group,gptq_utils
import rotate_sam
from quantizer import replace_linear_with_int4 ,save_cuda_quantized_model, replace_linear_with_int4_gptq

def setup_logger(path_log,state):
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(path_log, f'{state}.log'))
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
def print_model_structure(model, title="Model Structure"):
    print(f"\n{title}")
    print("=" * len(title))
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")
    print("=" * len(title))
class SeginwInferenceStrategy(InferenceStrategy):
    def __init__(self, sam_config:dict):
        self.model_type = sam_config['model']['model_type']
        self.use_sam_hq = sam_config['model']['use_hq']
        if sam_config['model']['use_hq']:
            self.sam_ckt = sam_config['model']['hq_checkpoint']
        else:
            self.sam_ckt = sam_config['model']['checkpoint']
        # self.device = torch.device(sam_config['model']['device'])
        self.device = "cpu"
        self.predictor = None
        self.image = None
        
        self.quant_rtn = sam_config.quantization.quanrtn
        self.quant_gptq = sam_config.quantization.quangptq
        self.quant_smooth = sam_config.quantization.quansmooth
        self.quant_ro  = sam_config.quantization.quanro
        self.act_scales_file = sam_config.quantization.act_scales_file
        self.act_quant = sam_config.quantization.act_quant
        self.weight_quant = sam_config.quantization.weight_quant
        self.n_bits = sam_config.quantization.n_bits
        self.quantize_output = sam_config.quantization.quantize_output
        self.rtn_cuda = sam_config.quantization.rtn_cuda
        self.gptq_cuda = sam_config.quantization.gptq_cuda
        if self.quant_gptq or self.gptq_cuda:
            self.args_gptq = sam_config.gptq
        if self.rtn_cuda:
            self.save_rtn_cuda = sam_config.quantization.save_rtn_cuda
        if self.quant_gptq or self.gptq_cuda:
            self.args_gptq = sam_config.gptq
        if self.rtn_cuda:
            self.save_rtn_cuda = sam_config.quantization.save_rtn_cuda
        if self.quant_ro:
            self.rot_args = sam_config.quarot_inf
    def build_predictor(self)->SamPredictor:
        if self.use_sam_hq:
            self.predictor = SamPredictor(build_sam_hq_vit_l(checkpoint=self.sam_ckt).to(self.device))
        else:
            self.predictor = SamPredictor(build_sam(checkpoint=self.sam_ckt).to(self.device))
            
        if self.quant_smooth:
            assert self.act_scales_file is not None, "Run Smooth_sam.py to generate act_scales_file"
            act_scales = torch.load(self.act_scales_file)
            self.predictor.model = rtn_utils.smooth_sam(self.predictor.model, act_scales, alpha=0.5)
        elif self.quant_ro:      
            rotate_sam.rotate_sam(self.predictor.model,self.rot_args,self.rtn_ro)
            self.quant_rtn = False # do not quantize again
        if self.quant_gptq or self.gptq_cuda:
            from train.utils.dataloader import OnlineDataset
            from torchvision import transforms
            import numpy as np
            
            # Dataset configuration
            dataset_coift_val = {"name": "COIFT",
                    "im_dir": "./data/thin_object_detection/COIFT/images",
                    "gt_dir": "./data/thin_object_detection/COIFT/masks",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

            # Get file list
            valid_im_gt_list = get_im_gt_name_dict([dataset_coift_val], flag="valid")
            
            # Create dataset with transforms
            transform = transforms.Compose([Resize([1024,1024])])
            gos_dataset = OnlineDataset(valid_im_gt_list, transform=transform, eval_ori_resolution=True)
            
            # Create simple dataloader without DistributedSampler
            valid_dataloader = torch.utils.data.DataLoader(
                gos_dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=0,
                drop_last=False
            )
            
            print(f"Created dataloader with {len(valid_dataloader)} samples for GPTQ")
            
            # Run GPTQ quantization
            quantizer = gptq_utils.gptq_fwrd_sam(
                self.predictor.model, 
                valid_dataloader, 
                self.args_gptq.device, 
                self.args_gptq
            )
            
            modules_to_exclude = ["pos_embed", "cls_token", "patch_embed", "neck", "fpn", "mask_tokens", "iou_token", "output_upscaling", "output_hypernetworks_mlps"]
            if self.gptq_cuda:
                replace_linear_with_int4_gptq(self.predictor,quantizer, exclude_modules=modules_to_exclude)
            else:
                rtn_utils.replace_linear_with_target_and_quantize(module=self.predictor.model,
                                                            target_class=per_tensor_channel_group.W8A8Linear,
                                                            n_bit_w=self.n_bits,
                                                            n_bit_ac=self.args_gptq.ac_bits,
                                                            module_name_to_exclude=modules_to_exclude,
                                                            weight_quant=self.weight_quant,    
                                                            act_quant=self.act_quant,           
                                                            quantize_output=self.quantize_output,
                                                            quantize_weight=False) # weight already quantized in gptq
        
        if self.quant_rtn:
            modules_to_exclude = ["pos_embed", "cls_token", "patch_embed", "neck", "fpn", "mask_tokens", "iou_token", "output_upscaling", "output_hypernetworks_mlps"]
            rtn_utils.replace_linear_with_target_and_quantize(module=self.predictor.model,
                                                        target_class=per_tensor_channel_group.W8A8Linear,
                                                        n_bit_w=self.n_bits,
                                                        n_bit_ac=self.n_bits,
                                                        module_name_to_exclude=modules_to_exclude,
                                                        weight_quant=self.weight_quant,    
                                                        act_quant=self.act_quant,           
                                                        quantize_output=self.quantize_output)  
        if self.rtn_cuda:
            modules_to_exclude = ["pos_embed", "cls_token", "patch_embed", "neck", "fpn", "mask_tokens", "iou_token", "output_upscaling","output_hypernetworks_mlps"]
            replace_linear_with_int4(self.predictor.model,  exclude_modules=modules_to_exclude)                
            if self.save_rtn_cuda:
                save_cuda_quantized_model(self.predictor.model, save_dir="./pretrained_checkpoint", model_name="sam_int4_full")
        
        # print_model_structure(self.predictor.model, title="Final Structure")
        # exit()
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

    def demo(self, prompts:dict, image_dir:str, show_image:bool=False):
        self.set_image(image_dir)
        masks, scores, logits = self.inference(prompts, use_torch=True)
        if show_image:
            self.visualize(prompts, masks, scores, image_dir)
        return masks, scores, logits

        
    def visualize(self, prompts:dict, masks:torch.Tensor, scores:torch.Tensor, result_path:str):
        show_res_multi(masks, scores, prompts['point_coords'], prompts['point_labels'], prompts['box'], result_path, self.image)
        


class SeginwSamEngine(Engine):
    def __init__(self, strategy:InferenceStrategy):
        super().__init__(strategy)
        self.strategy.build_predictor()

    
    def load_model(self, model_config_path: str, model_checkpoint_path: str): 
        args = SLConfig.fromfile(model_config_path)
        args.device = self.strategy.device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        return model
    def demo(self,):
        pass
# %%
    def evaluate(self, args,args_quant):
        state="seginw_"
        if args_quant.quanrtn:
            state +="rtn"
        if args_quant.quansmooth:
            state += "smooth"
        if args_quant.quanro:
            state += "ro"

        logger =setup_logger(args.logging_path,state)
        
        objects = os.listdir(args.data_path)
        cfg = SLConfig.fromfile(args.config_file)
        # build model
        model = self.load_model(args.config_file, args.checkpoint_path)
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
            image_dir = os.path.join(args.data_path, object, 'valid')
            anno_path = os.path.join(args.data_path, object, 'valid', '_annotations_min1cat.coco.json')
            dataset = CocoDetection(
                image_dir,
                anno_path, 
                transforms=transform
            )
            data_loader = DataLoader(
                dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=args.num_workers, 
                collate_fn=collate_fn
            )

            # build post processor
            # ipdb.set_trace()
            tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
            postprocessor = PostProcessSeginw(num_select=args.num_select, coco_api=dataset.coco, tokenlizer=tokenlizer)

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
             
                if args.save_json:
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

            if args.save_json:
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
    def evaluate_coco(self, args, args_quant):
        state = "coco_"
        if args_quant.quanrtn:
            state += "rtn"
        if args_quant.quansmooth:
            state += "smooth"
        if args_quant.quanro:
            state += "ro"

        logger = setup_logger(args.logging_path, state)
        
        # Load GroundingDINO model
        cfg = SLConfig.fromfile(args.config_file)
        model = self.load_model(args.config_file, args.checkpoint_path)
        model = model.to(self.strategy.device)
        model = model.eval()

        # Build COCO dataset and dataloader
        
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Create COCO dataset from your config
        dataset = CocoDetection(
            img_folder=args.test.img_prefix,     # ./data/coco/val2017/
            ann_file=args.test.ann_file,         # ./data/coco/annotations/instances_val2017.json
            transforms=transform
        )
        data_loader = DataLoader(
            dataset,
            batch_size=1,                           # Keep batch_size=1 for evaluation
            shuffle=False,                          # No shuffling for evaluation
            num_workers=args.num_workers,           # From your config: 4
            collate_fn=collate_fn                 # Use GroundingDINO's collate function
        )
        tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
        postprocessor = PostProcessSeginw(num_select=args.num_select, coco_api=dataset.coco, tokenlizer=tokenlizer)

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
        # ipdb.set_trace()
        for i, (images, targets) in enumerate(data_loader):
            # get images and captions
            images = images.tensors.to(self.strategy.device)
            bs = images.shape[0]
            assert bs == 1
            input_captions = [caption] * bs

            # feed to the model
            # ipdb.set_trace()
            outputs = model(images, captions=input_captions)
            orig_target_sizes = torch.stack(
                [t["orig_size"] for t in targets], dim=0).to(images.device)
            results = postprocessor(outputs, orig_target_sizes)
            self.strategy.set_image(image_dir=f'{args.test.img_prefix}/{targets[0]["file_path"]}')

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

            if args.save_json:
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
        if args.save_json:
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
    @staticmethod
    def build_dataloader_runner(dataloader: Union[DataLoader, Dict],
                         seed: Optional[int] = None,
                         diff_rank_seed: bool = False) -> DataLoader:
        """Build dataloader.

        The method builds three components:

        - Dataset
        - Sampler
        - Dataloader

        An example of ``dataloader``::

            dataloader = dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=1,
                num_workers=9
            )

        Args:
            dataloader (DataLoader or dict): A Dataloader object or a dict to
                build Dataloader object. If ``dataloader`` is a Dataloader
                object, just returns itself.
            seed (int, optional): Random seed. Defaults to None.
            diff_rank_seed (bool): Whether or not set different seeds to
                different ranks. If True, the seed passed to sampler is set
                to None, in order to synchronize the seeds used in samplers
                across different ranks.


        Returns:
            Dataloader: DataLoader build from ``dataloader_cfg``.
        """
        if isinstance(dataloader, DataLoader):
            return dataloader

        dataloader_cfg = copy.deepcopy(dataloader)

        # build dataset
        dataset_cfg = dataloader_cfg.pop('dataset')
        if isinstance(dataset_cfg, dict):
            dataset = DATASETS.build(dataset_cfg)
            if hasattr(dataset, 'full_init'):
                dataset.full_init()
        else:
            # fallback to raise error in dataloader
            # if `dataset_cfg` is not a valid type
            dataset = dataset_cfg

        num_batch_per_epoch = dataloader_cfg.pop('num_batch_per_epoch', None)
        if num_batch_per_epoch is not None:
            world_size = get_world_size()
            num_samples = (
                num_batch_per_epoch * _get_batch_size(dataloader_cfg) *
                world_size)
            dataset = _SlicedDataset(dataset, num_samples)

        # build sampler
        sampler_cfg = dataloader_cfg.pop('sampler')
        if isinstance(sampler_cfg, dict):
            sampler_seed = None if diff_rank_seed else seed
            sampler = DATA_SAMPLERS.build(
                sampler_cfg,
                default_args=dict(dataset=dataset, seed=sampler_seed))
        else:
            # fallback to raise error in dataloader
            # if `sampler_cfg` is not a valid type
            sampler = sampler_cfg

        # build batch sampler
        batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
        if batch_sampler_cfg is None:
            batch_sampler = None
        elif isinstance(batch_sampler_cfg, dict):
            batch_sampler = DATA_SAMPLERS.build(
                batch_sampler_cfg,
                default_args=dict(
                    sampler=sampler,
                    batch_size=dataloader_cfg.pop('batch_size')))
        else:
            # fallback to raise error in dataloader
            # if `batch_sampler_cfg` is not a valid type
            batch_sampler = batch_sampler_cfg

        # build dataloader
        init_fn: Optional[partial]

        if 'worker_init_fn' in dataloader_cfg:
            worker_init_fn_cfg = dataloader_cfg.pop('worker_init_fn')
            worker_init_fn_type = worker_init_fn_cfg.pop('type')
            if isinstance(worker_init_fn_type, str):
                worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
            elif callable(worker_init_fn_type):
                worker_init_fn = worker_init_fn_type
            else:
                raise TypeError(
                    'type of worker_init_fn should be string or callable '
                    f'object, but got {type(worker_init_fn_type)}')
            assert callable(worker_init_fn)
            init_fn = partial(worker_init_fn,
                              **worker_init_fn_cfg)  # type: ignore
        else:
            if seed is not None:
                disable_subprocess_warning = dataloader_cfg.pop(
                    'disable_subprocess_warning', False)
                assert isinstance(disable_subprocess_warning, bool), (
                    'disable_subprocess_warning should be a bool, but got '
                    f'{type(disable_subprocess_warning)}')
                init_fn = partial(
                    default_worker_init_fn,
                    num_workers=dataloader_cfg.get('num_workers'),
                    rank=get_rank(),
                    seed=seed,
                    disable_subprocess_warning=disable_subprocess_warning)
            else:
                init_fn = None

        # `persistent_workers` requires pytorch version >= 1.7
        if ('persistent_workers' in dataloader_cfg
                and digit_version(TORCH_VERSION) < digit_version('1.7.0')):
            print_log(
                '`persistent_workers` is only available when '
                'pytorch version >= 1.7',
                logger='current',
                level=logging.WARNING)
            dataloader_cfg.pop('persistent_workers')

        # The default behavior of `collat_fn` in dataloader is to
        # merge a list of samples to form a mini-batch of Tensor(s).
        # However, in mmengine, if `collate_fn` is not defined in
        # dataloader_cfg, `pseudo_collate` will only convert the list of
        # samples into a dict without stacking the batch tensor.
        collate_fn_cfg = dataloader_cfg.pop('collate_fn',
                                            dict(type='pseudo_collate'))
        if isinstance(collate_fn_cfg, dict):
            collate_fn_type = collate_fn_cfg.pop('type')
            if isinstance(collate_fn_type, str):
                collate_fn = FUNCTIONS.get(collate_fn_type)
            else:
                collate_fn = collate_fn_type
            collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
        elif callable(collate_fn_cfg):
            collate_fn = collate_fn_cfg
        else:
            raise TypeError(
                'collate_fn should be a dict or callable object, but got '
                f'{collate_fn_cfg}')
        data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler if batch_sampler is None else None,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            worker_init_fn=init_fn,
            **dataloader_cfg)
        return data_loader
# %%
    def evaluate_coco_mmdet(self, args, args_quant):
    
        state = "coco_"
        if args_quant.quanrtn:
            state += "rtn"
        if args_quant.quansmooth:
            state += "smooth"
        if args_quant.quanro:
            state += "ro"

        logger = setup_logger(args.logging_path, state)
        
        
        # Load GroundingDINO config and model
        cfg_grounding = SLConfig.fromfile(args.config_file)
        model = self.load_model(args.config_file, args.checkpoint_path)
        model = model.to(self.strategy.device)
        model = model.eval()
        
        # Parse MMDet arguments
        args_mmdet = parse_args_train()

        # Reduce the number of repeated compilations and improve training speed
        setup_cache_size_limit_of_dynamo()

        # Load MMDet config
        cfg_mmdet = Config.fromfile(args_mmdet.config)
   
        cfg_mmdet.launcher = args_mmdet.launcher
        if args_mmdet.cfg_options is not None:
            cfg_mmdet.merge_from_dict(args_mmdet.cfg_options)

        # work_dir is determined in this priority: CLI > segment in file > filename
        if args_mmdet.work_dir is not None:
            # update configs according to CLI args if args_mmdet.work_dir is not None
            cfg_mmdet.work_dir = args_mmdet.work_dir
        elif cfg_mmdet.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg_mmdet.work_dir is None
            cfg_mmdet.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args_mmdet.config))[0])

        # Build dataloader using MMEngine's approach
        data_loader = self.build_dataloader_runner(
            dataloader=cfg_mmdet.val_dataloader,
            seed=None,
            diff_rank_seed=False
        )
        
        dataset = data_loader.dataset

        # Build GroundingDINO transform for model input
        transform_grounding = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        if not hasattr(dataset, 'coco'):
            from pycocotools.coco import COCO
            ann_file = cfg_mmdet.val_dataloader.dataset.ann_file
            data_root = cfg_mmdet.val_dataloader.dataset.data_root
            full_ann_path = osp.join(data_root, ann_file) if not osp.isabs(ann_file) else ann_file
            coco_api = COCO(full_ann_path)
        else:
            coco_api = dataset.coco
        
        # Use GroundingDINO config for tokenizer
        tokenlizer = get_tokenlizer.get_tokenlizer(cfg_grounding.text_encoder_type)
        postprocessor = PostProcessSeginw(num_select=args.num_select, coco_api=coco_api, tokenlizer=tokenlizer)

        # build evaluator
        evaluator = CocoGroundingEvaluator(
            coco_api, iou_types=("bbox","segm"), useCats=True)

        # build captions
        category_dict = coco_api.dataset['categories']
        cat_list = [item['name'] for item in category_dict]
        caption = " . ".join(cat_list) + ' .'
        print("Input text prompt:", caption)
        predictor = self.strategy.build_predictor()

        json_file = []
        start = time.time()
        progress_bar = tqdm(data_loader, desc=f"Evaluating COCO", total=len(data_loader))
        
        for i, data_batch in enumerate(data_loader):
            # Extract data from MMDet format
            inputs = data_batch['inputs']  # Shape: [B, C, H, W]
            data_samples = data_batch['data_samples']  # List of DetDataSample
         
            inputs = inputs[0]
            batch_size = 1
            assert batch_size == 1, f"Expected batch size 1, got {batch_size}"
            
            # Get the single sample
            data_sample = data_samples[0]
            img_path = data_sample.img_path
            img_id = data_sample.img_id
            
            # Convert MMDet tensor format to PIL Image for GroundingDINO
            image_np = inputs.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [H, W, C]
            
            # Apply GroundingDINO transform
            from PIL import Image
            image_pil = Image.fromarray(image_np)
            image_tensor, _ = transform_grounding(image_pil, None)
            image_tensor = image_tensor.unsqueeze(0).to(self.strategy.device)  # Add batch dimension
            
            # Prepare captions for GroundingDINO
            input_captions = [caption]
            
            outputs = model(image_tensor, captions=input_captions)
            orig_size = data_sample.ori_shape  # (H, W)
            orig_target_sizes = torch.tensor([orig_size]).to(self.strategy.device)
            
            # Create target format for postprocessor
            targets = [{
                "orig_size": torch.tensor(orig_size),
                "image_id": img_id,
                "file_path": os.path.basename(img_path)
            }]
            
            # Postprocess GroundingDINO results
            results = postprocessor(outputs, orig_target_sizes)
            
            if len(results[0]['boxes']) > 0:
                # Set image for SAM
                self.strategy.set_image(image_dir=img_path)
                
                # Prepare boxes for SAM
                input_boxes = results[0]['boxes'].cpu()
                transformed_boxes = predictor.transform.apply_boxes_torch(
                    input_boxes, self.strategy.image.shape[:2]
                ).to(self.strategy.device)
                # SAM inference
                prompts = {
                    'point_coords': None, 
                    'point_labels': None,
                    'boxes': transformed_boxes,
                    'hq_token_only': True,
                }
                masks, _, _ = self.strategy.inference(prompts, use_torch=True)
                results[0]['masks'] = masks.cpu().numpy()
                
            else:
                # No detections, add empty masks
                results[0]['masks'] = np.array([])
            
            # Prepare for evaluation
            cocogrounding_res = {targets[0]["image_id"]: results[0]}
            save_items = evaluator.update(cocogrounding_res)
            # ipdb.set_trace()
            if args.save_json:
                new_items = []
                for item in save_items:
                    new_item = {
                        'image_id': item['image_id'],
                        'category_id': item['category_id'], 
                        'segmentation': item['segmentation'],
                        'score': item['score']
                    }
                    new_items.append(new_item)
                json_file.extend(new_items)

            if (i+1) % 30 == 0:
                used_time = time.time() - start
                eta = len(data_loader) / (i+1e-5) * used_time - used_time
                print(f"processed {i+1}/{len(data_loader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s")
            
            progress_bar.update(1)

        progress_bar.close()

        # Final evaluation
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        print("Final results:", evaluator.coco_eval["segm"].stats.tolist())
        
        if args.save_json:
            if self.strategy.use_sam_hq:
                os.makedirs('coco_output/sam_hq/', exist_ok=True)
                save_path = 'coco_output/sam_hq/coco_val.json'
            else:
                os.makedirs('coco_output/sam/', exist_ok=True)
                save_path = 'coco_output/sam/coco_val.json'
            with open(save_path, 'w') as f:
                json.dump(json_file, f)
            print(f"Results saved to: {save_path}")

# %%
    def evaluate_loadptq4sam(self, args, args_quant):
        pass

# %%

if __name__ == "__main__":

    args = OmegaConf.load('quant/config/coco/base_h.yaml')

    engine = SeginwSamEngine(SeginwInferenceStrategy(args))
    # breakpoint()
    # engine.evaluate(args.data,args.quantization)
    # engine.evaluate_coco(args.data,args.quantization)
    engine.evaluate_coco_mmdet(args.data,args.quantization)
  
    prompts = {
        'point_coords': None, 
        'point_labels': None,
        'box': np.array([[4,13,1007,1023]]),
        'hq_token_only': True,
    }
    # image = Image.open('../input_imgs/example0.png')
    engine.demo(prompts, image_dir='../input_imgs/example1.png', show_image=True)

# %%

