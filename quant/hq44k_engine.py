# %%

import os
import yaml
import cv2
import torch
from engine import InferenceStrategy, Engine
import numpy as np
from tqdm.auto import tqdm
from train.utils.dataloader import get_im_gt_name_dict, create_dataloaders, Resize
from train.utils.misc import   F, random
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou, show_anns, MaskDecoderHQ
from train.segment_anything_training import sam_model_registry
import cv2
from tqdm.auto import tqdm
import json
import time
from omegaconf import OmegaConf
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

    # parser.add_argument("--output", type=str, required=True, 
                        # help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    # parser.add_argument("--checkpoint", type=str, required=True, 
                        # help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--max_epoch_num', default=12, type=int)
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()



class Hq44kInferenceStrategy(InferenceStrategy):
    def __init__(self, args):
        # build sam encoder
        self.checkpoint = args.checkpoint
        self.model_type = args.model_type
        self.restore_model = args.restore_model
        self.predictor = None

    def build_predictor(self):
        self.hq_mask_decoder = MaskDecoderHQ(self.model_type) 
        self.predictor = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        if self.restore_model:
            print("restore model from:", self.restore_model)
            self.hq_mask_decoder.load_state_dict(torch.load(self.restore_model))

    def distribute(self, dist_args):
        if torch.cuda.is_available():
            self.predictor.to(device=dist_args.device)
            self.hq_mask_decoder.to(device=dist_args.device)
            self.predictor = torch.nn.parallel.DistributedDataParallel(self.predictor, device_ids=[dist_args.gpu], find_unused_parameters=dist_args.find_unused_params)
            self.hq_mask_decoder = torch.nn.parallel.DistributedDataParallel(self.hq_mask_decoder, device_ids=[dist_args.gpu], find_unused_parameters=dist_args.find_unused_params)
        else:
            raise NotImplementedError("Distributed training supported on this machine")






    def set_image(self, image_dir:str):
        raise NotImplementedError("")

    def set_video(self, video_dir:str):
        raise NotImplementedError("Video inference is not supported for SAM")
        
    @torch.inference_mode()
    @torch.no_grad()
    def inference(self, inputs:dict):
        self.hq_mask_decoder.eval()
        # encoder image and prompts
        batched_output, interm_embeddings = self.predictor(inputs, multimask_output=False) 
        batch_len = len(batched_output)
        encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
        image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
        sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
        dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

        # decode high quality mask
        masks_sam, masks_hq = self.hq_mask_decoder(
            image_embeddings=encoder_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            hq_token_only=False,
            interm_embeddings=interm_embeddings,
        )

        return masks_sam, masks_hq 

        
    def visualize(self, prompts:dict, masks:torch.Tensor, scores:torch.Tensor, result_path:str):
        raise NotImplementedError("")
        




class Hq44kSamEngine(Engine):
    def __init__(self, strategy:InferenceStrategy):
        super().__init__(strategy)
        self.strategy.build_predictor()
        dataset_dis = {"name": "DIS5K-TR",
                    "im_dir": "./data/DIS5K/DIS-TR/im",
                    "gt_dir": "./data/DIS5K/DIS-TR/gt",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_thin = {"name": "ThinObject5k-TR",
                    "im_dir": "./data/thin_object_detection/ThinObject5K/images_train",
                    "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_fss = {"name": "FSS",
                    "im_dir": "./data/cascade_psp/fss_all",
                    "gt_dir": "./data/cascade_psp/fss_all",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_duts = {"name": "DUTS-TR",
                    "im_dir": "./data/cascade_psp/DUTS-TR",
                    "gt_dir": "./data/cascade_psp/DUTS-TR",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_duts_te = {"name": "DUTS-TE",
                    "im_dir": "./data/cascade_psp/DUTS-TE",
                    "gt_dir": "./data/cascade_psp/DUTS-TE",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_ecssd = {"name": "ECSSD",
                    "im_dir": "./data/cascade_psp/ecssd",
                    "gt_dir": "./data/cascade_psp/ecssd",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_msra = {"name": "MSRA10K",
                    "im_dir": "./data/cascade_psp/MSRA_10K",
                    "gt_dir": "./data/cascade_psp/MSRA_10K",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        # valid set
        dataset_coift_val = {"name": "COIFT",
                    "im_dir": "./data/thin_object_detection/COIFT/images",
                    "gt_dir": "./data/thin_object_detection/COIFT/masks",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_hrsod_val = {"name": "HRSOD",
                    "im_dir": "./data/thin_object_detection/HRSOD/images",
                    "gt_dir": "./data/thin_object_detection/HRSOD/masks_max255",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_thin_val = {"name": "ThinObject5k-TE",
                    "im_dir": "./data/thin_object_detection/ThinObject5K/images_test",
                    "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_test",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        dataset_dis_val = {"name": "DIS5K-VD",
                    "im_dir": "./data/DIS5K/DIS-VD/im",
                    "gt_dir": "./data/DIS5K/DIS-VD/gt",
                    "im_ext": ".jpg",
                    "gt_ext": ".png"}

        self.train_datasets = [dataset_dis, dataset_thin, dataset_fss, dataset_duts, dataset_duts_te, dataset_ecssd, dataset_msra]
        self.valid_datasets = [dataset_dis_val, dataset_coift_val, dataset_hrsod_val, dataset_thin_val] 

        
    def train(self, args:dict):
        pass

    @torch.no_grad()
    def demo(self,):
        pass 


    @torch.no_grad()
    def evaluate(self, args, visualize:bool=False):
        misc.init_distributed_mode(args)
        print('world size: {}'.format(args.world_size))
        print('rank: {}'.format(args.rank))
        print('local_rank: {}'.format(args.local_rank))
        print("args: " + str(args) + '\n')
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.strategy.distribute(args)

        device = self.strategy.predictor.device
        valid_im_gt_list = get_im_gt_name_dict(self.valid_datasets, flag="valid")
        valid_dataloaders, _ = create_dataloaders(
            valid_im_gt_list,
            my_transforms = [
                        Resize(args.input_size)
                    ],
            batch_size=args.batch_size_valid,
            training=False
        )
        test_stats = {}
        for k in range(len(valid_dataloaders)):
            metric_logger = misc.MetricLogger(delimiter="  ")
            valid_dataloader = valid_dataloaders[k]
            print('valid_dataloader len:', len(valid_dataloader))
            progress_bar = tqdm(total=len(valid_dataloader), desc=f"Validating {self.valid_datasets[k]['name']}")
            start = time.time()
            for data_val in metric_logger.log_every(valid_dataloader, 2):
                _, inputs_val, labels_val, _, labels_ori = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label']

                # prepare image & prompts 
                if torch.cuda.is_available():
                    inputs_val = inputs_val.cuda()
                    labels_val = labels_val.cuda()
                    labels_ori = labels_ori.cuda()

                imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
                labels_box = misc.masks_to_boxes(labels_val[:,0,:,:])

                batched_input = []
                for b_i in range(len(imgs)):
                    dict_input = dict()
                    input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=device).permute(2, 0, 1).contiguous()
                    dict_input['image'] = input_image 
                    dict_input['boxes'] = labels_box[b_i:b_i+1]
                    dict_input['original_size'] = imgs[b_i].shape[:2]
                    batched_input.append(dict_input)

                _, masks_hq = self.strategy.inference(batched_input)
                # compute metric & update
                iou = compute_iou(masks_hq,labels_ori)
                boundary_iou = compute_boundary_iou(masks_hq,labels_ori)
                loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
                loss_dict_reduced = misc.reduce_dict(loss_dict)
                metric_logger.update(**loss_dict_reduced)
                progress_bar.update(1)
            total_time = time.time() - start

            print('=' * 100) 
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
            test_stats.update(resstat)
            test_stats['total_time'] = f'{total_time/1e9:.2f}'
            
            print(test_stats)

        return test_stats
                

# %%

if __name__ == "__main__":
    model_args = OmegaConf.load('quant/config/hq44k/base_l.yaml')
    args = get_args_parser()
    
    engine = Hq44kSamEngine(Hq44kInferenceStrategy(model_args.model))
    engine.evaluate(args)

# %%

