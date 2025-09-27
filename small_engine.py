import os
import math
import torch
import torch.nn as nn
from functools import  partial
import numpy as np
import os
import logging
import time
from train.utils.dataloader import get_im_gt_name_dict, Resize
from abc import abstractmethod
from data_utils import OnlineDataset
from torchvision import transforms
from segment_anything.modeling.transformer import TwoWayTransformer
from train.train import compute_iou, compute_boundary_iou, show_anns, MaskDecoderHQ
from torch.utils.data import DataLoader
from data_utils import OnlineDataset
from segment_anything import SamPredictor, sam_model_registry
from matplotlib import pyplot as plt
from functools import partial
from accelerate import Accelerator
import train.utils.misc as misc
from tqdm.auto import tqdm
from utils import show_mask_image
from collections import defaultdict
from decoder_quant import  mask_decoder_monkey_patch, inference_image






def create_calib_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1 ):
    gos_dataloaders = []
    gos_datasets = []
    for i in range(len(name_im_gt_list)):   
        gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
        dataloader = DataLoader(gos_dataset, batch_size, drop_last=False)
        gos_dataloaders.append(dataloader)
        gos_datasets.append(gos_dataset) 
    return gos_dataloaders, gos_datasets

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

def plot_output(imgs, masks, labels_boxes, scores:np.ndarray, example_idx, output_path='./output' ):
    plt.figure(figsize=(10, 10))
    plt.imshow(imgs.squeeze())
    os.makedirs(output_path, exist_ok=True)


    if len(masks) > 0:
        show_mask_image(masks[0], plt.gca(), random_color=False)
    
    box = labels_boxes[0]
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    plt.gca().add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
                            
    plt.title(f'Example {example_idx} - Score: {scores.item():.3f}')
    plt.savefig(f'{output_path}/sample_{example_idx}.png')
    plt.axis('off')
    plt.show()



class Engine():
    def __init__(self, strategy_name:str) -> None:
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.stat = {}
        dataset_dis = {
            "name": "DIS5K-VD",
            "im_dir": "./data/DIS5K/DIS-VD/im",
            "gt_dir": "./data/DIS5K/DIS-VD/gt",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        }
        dataset_thin = {"name": "ThinObject5k-TR",
            "im_dir": "./data/thin_object_detection/ThinObject5K/images_train",
            "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train",
            "im_ext": ".jpg",
            "gt_ext": ".png"}

        valid_im_gt_list = get_im_gt_name_dict([dataset_dis], flag="valid")
        self.dataloaders, self.datasets = create_calib_dataloaders(
            valid_im_gt_list,
            my_transforms = [
                        Resize([1024, 1024])
                    ],
            batch_size=1,
        )

    def eval_hq44k(self, predictor:SamPredictor, num_samples=None, plot_figures=False):
        # model.eval()

        #load data
        for k in range(len(self.dataloaders)):
            dataloader = self.accelerator.prepare(self.dataloaders[k])
            print('valid_dataloader len:', len(dataloader))
            # logger.info(f"\nCalibarating {self.datasets[k]['name']}:")
            progress_bar = tqdm(total=len(dataloader) if not num_samples else num_samples, desc=f"Eval HQ44k")
            metric_logger = misc.MetricLogger(delimiter="  ")
            index = 0
            for  data_val in metric_logger.log_every(dataloader, 2):
                if index == num_samples: break 
                _, inputs_val, labels_val, _, labels_ori, ori_image = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label'], data_val['ori_im']
                imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
                predictor.set_image(imgs.squeeze())
                labels_boxes = misc.masks_to_boxes(labels_val[:,0,:,:])
                masks, scores, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=labels_boxes, 
                    hq_token_only=True
                )
                # breakpoint()
                iou = compute_iou(masks,labels_ori)
                boundary_iou = compute_boundary_iou(masks,labels_ori)
                loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
                loss_dict_reduced = misc.reduce_dict(loss_dict)
                metric_logger.update(**loss_dict_reduced)
                progress_bar.update(1)
                if plot_figures:
                    masks = masks.squeeze(1).cpu().detach().numpy()
                    labels_boxes = labels_boxes.cpu().detach().numpy()
                    scores = scores.squeeze().cpu().detach().numpy()
                    # breakpoint()

                    plot_output(imgs, masks, labels_boxes, scores, index )
                index += 1

                





if __name__ == '__main__':
    from quant_utils import AttnBasedProcessor, DoNothingProcessor, SignProcessor


    model_type = 'vit_l'
    num_calib_samples=16
    checkpoint_path= './pretrained_checkpoint/sam_hq_vit_l.pth'
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)
    # 
    # processor = SignProcessor('sign') 
    # processor = DoNothingProcessor('base') 
    processor = AttnBasedProcessor('attn') 
    
    processor.calibrate(
        predictor=predictor, 
        modules=(TwoWayTransformer),
        num_samples=num_calib_samples
    )
    mask_decoder_monkey_patch(predictor.model, processor, n_bits=4)
    engine = Engine('hq44k') 
    engine.eval_hq44k(predictor=predictor, num_samples=32, plot_figures=True)
    
    

    