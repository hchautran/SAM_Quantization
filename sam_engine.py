import os
import numpy as np
import torch
from accelerate import Accelerator
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from segment_anything import SamPredictor
from train.train import compute_iou, compute_boundary_iou
import train.utils.misc as misc
# from utils.utils import show_mask_image


def get_default_datasets():
    """Get default dataset configurations"""
    return [
        # {
        #     "name": "DIS5K-VD",
        #     "im_dir": "./data/DIS5K/DIS-VD/im",
        #     "gt_dir": "./data/DIS5K/DIS-VD/gt",
        #     "im_ext": ".jpg",
        #     "gt_ext": ".png"
        # },
        # {
        #     "name": "DIS5K-TR",
        #     "im_dir": "./data/DIS5K/DIS-TR/im",
        #     "gt_dir": "./data/DIS5K/DIS-TR/gt",
        #     "im_ext": ".jpg",
        #     "gt_ext": ".png"
        # },
        {
            "name": "ThinObject5k-TR",
            "im_dir": "./data/thin_object_detection/ThinObject5K/images_train",
            "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        }
    ]


def plot_output(imgs, masks, labels_boxes, scores: np.ndarray, example_idx, output_path='./output'):
    """Plot prediction output with mask and bounding box"""
    plt.figure(figsize=(10, 10))
    plt.imshow(imgs.squeeze())
    os.makedirs(output_path, exist_ok=True)

    if len(masks) > 0:
        show_mask_image(masks[0], plt.gca(), random_color=False)

    box = labels_boxes[0]
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    plt.gca().add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    plt.title(f'Example {example_idx} - Score: {scores.item():.3f}')
    plt.savefig(f'{output_path}/sample_{example_idx}.png')
    plt.axis('off')
    plt.show()


class Evaluator:
    """Handles model evaluation on HQ44k dataset"""

    def __init__(self, accelerator, dataloaders, datasets):
        self.accelerator = accelerator
        self.dataloaders = dataloaders
        self.datasets = datasets

    def eval_hq44k(self, predictor: SamPredictor, num_samples=None, plot_figures=False):
        """Evaluate model on HQ44k dataset"""
        test_stats = {}

        for k in range(len(self.dataloaders)):
            dataloader = self.accelerator.prepare(self.dataloaders[k])
            print('valid_dataloader len:', len(dataloader))
            progress_bar = tqdm(total=len(dataloader) if not num_samples else num_samples, desc=f"Eval HQ44k")
            metric_logger = misc.MetricLogger(delimiter="  ")
            index = 0

            for data_val in metric_logger.log_every(dataloader, 2):
                if index == num_samples:
                    break

                _, inputs_val, labels_val, _, labels_ori, ori_image = (
                    data_val['imidx'], data_val['image'], data_val['label'],
                    data_val['shape'], data_val['ori_label'], data_val['ori_im']
                )

                imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy()
                predictor.set_image(imgs.squeeze())
                labels_boxes = misc.masks_to_boxes(labels_val[:, 0, :, :])

                masks, scores, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=labels_boxes,
                    hq_token_only=True
                )

                iou = compute_iou(masks, labels_ori)
                boundary_iou = compute_boundary_iou(masks, labels_ori)
                loss_dict = {"val_iou_" + str(k): iou, "val_boundary_iou_" + str(k): boundary_iou}
                loss_dict_reduced = misc.reduce_dict(loss_dict)
                metric_logger.update(**loss_dict_reduced)
                progress_bar.update(1)

                if plot_figures:
                    masks = masks.squeeze(1).cpu().detach().numpy()
                    labels_boxes = labels_boxes.cpu().detach().numpy()
                    scores = scores.squeeze().cpu().detach().numpy()
                    plot_output(imgs, masks, labels_boxes, scores, index)

                index += 1

            metric_logger.synchronize_between_processes()
            print("Averaged stats:", metric_logger)
            resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
            test_stats.update(resstat)

        return test_stats
