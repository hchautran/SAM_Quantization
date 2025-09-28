
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PTQ4SAM_path = os.path.join(project_root, 'PTQ4SAM')

sys.path.insert(0,project_root)
sys.path.insert(0,PTQ4SAM_path)

# from PTQ4SAM.projects.instance_segment_anything.models.det_wrapper_instance_sam import DetWrapperInstanceSAM
import cv2
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.core import bbox2result
from mmdet.models import DETECTORS, BaseDetector
from projects.instance_segment_anything.models.segment_anything import sam_model_registry, SamPredictor
from projects.instance_segment_anything.models.focalnet_dino.focalnet_dino_wrapper import FocalNetDINOWrapper
from projects.instance_segment_anything.models.hdetr.hdetr_wrapper import HDetrWrapper
from projects.instance_segment_anything.models.detector.detector_wrapper import DetectorWrapper
import numpy as np

@DETECTORS.register_module()
class DetObserverInstanceSAM(BaseDetector):
    
    wrapper_dict = {'hdetr': HDetrWrapper,
                    'focalnet_dino': FocalNetDINOWrapper,
                    'generalized_detector': DetectorWrapper}

    def __init__(self,
                 det_wrapper_type='hdetr',
                 det_wrapper_cfg=None,
                 det_model_ckpt=None,
                 num_classes=80,

                 model_type='vit_b',
                 sam_checkpoint=None,
                 use_sam_iou=True,
                 best_in_multi_mask=False,
                 show_image=0,
                 result_coco_path = None,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(DetObserverInstanceSAM, self).__init__(init_cfg)
        self.learnable_placeholder = nn.Embedding(1, 1)
        det_wrapper_cfg = Config(det_wrapper_cfg)
        assert det_wrapper_type in self.wrapper_dict.keys()
        self.det_model = self.wrapper_dict[det_wrapper_type](args=det_wrapper_cfg)
        if det_model_ckpt is not None:
            load_checkpoint(self.det_model.model,
                            filename=det_model_ckpt,
                            map_location='cpu')

        self.num_classes = num_classes

        # Segment Anything
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        _ = sam.to(device=self.learnable_placeholder.weight.device)
        self.predictor = SamPredictor(sam)
        # Whether use SAM's predicted IoU to calibrate the confidence score.
        self.use_sam_iou = use_sam_iou
        # If True, set multimask_output=True and return the mask with highest predicted IoU.
        # if False, set multimask_output=False and return the unique output mask.
        self.best_in_multi_mask = best_in_multi_mask
        self.save_position_encoding = False
        self.show_image = show_image
        self.result_coco_path = result_coco_path
    def init_weights(self):
        pass

    def simple_test(self, img, img_metas, rescale=True, ori_img=None, calib = False,get_det_results=False):
        """Test without augmentation.
        Args:
            imgs (Tensor): A batch of images.
            img_metas (list[dict]): List of image information.
        """
        assert rescale
        assert len(img_metas) == 1
        # results: List[dict(scores, labels, boxes)]
        with torch.no_grad():
            results = self.det_model.simple_test(img,
                                                img_metas,
                                                rescale)
        if get_det_results:
            return results
        # Tensor(n,4), xyxy, ori image scale
        output_boxes = results[0]['boxes']

        if ori_img is None:
            image_path = img_metas[0]['filename']
            ori_img = cv2.imread(image_path)
            ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(ori_img)
        
        limit_boxes = 120
        if len(output_boxes) >=limit_boxes:
            output_boxes= output_boxes[:limit_boxes]
        transformed_boxes = self.predictor.transform.apply_boxes_torch(output_boxes, ori_img.shape[:2])
        
        # mask_pred: n,1/3,h,w
        # sam_score: n, 1/3
        if calib:
            # import pdb;pdb.set_trace()
            if not self.save_position_encoding:
                try:
                    self.predictor.predict_pe()
                    self.save_position_encoding = True
                except:
                    pass
            return self.predictor.predict_calib(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=self.best_in_multi_mask,
                return_logits=True)
        mask_pred, sam_score, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=self.best_in_multi_mask,
            return_logits=True,
        )
        
        if self.best_in_multi_mask:
            # sam_score: n
            sam_score, max_iou_idx = torch.max(sam_score, dim=1)
            # mask_pred: n,h,w
            mask_pred = mask_pred[torch.arange(mask_pred.size(0)),
                                  max_iou_idx]
        else:
            # Tensor(n,h,w), raw mask pred
            # n,1,h,w->n,h,w
            mask_pred = mask_pred.squeeze(1)
            # n,1->n
            sam_score = sam_score.squeeze(-1)

        # Tensor(n,)
        label_pred = results[0]['labels'][:limit_boxes]

        score_pred = results[0]['scores'][:limit_boxes]

        # mask_pred: Tensor(n,h,w)
        # label_pred: Tensor(n,)
        # score_pred: Tensor(n,)
        # sam_score: Tensor(n,)
        mask_pred_binary = (mask_pred > self.predictor.model.mask_threshold).float()
        if self.use_sam_iou:
            det_scores = score_pred * sam_score
        else:
            # n
            mask_scores_per_image = (mask_pred * mask_pred_binary).flatten(1).sum(1) / (
                    mask_pred_binary.flatten(1).sum(1) + 1e-6)
            det_scores = score_pred * mask_scores_per_image
        # det_scores = score_pred
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = torch.cat([output_boxes, det_scores[:, None]], dim=-1)
        bbox_results = bbox2result(bboxes, label_pred, self.num_classes)
        mask_results = [[] for _ in range(self.num_classes)]
        for j, label in enumerate(label_pred):
            mask = mask_pred_binary[j].detach().cpu().numpy()
            mask_results[label].append(mask)
        output_results = [(bbox_results, mask_results)]
        # import ipdb; ipdb.set_trace()
        print(self.show_image)
        if self.show_image >0:
            print("chiii")
            self.show_image -=1
            self.save_image_coco(ori_img, bbox_results, mask_results,  self.show_image,self.result_coco_path)
        return output_results
    def save_image_coco(self, img, bbox_results, mask_results, index, image_path):
        """Save image with predicted bounding boxes and masks.
        
        Args:
            img (np.ndarray): Original image in RGB format
            bbox_results (list[np.ndarray]): Bbox results for each class
            mask_results (list[list]): Mask results for each class  
            index (int): Index for naming the output image
            image_path (str): Directory path to save the output image
        """
        import mmcv
        from mmdet.core.visualization import imshow_det_bboxes
        import os

        if image_path is None:
            return

        # Ensure directory exists
        os.makedirs(image_path, exist_ok=True)

        # Build filename
        save_path = os.path.join(image_path, f"image_{index}.png")

        # Convert bbox_results back to format needed for visualization
        bboxes = []
        labels = []
        for class_id, class_bboxes in enumerate(bbox_results):
            if len(class_bboxes) > 0:
                bboxes.append(class_bboxes)
                labels.extend([class_id] * len(class_bboxes))

        if len(bboxes) == 0:
            # No detections, just save original image
            mmcv.imwrite(img, save_path)
            print(f"Saved original image (no detections) to: {save_path}")
            return

        # Stack all bboxes
        bboxes = np.vstack(bboxes)
        labels = np.array(labels)

        # Prepare masks for visualization
        segms = None
        if mask_results is not None:
            segms = []
            for class_id, class_masks in enumerate(mask_results):
                segms.extend(class_masks)
            if len(segms) > 0:
                segms = np.stack(segms, axis=0)
            else:
                segms = None

        # Draw bounding boxes and masks on image
        result_img = imshow_det_bboxes(
            img.copy(),
            bboxes,
            labels,
            segms,
            class_names=getattr(self, 'CLASSES', None),
            score_thr=0.3,
            bbox_color=(72, 101, 241),
            text_color=(72, 101, 241),
            mask_color=None,
            thickness=2,
            font_size=13,
            win_name='',
            show=False,
            wait_time=0,
            out_file=save_path
        )

        print(f"Saved visualization to: {save_path}")
    # not implemented:
    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError

    def onnx_export(self, img, img_metas):
        raise NotImplementedError

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    def forward_train(self, imgs, img_metas, **kwargs):
        raise NotImplementedError

    def extract_feat(self, img_img_metas_dict, **kwargs):
        if isinstance(img_img_metas_dict,dict):
            img = img_img_metas_dict['img']
            img_metas = img_img_metas_dict['img_metas']
            feat = self.forward(return_loss=False,img=img,img_metas=img_metas,calib = True, **kwargs)
            return feat
        else:
            feat = self.only_forward_sam(img_img_metas_dict,calib=True)
            return feat

    def only_forward_sam(self, det_results_and_torch_img, rescale=True, calib = False):
        
        results = det_results_and_torch_img[0]
        torch_img = det_results_and_torch_img[1]
        assert rescale
        # Tensor(n,4), xyxy, ori image scale
        output_boxes = results[0]['boxes']


        self.predictor.set_torch_image(torch_img[0], torch_img[1])

        transformed_boxes = self.predictor.transform.apply_boxes_torch(output_boxes, torch_img[1])

        # mask_pred: n,1/3,h,w
        # sam_score: n, 1/3
        if calib:
            # import pdb;pdb.set_trace()
            if not self.save_position_encoding:
                self.predictor.predict_pe()
                self.save_position_encoding = True
            return self.predictor.predict_calib(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=self.best_in_multi_mask,
                return_logits=True)
        
        mask_pred, sam_score, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=self.best_in_multi_mask,
            return_logits=True,
        )
        
        if self.best_in_multi_mask:
            # sam_score: n
            sam_score, max_iou_idx = torch.max(sam_score, dim=1)
            # mask_pred: n,h,w
            mask_pred = mask_pred[torch.arange(mask_pred.size(0)),
                                  max_iou_idx]
        else:
            # Tensor(n,h,w), raw mask pred
            # n,1,h,w->n,h,w
            mask_pred = mask_pred.squeeze(1)
            # n,1->n
            sam_score = sam_score.squeeze(-1)

        # Tensor(n,)
        label_pred = results[0]['labels']

        score_pred = results[0]['scores']

        # mask_pred: Tensor(n,h,w)
        # label_pred: Tensor(n,)
        # score_pred: Tensor(n,)
        # sam_score: Tensor(n,)
        mask_pred_binary = (mask_pred > self.predictor.model.mask_threshold).float()
        if self.use_sam_iou:
            det_scores = score_pred * sam_score
        else:
            # n
            mask_scores_per_image = (mask_pred * mask_pred_binary).flatten(1).sum(1) / (
                    mask_pred_binary.flatten(1).sum(1) + 1e-6)
            det_scores = score_pred * mask_scores_per_image
        # det_scores = score_pred
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = torch.cat([output_boxes, det_scores[:, None]], dim=-1)
        bbox_results = bbox2result(bboxes, label_pred, self.num_classes)
        mask_results = [[] for _ in range(self.num_classes)]
        for j, label in enumerate(label_pred):
            mask = mask_pred_binary[j].detach().cpu().numpy()
            mask_results[label].append(mask)
        output_results = [(bbox_results, mask_results)]
        return output_results

    def get_det_results(self,img_img_metas_dict,rescale=True):
        assert rescale
        img = img_img_metas_dict['img']
        img_metas = img_img_metas_dict['img_metas']
        
        results = self.forward(return_loss=False,img=img,img_metas=img_metas,get_det_results=True)

        image_path = img_metas[0][0]['filename']
        ori_img = cv2.imread(image_path)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        torch_img = self.predictor.get_tor_image(ori_img)
        
        return [results, torch_img]
    def replace_quant_sam(self, quantsam):
        object.__setattr__(self, 'predictor', quantsam)