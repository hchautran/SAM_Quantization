from collections import defaultdict
from typing import Dict, List

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY

from training.utils.distributed import get_world_size, is_dist_avail_and_initialized

def get_default_datasets():
    """Get default dataset configurations"""
    return [
        {
            "name": "DIS5K-VD",
            "im_dir": "./data/DIS5K/DIS-VD/im",
            "gt_dir": "./data/DIS5K/DIS-VD/gt",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        },
        {
            "name": "DIS5K-TR",
            "im_dir": "./data/DIS5K/DIS-TR/im",
            "gt_dir": "./data/DIS5K/DIS-TR/gt",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        },
        {
            "name": "ThinObject5k-TR",
            "im_dir": "./data/thin_object_detection/ThinObject5K/images_train",
            "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train",
            "im_ext": ".jpg",
            "gt_ext": ".png"
        },
        {
            "name": "DIS5K-TR",
            "im_dir": "./data/DIS5K/DIS-TR/im",
            "gt_dir": "./data/DIS5K/DIS-TR/gt",
            "im_ext": ".jpg",
            "gt_ext": ".png"
            },
        
    ]

def get_full_attention_heads(model):
    full_attention_heads = []
    # import ipdb; ipdb.set_trace()
    for layer in model.module.image_encoder.trunk.blocks:
        module = layer.attn
        if not hasattr(module, "full_attention_heads"):
            continue
        full_attention_heads.append(module.full_attention_heads)
    return full_attention_heads
def l1_loss(x):
    numel = x.numel()
    l1 = x.abs().sum()
    return l1 / numel
def get_head_probability(model):
    full_head_probability=[]
    for layer in model.module.image_encoder.trunk.blocks:
        module = layer.attn
        if not hasattr(module, "prune_ddp"):
            continue
        head_probability = module.prune_ddp.get_head_probability_diff_duo()
        full_head_probability.append(head_probability)
    return full_head_probability

class DuoDistillationLoss(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
        distillation_temperature=1.0,
        distillation_alpha=1,
        action_on_loss= "average",
    ):
        """
        Distillation loss for DuoPruneSAM2 training.
        Args:
            weight_dict: dict containing weights for focal, dice, iou losses
            distillation_temperature: Temperature for knowledge distillation
            distillation_alpha: Weight for distillation loss vs task loss
            ... (other args same as MultiStepMultiMasksAndIous)
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.distillation_temperature = distillation_temperature
        self.distillation_alpha = distillation_alpha
        self.action_on_loss = action_on_loss
        
        # Ensure required loss keys exist
        assert "loss_mask_distill" in self.weight_dict
        assert "loss_dice_distill" in self.weight_dict
        assert "loss_iou_distill" in self.weight_dict
       
       

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(
            (targets_batch.shape[1]), device=targets_batch.device, dtype=torch.float
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = torch.clamp(num_objects / get_world_size(), min=1).item()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v
        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        
        batch_size = outputs['multistep_pred_multimasks_high_res'][0].shape[0] //2

        student_masks_list = []
        student_object_score_logits_list = []
        student_ious_list = []
        teacher_masks_list = []
        teacher_ious_list = []
        teacher_object_score_logits_list = []
        for i in range(len(outputs["multistep_pred_multimasks_high_res"])):
            
            student_masks_list.append(outputs["multistep_pred_multimasks_high_res"][i][batch_size:])
            student_ious_list.append(outputs["multistep_pred_ious"][i][batch_size:])
            student_object_score_logits_list.append(outputs["multistep_object_score_logits"][i][batch_size:])
            teacher_masks_list.append(outputs["multistep_pred_multimasks_high_res"][i][:batch_size])
            teacher_ious_list.append(outputs["multistep_pred_ious"][i][:batch_size])
            teacher_object_score_logits_list.append(outputs["multistep_object_score_logits"][i][:batch_size])
        # Compute standard task losses for student
        
        distill_loss = {"loss_mask_distill": 0, "loss_dice_distill": 0, "loss_iou_distill": 0}
        for teacher_masks, student_masks, teacher_ious, student_ious in zip(
            teacher_masks_list, student_masks_list, teacher_ious_list, student_ious_list
        ):
            

            self._compute_distillation_loss(distill_loss ,student_masks , student_ious, teacher_ious, teacher_masks, num_objects)
    
        
        distill_loss[CORE_LOSS_KEY] = self.reduce_loss(distill_loss)
        return distill_loss



    def _compute_distillation_loss(self,distill_loss, student_masks , student_ious, teacher_ious, teacher_masks, num_objects):
        
        
        # Convert teacher masks to soft targets using temperature scaling
        teacher_soft_targets = torch.sigmoid(teacher_masks / self.distillation_temperature)
        
        # Compute raw distillation losses (without averaging/selection)
        loss_multimask = self._compute_focal_distillation_loss(
            student_masks, teacher_soft_targets, num_objects
        )
        loss_multidice = self._compute_dice_distillation_loss(
            student_masks, teacher_soft_targets, num_objects
        )
        loss_multiiou = self._compute_iou_distillation_loss(
            student_ious, teacher_ious, num_objects
        )
        
        # Handle mask selection based on action_on_loss (following loss_fns.py logic)
        if loss_multimask.size(1) > 1:  # Multiple masks
            if self.action_on_loss == "average":
                # Average over all masks
                loss_mask = loss_multimask.mean(dim=-1).unsqueeze(1)  # [N, 1]
                loss_dice = loss_multidice.mean(dim=-1).unsqueeze(1)  # [N, 1]
                
                # IoU loss: use supervise_all_iou logic from original
                if self.supervise_all_iou:
                    loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)  # [N, 1]
                else:
                    loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)  # [N, 1] - for distillation, still average
                
            elif self.action_on_loss == "best":
                # Select best performing masks (following loss_fns.py logic)
                loss_combo = (
                    loss_multimask * self.weight_dict["loss_mask"]
                    + loss_multidice * self.weight_dict["loss_dice"]
                )
                best_loss_inds = torch.argmin(loss_combo, dim=-1)  # [N]
                batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
                
                loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)  # [N, 1]
                loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)  # [N, 1]
                
                # IoU loss: use supervise_all_iou logic from original
                if self.supervise_all_iou:
                    loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)  # [N, 1]
                else:
                    loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)  # [N, 1]
            else:
                raise ValueError(f"Unknown action_on_loss: {self.action_on_loss}. Use 'average' or 'best'")
        else:
            # Single mask case
            loss_mask = loss_multimask  # [N, 1]
            loss_dice = loss_multidice  # [N, 1]
            loss_iou = loss_multiiou    # [N, 1]
        # Accumulate losses (no target_obj masking)
        distill_loss["loss_mask_distill"] += loss_mask.sum()
        distill_loss["loss_dice_distill"] += loss_dice.sum()
        distill_loss["loss_iou_distill"] += loss_iou.sum()
        

    def _compute_focal_distillation_loss(self, student_masks, teacher_soft_targets, num_objects):
        """Compute focal loss for distillation - returns raw multimask losses [N, M]."""
        # Apply temperature scaling to student predictions
        student_scaled = student_masks / self.distillation_temperature
        student_prob = student_scaled.sigmoid()
        
        # Use teacher soft targets for focal loss computation
        ce_loss = F.binary_cross_entropy_with_logits(
            student_scaled, teacher_soft_targets, reduction="none"
        )
        
        # Compute p_t for focal loss (probability of correct prediction)
        p_t = student_prob * teacher_soft_targets + (1 - student_prob) * (1 - teacher_soft_targets)
        
        # Apply focal loss formulation
        focal_loss = ce_loss * ((1 - p_t) ** self.focal_gamma)
        
        # Apply alpha weighting
        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * teacher_soft_targets + (1 - self.focal_alpha) * (1 - teacher_soft_targets)
            focal_loss = alpha_t * focal_loss
        
        # Apply temperature scaling factor
        focal_loss = focal_loss * (self.distillation_temperature ** 2)
        
        # Handle multimask case (same as original sigmoid_focal_loss)
        assert focal_loss.dim() == 4  # [N, M, H, W]
        focal_loss = focal_loss.flatten(2).mean(-1)  # Average over spatial dims -> [N, M]
        focal_loss = focal_loss / num_objects  # Normalize by num_objects
        
        return focal_loss  # [N, M] - no averaging/selection here

    def _compute_dice_distillation_loss(self, student_masks, teacher_soft_targets, num_objects):
        """Compute dice loss for distillation - returns raw multimask losses [N, M]."""
        # Apply temperature scaling and sigmoid to student predictions
        student_scaled = student_masks / self.distillation_temperature
        student_prob = student_scaled.sigmoid()
        
        # Compute dice loss using teacher soft targets
        assert student_prob.dim() == 4 and teacher_soft_targets.dim() == 4
        student_flat = student_prob.flatten(2)  # [N, M, H*W]
        teacher_flat = teacher_soft_targets.flatten(2)  # [N, M, H*W]
        
        # Dice coefficient computation
        numerator = 2 * (student_flat * teacher_flat).sum(-1)  # [N, M]
        denominator = student_flat.sum(-1) + teacher_flat.sum(-1)  # [N, M]
        dice_loss = 1 - (numerator + 1) / (denominator + 1)  # [N, M]
        
        # Apply temperature scaling factor
        dice_loss = dice_loss * (self.distillation_temperature ** 2)
        dice_loss = dice_loss / num_objects  # Normalize by num_objects
        
        return dice_loss  # [N, M] - no averaging/selection here

    def _compute_iou_distillation_loss(self, student_ious, teacher_ious, num_objects):
        """Compute IoU loss for distillation - returns raw multimask losses [N, M]."""
        # Direct loss between IoU predictions
        if self.iou_use_l1_loss:
            iou_loss = F.l1_loss(student_ious, teacher_ious, reduction="none")
        else:
            iou_loss = F.mse_loss(student_ious, teacher_ious, reduction="none")
        
        # Normalize by num_objects
        iou_loss = iou_loss / num_objects  # [N, M] or [N, 1]
        
        return iou_loss  # [N, M] - no averaging/selection here
    
    def reduce_loss(self, losses):
        """Reduce all losses with their respective weights."""
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight
        return reduced_loss