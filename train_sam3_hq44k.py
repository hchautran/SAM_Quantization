#!/usr/bin/env python3
"""
Fine-tune SAM-3's mask decoder on HQ44k (frozen vision tower + text encoder +
DETR + geometry encoder).

Recipe (mirrors sam-hq's `train_hq.py`):
  • Freeze everything except `model.mask_decoder.*`.
  • For each image: pass GT-derived box prompt + `text_prompt` ("object").
  • Match prediction queries to the GT box by box-IoU (single-instance HQ44k),
    take the matched mask logits, supervise vs GT mask with focal + dice
    using `train.utils.loss_mask.loss_masks`.
  • Save unfrozen state_dict every `--save-every` epochs to `--output-dir`.

The eval script (`eval_sam3_hq44k.py`) gains a `--checkpoint` arg that loads
this state_dict on top of the pretrained model before evaluation.

Usage:
    python train_sam3_hq44k.py --epochs 8 --batch-size 4 --lr 1e-5
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from contextlib import nullcontext

# Local repo imports
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "sam-hq"),):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from transformers.models.sam3.modeling_sam3 import Sam3Model
from transformers.models.sam3.processing_sam3 import Sam3Processor

from train.utils.dataloader import (
    get_im_gt_name_dict, Resize, RandomHFlip, LargeScaleJitter,
    OnlineDataset,
)
from train.utils.loss_mask import loss_masks
import train.utils.misc as misc
from sam_engine import get_default_datasets


# ─────────────────────────────────────────────────────────────────────────────
# Train/val dataset configs — mirror sam-hq/train/train.py
# ─────────────────────────────────────────────────────────────────────────────

def get_hq44k_train_datasets():
    """Train splits used by SAM-HQ. We keep DIS-TR + ThinObject5K-TR
    (binary masks); FSS / DUTS-TR / ECSSD / MSRA10K rely on cascade_psp soft
    masks that need thresholding before use as binary GT — drop here for
    simplicity. Add them back when you have thresholded copies."""
    return [
        {
            "name": "DIS5K-TR",
            "im_dir": "./data/DIS5K/DIS-TR/im",
            "gt_dir": "./data/DIS5K/DIS-TR/gt",
            "im_ext": ".jpg", "gt_ext": ".png",
        },
        {
            "name": "ThinObject5K-TR",
            "im_dir": "./data/thin_object_detection/ThinObject5K/images_train",
            "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train",
            "im_ext": ".jpg", "gt_ext": ".png",
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xyxy_to_cxcywh_norm(box_xyxy: torch.Tensor, H: int, W: int) -> torch.Tensor:
    x1, y1, x2, y2 = box_xyxy.unbind(-1)
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    return torch.stack([cx, cy, w, h], dim=-1)


def _box_iou_xyxy(boxes_pred: torch.Tensor, box_target: torch.Tensor) -> torch.Tensor:
    bp = boxes_pred.float()
    bt = box_target.float()
    x1 = torch.maximum(bp[:, 0], bt[0])
    y1 = torch.maximum(bp[:, 1], bt[1])
    x2 = torch.minimum(bp[:, 2], bt[2])
    y2 = torch.minimum(bp[:, 3], bt[3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a_pred = (bp[:, 2] - bp[:, 0]).clamp(min=0) * (bp[:, 3] - bp[:, 1]).clamp(min=0)
    a_targ = (bt[2] - bt[0]).clamp(min=0) * (bt[3] - bt[1]).clamp(min=0)
    union = a_pred + a_targ - inter
    return inter / union.clamp(min=1e-6)


def freeze_for_finetune(model: Sam3Model, train_modules=("mask_decoder",)):
    """Freeze every param whose name does NOT start with one of `train_modules`."""
    n_train, n_freeze = 0, 0
    for name, p in model.named_parameters():
        if any(name.startswith(prefix) for prefix in train_modules):
            p.requires_grad_(True)
            n_train += p.numel()
        else:
            p.requires_grad_(False)
            n_freeze += p.numel()
    print(f"  trainable: {n_train/1e6:.2f}M  |  frozen: {n_freeze/1e6:.2f}M  |  "
          f"modules: {train_modules}")
    return [p for p in model.parameters() if p.requires_grad]


def trainable_state_dict(model: Sam3Model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if model.get_parameter(k).requires_grad if k in dict(model.named_parameters())}


def save_checkpoint(model: Sam3Model, path: str, extra: dict = None):
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()
          if k in {n for n, p in model.named_parameters() if p.requires_grad}}
    payload = {"trainable_state_dict": sd}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


# ─────────────────────────────────────────────────────────────────────────────
# Build inputs + select supervised query
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _prepare_batch(processor: Sam3Processor, images_t: torch.Tensor,
                   labels_val: torch.Tensor, text_prompt: str, device: str):
    """Build a batched processor input from the in-memory image tensor + GT
    masks. Returns `(inputs_dict, gt_box_xyxy_pixel, original_size)`.

    `images_t`     — (B, 3, 1024, 1024) float in [0, 1] or [0, 255]
    `labels_val`   — (B, 1, 1024, 1024) GT mask at 1024x1024
    """
    B, _, H, W = images_t.shape
    pil_imgs, boxes_xyxy_pix, boxes_norm = [], [], []
    for i in range(B):
        img_np = images_t[i].permute(1, 2, 0).cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        pil_imgs.append(Image.fromarray(img_np))

        # GT-derived box (xyxy in pixels at 1024×1024).
        bx = misc.masks_to_boxes(labels_val[i, 0:1])[0].cpu()
        boxes_xyxy_pix.append(bx)
        boxes_norm.append(_xyxy_to_cxcywh_norm(bx, H, W).tolist())

    input_boxes = [[b] for b in boxes_norm]                       # [B][1 box][4]

    inputs = processor(
        images=pil_imgs,
        text=[text_prompt] * B,
        input_boxes=input_boxes,
        return_tensors="pt",
    )
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    return inputs, torch.stack(boxes_xyxy_pix).to(device)         # boxes at 1024×1024


def _resize_mask_logits(mask_logits: torch.Tensor, target_hw) -> torch.Tensor:
    """`mask_logits`: (..., h, w) → bilinear-resized to `target_hw`."""
    if mask_logits.shape[-2:] == tuple(target_hw):
        return mask_logits
    while mask_logits.dim() < 4:
        mask_logits = mask_logits.unsqueeze(0)
    out = F.interpolate(mask_logits, size=tuple(target_hw),
                         mode="bilinear", align_corners=False)
    return out


def _select_query(pred_boxes_xyxy_at_1024: torch.Tensor,
                  gt_box_xyxy_at_1024: torch.Tensor) -> int:
    """Hungarian-1 match: query index with highest box-IoU vs GT box."""
    ious = _box_iou_xyxy(pred_boxes_xyxy_at_1024, gt_box_xyxy_at_1024)
    return int(ious.argmax().item())


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, processor, dataloader, optimizer, *,
                    device, amp_dtype, text_prompt, scaler, epoch, max_iters=None):
    model.train()
    amp_ctx = (torch.autocast("cuda", dtype=amp_dtype)
               if amp_dtype is not None else nullcontext())

    losses, mask_losses, dice_losses = [], [], []
    pbar = tqdm(dataloader, desc=f"train epoch {epoch}", leave=False)
    n_iter = 0

    for data in pbar:
        if max_iters is not None and n_iter >= max_iters:
            break

        images     = data['image'].to(device)            # (B, 3, 1024, 1024)
        labels_val = data['label'].to(device)            # (B, 1, 1024, 1024)
        B = images.shape[0]

        inputs, gt_boxes_at_1024 = _prepare_batch(
            processor, images, labels_val, text_prompt, device,
        )

        with amp_ctx:
            outputs = model(**inputs)

            # outputs.pred_masks:  (B, Q, h, w) at processor's target resolution
            # outputs.pred_boxes:  (B, Q, 4)    in xyxy at the processor's target_size
            pred_masks = outputs.pred_masks                                # (B, Q, h, w)
            pred_boxes = outputs.pred_boxes                                # (B, Q, 4)

            # Processor target_size (square): use the mask spatial size as proxy.
            tgt = float(pred_masks.shape[-1])
            # Convert pred boxes from processor coords (target_size) to 1024
            # for matching against gt_boxes_at_1024.
            scale = images.shape[-1] / tgt
            pred_boxes_at_1024 = pred_boxes * scale

            # Per-sample query selection + per-sample loss accumulation.
            sel_logits, sel_targets = [], []
            for b in range(B):
                q = _select_query(pred_boxes_at_1024[b], gt_boxes_at_1024[b])
                m = pred_masks[b, q].unsqueeze(0).unsqueeze(0)              # (1, 1, h, w)
                m_up = _resize_mask_logits(m, labels_val.shape[-2:])         # (1,1,1024,1024)
                sel_logits.append(m_up.squeeze(0))                           # (1, 1024, 1024)
                tgt_mask = (labels_val[b] / 255.0).clamp(0, 1).float()       # (1, 1024, 1024)
                sel_targets.append(tgt_mask)

            sel_logits  = torch.stack(sel_logits)                            # (B, 1, 1024, 1024)
            sel_targets = torch.stack(sel_targets)                           # (B, 1, 1024, 1024)
            loss_mask, loss_dice = loss_masks(sel_logits, sel_targets, num_masks=B)
            loss = loss_mask + loss_dice

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        mask_losses.append(loss_mask.item())
        dice_losses.append(loss_dice.item())
        pbar.set_postfix(
            loss=f"{np.mean(losses[-50:]):.4f}",
            mask=f"{np.mean(mask_losses[-50:]):.4f}",
            dice=f"{np.mean(dice_losses[-50:]):.4f}",
        )
        n_iter += 1

    return float(np.mean(losses)) if losses else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch validation (SAM-HQ-style mIoU + boundary IoU)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, processor, dataloader, *, device, amp_dtype, text_prompt,
             max_iters=None):
    from train.train import compute_iou, compute_boundary_iou
    model.eval()
    amp_ctx = (torch.autocast("cuda", dtype=amp_dtype)
               if amp_dtype is not None else nullcontext())

    ious, biou = [], []
    pbar = tqdm(dataloader, desc="val", leave=False)
    n_iter = 0
    for data in pbar:
        if max_iters is not None and n_iter >= max_iters:
            break
        images     = data['image'].to(device)
        labels_val = data['label'].to(device)
        labels_ori = data['ori_label'].to(device)
        B = images.shape[0]

        inputs, gt_boxes_at_1024 = _prepare_batch(
            processor, images, labels_val, text_prompt, device,
        )
        with amp_ctx:
            outputs = model(**inputs)

        pred_masks = outputs.pred_masks                   # (B, Q, h, w)
        pred_boxes = outputs.pred_boxes                   # (B, Q, 4)
        scale = images.shape[-1] / float(pred_masks.shape[-1])
        pred_boxes_at_1024 = pred_boxes * scale

        for b in range(B):
            q = _select_query(pred_boxes_at_1024[b], gt_boxes_at_1024[b])
            m = pred_masks[b, q].unsqueeze(0).unsqueeze(0)       # (1, 1, h, w)
            m = (m.sigmoid() > 0.5).float()
            m = F.interpolate(m, size=labels_ori[b:b+1].shape[-2:], mode="nearest")
            iou_v  = compute_iou(m, labels_ori[b:b+1])
            biou_v = compute_boundary_iou(m, labels_ori[b:b+1])
            ious.append(iou_v.item()  if torch.is_tensor(iou_v)  else iou_v)
            biou.append(biou_v.item() if torch.is_tensor(biou_v) else biou_v)

        if ious:
            pbar.set_postfix(
                miou=f"{np.mean(ious):.4f}",
                biou=f"{np.mean(biou):.4f}",
                n=len(ious),
            )
        n_iter += 1

    return float(np.mean(ious)) if ious else 0.0, float(np.mean(biou)) if biou else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='facebook/sam3')
    p.add_argument('--processor', default=None)
    p.add_argument('--text-prompt', default='object')
    p.add_argument('--output-dir', default='./checkpoints/sam3_hq44k')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--lr-drop-epoch', type=int, default=6)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--max-iters-per-epoch', type=int, default=None)
    p.add_argument('--val-iters', type=int, default=200,
                   help='Cap validation at this many iters per epoch (None = full).')
    p.add_argument('--save-every', type=int, default=1)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', default='fp16',
                   choices=['fp32', 'fp16', 'bf16'],
                   help="Autocast dtype. fp32 disables autocast (no GradScaler used either).")
    p.add_argument('--train-modules', nargs='+', default=['mask_decoder'],
                   help="Param-name prefixes to keep trainable; everything else frozen.")
    p.add_argument('--seed', type=int, default=42)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # ── load ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}\n  Fine-tuning SAM-3 mask decoder on HQ44k\n{'='*80}")
    print(f"  model={args.model}  device={args.device}  dtype={args.dtype}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}\n")

    print("Loading SAM-3 model + processor ...")
    processor = Sam3Processor.from_pretrained(args.processor or args.model)
    model = Sam3Model.from_pretrained(args.model).to(args.device)
    print("Model loaded.\n")

    # Freeze everything except mask decoder.
    print("Freezing for fine-tune:")
    train_params = freeze_for_finetune(model, train_modules=tuple(args.train_modules))

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        train_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)

    amp_dtype = {'fp32': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.dtype]
    scaler = torch.cuda.amp.GradScaler() if args.dtype == 'fp16' else None

    # ── data ──────────────────────────────────────────────────────────────────
    print("Loading HQ44k train splits (SAM-HQ recipe: DIS-TR + ThinObject5K-TR) ...")
    train_cfgs = get_hq44k_train_datasets()
    train_im_gt_list = get_im_gt_name_dict(train_cfgs, flag="train")
    # SAM-HQ uses RandomHFlip + LargeScaleJitter for training; jitter outputs
    # already at the model's input size (1024) so no extra Resize is needed.
    train_dataset = OnlineDataset(
        train_im_gt_list,
        transform=transforms.Compose([
            RandomHFlip(),
            LargeScaleJitter(output_size=1024),
        ]),
        eval_ori_resolution=False,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
    )
    print(f"  train_dataset n={len(train_dataset)}  iters/epoch≈{len(train_loader)}")

    # Validation split = whatever sam_engine.get_default_datasets() returns
    # (HQ44k VAL: DIS5K-VD + ThinObject5K-TE by default).
    print("Loading HQ44k validation split ...")
    val_cfgs = get_default_datasets()
    val_im_gt_list = get_im_gt_name_dict(val_cfgs, flag="valid")
    val_dataset = OnlineDataset(
        val_im_gt_list,
        transform=transforms.Compose([Resize([1024, 1024])]),
        eval_ori_resolution=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
    )
    print(f"  val_dataset n={len(val_dataset)}\n")

    # ── train ─────────────────────────────────────────────────────────────────
    best_miou = -1.0
    for epoch in range(args.epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(
            model, processor, train_loader, optimizer,
            device=args.device, amp_dtype=amp_dtype, text_prompt=args.text_prompt,
            scaler=scaler, epoch=epoch, max_iters=args.max_iters_per_epoch,
        )
        scheduler.step()
        dt_train = time.time() - t0

        # Per-epoch validation (capped via --val-iters for speed).
        t1 = time.time()
        val_miou, val_biou = validate(
            model, processor, val_loader,
            device=args.device, amp_dtype=amp_dtype, text_prompt=args.text_prompt,
            max_iters=args.val_iters,
        )
        dt_val = time.time() - t1

        print(f"  epoch {epoch}  loss={avg_loss:.4f}  "
              f"val_miou={val_miou:.4f}  val_biou={val_biou:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"({dt_train:.1f}s train + {dt_val:.1f}s val)")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch}.pt")
            save_checkpoint(model, ckpt_path, extra={
                "epoch": epoch, "avg_loss": avg_loss,
                "val_miou": val_miou, "val_biou": val_biou,
                "args": vars(args),
            })
            print(f"  saved {ckpt_path}")

        if val_miou > best_miou:
            best_miou = val_miou
            best_path = os.path.join(args.output_dir, "best.pt")
            save_checkpoint(model, best_path, extra={
                "epoch": epoch, "val_miou": val_miou, "val_biou": val_biou,
                "args": vars(args),
            })
            print(f"  ★ new best  val_miou={val_miou:.4f}  → {best_path}")

    final_path = os.path.join(args.output_dir, "final.pt")
    save_checkpoint(model, final_path, extra={"args": vars(args)})
    print(f"\n  Final checkpoint: {final_path}")


if __name__ == '__main__':
    main()
