#!/usr/bin/env python3
"""
Evaluate HuggingFace SAM-3 on HQ44k dataset, with optional sparsesam patch.

SAM-3's natural prompt is text + (optional) box. For HQ44k we derive a box
prompt from the GT mask (same as `eval_sam2_hq44k.py`), pass it to the model
along with a generic text query, and take the highest-scoring predicted mask.

Usage:
    python eval_sam3_hq44k.py \
        --model facebook/sam3 \
        --num-samples 100

    python eval_sam3_hq44k.py \
        --algorithms none sparsesam \
        --ratios 0.75 0.5 0.25 \
        --num-samples 470 \
        --batch-size 4
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from contextlib import nullcontext

# Make `train.*` and `data_utils.*` importable from the repo.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "sam-hq"),):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from transformers.models.sam3.modeling_sam3 import Sam3Model
from transformers.models.sam3.processing_sam3 import Sam3Processor

# Local imports
from train.utils.dataloader import get_im_gt_name_dict, Resize
from data_utils import OnlineDataset
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou
from sam_engine import get_default_datasets


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def custom_collate_fn(batch):
    ori_ims = [item['ori_im'] for item in batch]
    collated = {}
    for key in batch[0].keys():
        if key == 'ori_im':
            collated[key] = ori_ims
        elif key == 'ori_im_path' or key == 'ori_gt_path':
            collated[key] = [item[key] for item in batch]
        else:
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except Exception:
                collated[key] = [item[key] for item in batch]
    return collated


def _xyxy_to_cxcywh_norm(box_xyxy: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(x1, y1, x2, y2) pixel → (cx, cy, w, h) normalized in [0, 1]."""
    x1, y1, x2, y2 = box_xyxy.unbind(-1)
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    return torch.stack([cx, cy, w, h], dim=-1)


def _box_iou_xyxy(boxes_pred: torch.Tensor, box_target: torch.Tensor) -> torch.Tensor:
    """boxes_pred: (N, 4), box_target: (4,) — both xyxy. Returns (N,) IoU."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class SAM3Evaluator:
    def __init__(self, model, processor, device="cuda", text_prompt: str = "object"):
        self.model = model
        self.processor = processor
        self.device = device
        self.text_prompt = text_prompt

    def _segment_one(self, img_pil: Image.Image, box_xyxy_pix: np.ndarray,
                     amp_ctx) -> np.ndarray:
        """Run SAM-3 on a single image with one box prompt.
        Returns a binary mask at the image's original resolution."""
        H, W = img_pil.height, img_pil.width

        # Normalize box to (cx, cy, w, h) in [0, 1].
        box_t   = torch.tensor(box_xyxy_pix, dtype=torch.float32)
        cxcywh  = _xyxy_to_cxcywh_norm(box_t, H, W).tolist()       # 4-list
        # SAM-3 expects [image][box][4]
        input_boxes = [[cxcywh]]

        inputs = self.processor(
            images=img_pil,
            text=self.text_prompt,
            input_boxes=input_boxes,
            return_tensors="pt",
        )
        inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

        with torch.no_grad(), amp_ctx:
            outputs = self.model(**inputs)

        # Post-process to instance masks at original resolution.
        results = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.0, mask_threshold=0.5,
            target_sizes=[(H, W)],
        )[0]

        if results["masks"].numel() == 0:
            return np.zeros((H, W), dtype=np.uint8)

        # Pick the predicted mask whose pred_box best matches the prompt box.
        # Confidence (scores) optimizes "is this a `text_prompt`?", which is
        # orthogonal to "does this fit the user's box?". Box-IoU gates on the
        # latter, which is what HQ44k actually measures.
        prompt_box = torch.as_tensor(box_xyxy_pix, dtype=torch.float32,
                                     device=results["boxes"].device)
        ious = _box_iou_xyxy(results["boxes"], prompt_box)
        if ious.max().item() <= 0.0:
            # No overlapping prediction — fall back to top-score (rare edge case).
            best = int(results["scores"].argmax().item())
        else:
            best = int(ious.argmax().item())
        return results["masks"][best].cpu().numpy().astype(np.uint8)

    def eval_hq44k(self, dataloader: DataLoader, num_samples: int = None,
                   amp_dtype: torch.dtype = None):
        amp_ctx = (torch.autocast("cuda", dtype=amp_dtype)
                   if amp_dtype is not None else nullcontext())

        ious, biou = [], []
        total = 0

        pbar = tqdm(dataloader, desc="Evaluating")
        for data in pbar:
            if num_samples and total >= num_samples:
                break

            images     = data['image']                # (B, 3, 1024, 1024)
            labels_val = data['label']                # (B, 1, 1024, 1024)
            labels_ori = data['ori_label']            # (B, 1, h_ori, w_ori)

            B = images.shape[0]

            for i in range(B):
                img_np = images[i].permute(1, 2, 0).cpu().numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
                img_pil = Image.fromarray(img_np)

                # Box prompt from GT mask (same protocol as SAM2 eval).
                label = labels_val[i, 0]
                bbox  = misc.masks_to_boxes(label.unsqueeze(0))[0].cpu().numpy()  # xyxy

                try:
                    mask_np = self._segment_one(img_pil, bbox, amp_ctx)
                except Exception as e:
                    print(f"\n[warn] failed on idx={total + i}: {e}")
                    mask_np = np.zeros((img_pil.height, img_pil.width), dtype=np.uint8)

                # Resize to GT resolution if needed (SAM-3 already returned at that size).
                m = torch.from_numpy(mask_np).to(labels_ori.device).unsqueeze(0).unsqueeze(0).float()
                if m.shape[-2:] != labels_ori[i:i+1].shape[-2:]:
                    m = torch.nn.functional.interpolate(
                        m, size=labels_ori[i:i+1].shape[-2:], mode="nearest",
                    )
                iou_v   = compute_iou(m, labels_ori[i:i+1])
                biou_v  = compute_boundary_iou(m, labels_ori[i:i+1])
                ious.append(iou_v.item()  if torch.is_tensor(iou_v)  else iou_v)
                biou.append(biou_v.item() if torch.is_tensor(biou_v) else biou_v)

            total += B
            if ious:
                pbar.set_postfix(
                    miou=f"{float(np.mean(ious)):.4f}",
                    biou=f"{float(np.mean(biou)):.4f}",
                    n=len(ious),
                )

        miou  = float(np.mean(ious)) if ious else 0.0
        mbiou = float(np.mean(biou)) if biou else 0.0
        print(f"\n  mIoU={miou:.4f}  mBoundary-IoU={mbiou:.4f}  (n={len(ious)})")
        return {"miou": miou, "mboundary_iou": mbiou, "n": len(ious)}


# ─────────────────────────────────────────────────────────────────────────────
# Patch helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_sparsesam(model, ratio: float, group_size: int, prune_mlp: bool):
    from PiToMe.algo.sparsesam.patch.sam3_hf import apply_patch
    apply_patch(
        model,
        ratio=ratio,
        group_size=group_size,
        prune_mlp=prune_mlp,
    )


def _remove_sparsesam(model):
    try:
        from PiToMe.algo.sparsesam.patch.sam3_hf import remove_patch
        remove_patch(model)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Evaluate SAM3 on HQ44k')
    p.add_argument('--model', type=str, default='facebook/sam3',
                   help='HuggingFace model id (e.g. facebook/sam3, facebook/sam3-base)')
    p.add_argument('--processor', type=str, default=None,
                   help='Processor id (defaults to --model)')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Path to a .pt produced by train_sam3_hq44k.py — applied '
                        'on top of the pretrained model before eval.')
    p.add_argument('--text-prompt', type=str, default='object',
                   help='Generic text prompt passed alongside the box.')
    p.add_argument('--batch-size',  type=int, default=1)
    p.add_argument('--num-samples', type=int, default=None)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--device',      type=str, default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--dtype',       type=str, default='fp16',
                   choices=['fp32', 'fp16', 'bf16'],
                   help='Inference precision via autocast.')

    p.add_argument('--algorithms', type=str, nargs='+', default=['none'],
                   choices=['none', 'sparsesam'])
    p.add_argument('--ratios',     type=float, nargs='+', default=[0.5])
    p.add_argument('--group-size', type=int, default=4)
    p.add_argument('--no-mlp-prune', action='store_true')

    args = p.parse_args()

    print(f"\n{'='*80}")
    print(f"  SAM3 HQ44k Evaluation")
    print(f"  model={args.model}  device={args.device}  dtype={args.dtype}")
    print(f"{'='*80}\n")

    print("Loading SAM3 model + processor ...")
    # AutoModel/AutoProcessor on "facebook/sam3" returns the *Video* variants,
    # which expect a video inference session and have a kwarg-forwarding bug
    # for text. Use the image-level classes directly.
    processor = Sam3Processor.from_pretrained(args.processor or args.model)
    model = Sam3Model.from_pretrained(args.model)
    if args.checkpoint:
        print(f"Loading fine-tune checkpoint: {args.checkpoint}")
        payload = torch.load(args.checkpoint, map_location="cpu")
        sd = payload["trainable_state_dict"] if isinstance(payload, dict) and "trainable_state_dict" in payload else payload
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  applied {len(sd)} tensors  (missing={len(missing)}, unexpected={len(unexpected)})")
    model = model.to(args.device).eval()
    if args.dtype != 'fp32':
        model = model.to(dtype=torch.float16 if args.dtype == 'fp16' else torch.bfloat16)
    print("Model loaded\n")

    print("Loading dataset ...")
    datasets = get_default_datasets()
    valid_im_gt_list = get_im_gt_name_dict([datasets[0]], flag="valid")
    gos_dataset = OnlineDataset(
        [valid_im_gt_list[0]],
        transform=transforms.Compose([Resize([1024, 1024])]),
        eval_ori_resolution=True,
    )
    dataloader = DataLoader(
        gos_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
        collate_fn=custom_collate_fn if args.batch_size > 1 else None,
    )
    print(f"Dataset loaded: {len(gos_dataset)} samples  "
          f"batch_size={args.batch_size}  "
          f"num_samples={args.num_samples or 'all'}\n")

    # Build (algo, ratio) configs to sweep.
    RATIO_ALGOS = {'sparsesam'}
    configs = []
    for algo in args.algorithms:
        if algo in RATIO_ALGOS:
            for r in args.ratios:
                configs.append((algo, float(r)))
        else:
            configs.append((algo, None))

    amp_dtype = {'fp32': None, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.dtype]
    evaluator = SAM3Evaluator(model, processor, device=args.device,
                              text_prompt=args.text_prompt)

    all_results = []
    for algo, ratio in configs:
        tag = algo + (f"@r={ratio}" if ratio is not None else "")
        print(f"\n{'#'*80}\n# config: {tag}\n{'#'*80}")

        _remove_sparsesam(model)        # idempotent reset
        if algo == 'sparsesam':
            _apply_sparsesam(
                model,
                ratio=ratio,
                group_size=args.group_size,
                prune_mlp=not args.no_mlp_prune,
            )

        results = evaluator.eval_hq44k(
            dataloader=dataloader,
            num_samples=args.num_samples,
            amp_dtype=amp_dtype,
        )
        all_results.append({"algorithm": algo, "ratio": ratio, "results": results})

    print(f"\n{'='*80}\n  Summary\n{'='*80}")
    for r in all_results:
        tag = r["algorithm"] + (f"@r={r['ratio']}" if r["ratio"] is not None else "")
        print(f"  {tag:<24} mIoU={r['results']['miou']:.4f}  "
              f"mBIoU={r['results']['mboundary_iou']:.4f}  n={r['results']['n']}")
    print(f"{'='*80}\n")

    return all_results


if __name__ == '__main__':
    main()
