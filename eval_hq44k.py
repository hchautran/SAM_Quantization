#!/usr/bin/env python3
"""Evaluate SAM-2 or SAM-3 on the HQ44k validation set.

Two backbones share the same eval loop:

  --backbone sam2   SAM-2/2.1 hiera, sparsesam patch on the trunk
  --backbone sam3   HuggingFace SAM-3, sparsesam patch on the ViT

Protocol matches the existing per-backbone scripts: GT mask → bbox → predict
mask → mIoU / Boundary IoU at original resolution.

Usage:
    # SAM-2
    python eval_hq44k.py --backbone sam2 \
        --model-cfg configs/sam2.1/sam2.1_hq_hiera_l.yaml \
        --checkpoint ./ckts/sam2.1_hq_hiera_large.pt \
        --algorithms none sparsesam --ratios 0.75 0.5 0.25 \
        --num-samples 470

    # SAM-3
    python eval_hq44k.py --backbone sam3 \
        --sam3-model facebook/sam3 \
        --algorithms none sparsesam --ratios 0.5 0.25 \
        --num-samples 470
"""

import os
import sys
import argparse
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

# Make `train.*` (under sam-hq/) and SAM2 (under sam-hq/sam-hq2/) importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "sam-hq"),
           os.path.join(_HERE, "sam-hq", "sam-hq2")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from train.utils.dataloader import get_im_gt_name_dict, Resize
from data_utils import OnlineDataset
import train.utils.misc as misc
from train.train import compute_iou, compute_boundary_iou
from sam_engine import get_default_datasets


SAM2_VALID_ALGOS = ("none", "sparsesam")
SAM3_VALID_ALGOS = ("none", "sparsesam")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def custom_collate_fn(batch):
    """Variable-sized ori_im fields → keep as Python list."""
    ori_ims = [item['ori_im'] for item in batch]
    out = {}
    for key in batch[0].keys():
        if key == 'ori_im':
            out[key] = ori_ims
        elif key == 'ori_im_path' or key == 'ori_gt_path':
            out[key] = [item[key] for item in batch]
        else:
            try:
                out[key] = torch.stack([item[key] for item in batch])
            except Exception:
                out[key] = [item[key] for item in batch]
    return out


def _to_uint8_hwc(image_chw_tensor: torch.Tensor) -> np.ndarray:
    img = image_chw_tensor.permute(1, 2, 0).cpu().numpy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    return img


def _resize_mask_to(mask_t: torch.Tensor, target_hw: tuple) -> torch.Tensor:
    """Bilinear-then-threshold resize to target size if needed.
    `mask_t` is (1, 1, H, W) float."""
    if mask_t.shape[-2:] == tuple(target_hw):
        return mask_t
    return torch.nn.functional.interpolate(mask_t, size=target_hw, mode="nearest")


# ─────────────────────────────────────────────────────────────────────────────
# Backends — wrap each model behind a common `predict_one(image, box) -> mask`
# ─────────────────────────────────────────────────────────────────────────────

class Sam2Backend:
    """SAM-2/2.1 image predictor with optional sparsesam patch on the trunk."""

    name = "sam2"

    def __init__(self, predictor, amp_dtype: Optional[torch.dtype] = None,
                 use_batch: bool = False, group_size: int = 4,
                 prune_mlp: bool = True):
        self.predictor = predictor
        self.amp_dtype = amp_dtype
        self.use_batch = use_batch
        self.group_size = group_size
        self.prune_mlp = prune_mlp
        self._amp_ctx = (torch.autocast("cuda", dtype=amp_dtype)
                         if amp_dtype is not None else nullcontext())
        # Trunk where the sparsesam patch attaches.
        self._trunk = predictor.model.image_encoder.trunk

    def supports_batch(self) -> bool:
        return True

    def predict_one(self, img_np: np.ndarray, box_xyxy: np.ndarray,
                    target_hw: tuple) -> torch.Tensor:
        """Returns (1, 1, H, W) float mask resized to target_hw."""
        with self._amp_ctx:
            self.predictor.set_image(img_np)
            masks, _, _ = self.predictor.predict(
                point_coords=None, point_labels=None,
                box=box_xyxy, multimask_output=False,
            )
        m = torch.from_numpy(masks).float().unsqueeze(0)        # (1, K, h, w)
        if m.shape[1] > 1:
            m = m[:, 0:1]                                       # (1, 1, h, w)
        return _resize_mask_to(m, target_hw)

    def predict_batch(self, imgs_np: list, boxes_xyxy: list,
                      target_hws: list) -> list:
        """Returns list of (1, 1, H, W) masks resized to each target_hw."""
        with self._amp_ctx:
            self.predictor.set_image_batch(imgs_np)
            masks_batch, _, _ = self.predictor.predict_batch(
                point_coords_batch=None, point_labels_batch=None,
                box_batch=boxes_xyxy, multimask_output=False,
            )
        out = []
        for i, m_np in enumerate(masks_batch):
            m = torch.from_numpy(m_np).float().unsqueeze(0)
            if m.shape[1] > 1:
                m = m[:, 0:1]
            out.append(_resize_mask_to(m, target_hws[i]))
        return out

    def apply_patch(self, algo: str, ratio: float):
        if algo == "none":
            return
        if algo != "sparsesam":
            raise ValueError(f"sam2 only supports {SAM2_VALID_ALGOS}, got {algo!r}")
        from PiToMe.algo.sparsesam.patch.sam2_hiera import apply_patch
        apply_patch(self._trunk, ratio=ratio,
                    group_size=self.group_size, prune_mlp=self.prune_mlp)

    def remove_patch(self):
        try:
            from PiToMe.algo.sparsesam.patch.sam2_hiera import remove_patch
            remove_patch(self._trunk)
        except Exception:
            pass

    def valid_algos(self):
        return SAM2_VALID_ALGOS


class Sam3Backend:
    """HuggingFace SAM-3 with text + box prompt. One model call per box."""

    name = "sam3"

    def __init__(self, model, processor, text_prompt: str = "object",
                 amp_dtype: Optional[torch.dtype] = None,
                 group_size: int = 4, prune_mlp: bool = True,
                 device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.text_prompt = text_prompt
        self.amp_dtype = amp_dtype
        self.group_size = group_size
        self.prune_mlp = prune_mlp
        self.device = device
        self._amp_ctx = (torch.autocast("cuda", dtype=amp_dtype)
                         if amp_dtype is not None else nullcontext())

    def supports_batch(self) -> bool:
        return False

    @staticmethod
    def _xyxy_to_cxcywh_norm(box_xyxy: torch.Tensor, H: int, W: int):
        x1, y1, x2, y2 = box_xyxy.unbind(-1)
        cx = (x1 + x2) / 2.0 / W
        cy = (y1 + y2) / 2.0 / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H
        return torch.stack([cx, cy, w, h], dim=-1)

    @staticmethod
    def _box_iou_xyxy(boxes_pred: torch.Tensor, box_target: torch.Tensor):
        bp, bt = boxes_pred.float(), box_target.float()
        x1 = torch.maximum(bp[:, 0], bt[0])
        y1 = torch.maximum(bp[:, 1], bt[1])
        x2 = torch.minimum(bp[:, 2], bt[2])
        y2 = torch.minimum(bp[:, 3], bt[3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        a_p = (bp[:, 2] - bp[:, 0]).clamp(min=0) * (bp[:, 3] - bp[:, 1]).clamp(min=0)
        a_t = (bt[2] - bt[0]).clamp(min=0) * (bt[3] - bt[1]).clamp(min=0)
        return inter / (a_p + a_t - inter).clamp(min=1e-6)

    def predict_one(self, img_np: np.ndarray, box_xyxy: np.ndarray,
                    target_hw: tuple) -> torch.Tensor:
        pil = Image.fromarray(img_np)
        H, W = pil.height, pil.width
        box_t  = torch.as_tensor(box_xyxy, dtype=torch.float32)
        cxcywh = self._xyxy_to_cxcywh_norm(box_t, H, W).tolist()

        inputs = self.processor(
            images=pil, text=self.text_prompt,
            input_boxes=[[cxcywh]], return_tensors="pt",
        )
        inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                  for k, v in inputs.items()}

        with torch.no_grad(), self._amp_ctx:
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs, threshold=0.0, mask_threshold=0.5, target_sizes=[(H, W)],
        )[0]

        if results["masks"].numel() == 0:
            mask_np = np.zeros((H, W), dtype=np.uint8)
        else:
            prompt = torch.as_tensor(box_xyxy, dtype=torch.float32,
                                     device=results["boxes"].device)
            ious = self._box_iou_xyxy(results["boxes"], prompt)
            if ious.max().item() <= 0.0:
                best = int(results["scores"].argmax().item())
            else:
                best = int(ious.argmax().item())
            mask_np = results["masks"][best].cpu().numpy().astype(np.uint8)

        m = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return _resize_mask_to(m, target_hw)

    def predict_batch(self, imgs_np, boxes_xyxy, target_hws):
        return [self.predict_one(i, b, t)
                for i, b, t in zip(imgs_np, boxes_xyxy, target_hws)]

    def apply_patch(self, algo: str, ratio: float):
        if algo == "none":
            return
        if algo != "sparsesam":
            raise ValueError(f"sam3 only supports {SAM3_VALID_ALGOS}, got {algo!r}")
        from PiToMe.algo.sparsesam.patch.sam3_hf import apply_patch
        apply_patch(self.model, ratio=ratio,
                    group_size=self.group_size, prune_mlp=self.prune_mlp,
                    verbose=False)

    def remove_patch(self):
        try:
            from PiToMe.algo.sparsesam.patch.sam3_hf import remove_patch
            remove_patch(self.model)
        except Exception:
            pass

    def valid_algos(self):
        return SAM3_VALID_ALGOS


# ─────────────────────────────────────────────────────────────────────────────
# Eval loop
# ─────────────────────────────────────────────────────────────────────────────

def eval_hq44k(backend, dataloader: DataLoader, num_samples: int = None,
               use_batch: bool = False) -> dict:
    if use_batch and not backend.supports_batch():
        print(f"[{backend.name}] use_batch ignored (not supported by backend)")
        use_batch = False

    print(f"\n{'='*80}")
    print(f"Evaluating {backend.name} on HQ44k "
          f"{'(Batch)' if use_batch else '(Single)'}")
    print(f"{'='*80}\n")

    ious, b_ious = [], []
    total = 0
    pbar = tqdm(total=len(dataloader), desc="Evaluating")

    for idx, data_val in enumerate(dataloader):
        if num_samples and total >= num_samples:
            break

        images     = data_val['image']             # (B, 3, 1024, 1024)
        labels_val = data_val['label']             # (B, 1, 1024, 1024)
        labels_ori = data_val['ori_label']         # (B, 1, h_ori, w_ori) or list
        B = images.shape[0]

        # Per-sample target HW (from ori_label).
        target_hws = []
        for i in range(B):
            lo = labels_ori[i] if isinstance(labels_ori, list) else labels_ori[i]
            target_hws.append(tuple(lo.shape[-2:]))

        imgs_np = [_to_uint8_hwc(images[i]) for i in range(B)]
        boxes = []
        for i in range(B):
            label = labels_val[i, 0]
            bbox = misc.masks_to_boxes(label.unsqueeze(0))[0].cpu().numpy()
            boxes.append(bbox)

        try:
            if use_batch and B > 1:
                masks = backend.predict_batch(imgs_np, boxes, target_hws)
            else:
                masks = [backend.predict_one(imgs_np[i], boxes[i], target_hws[i])
                         for i in range(B)]
        except Exception as e:
            print(f"\n[warn] batch={idx} failed: {e}")
            import traceback; traceback.print_exc()
            for _ in range(B):
                ious.append(0.0); b_ious.append(0.0)
            total += B
            pbar.update(1)
            continue

        for i in range(B):
            lo = labels_ori[i:i+1] if not isinstance(labels_ori, list) \
                 else labels_ori[i].unsqueeze(0)
            m = masks[i].to(lo.device)
            iou_v  = compute_iou(m, lo)
            biou_v = compute_boundary_iou(m, lo)
            ious.append(iou_v.item()  if torch.is_tensor(iou_v)  else iou_v)
            b_ious.append(biou_v.item() if torch.is_tensor(biou_v) else biou_v)

        total += B
        pbar.update(1)
        if ious:
            pbar.set_postfix(
                miou=f"{float(np.mean(ious)):.4f}",
                biou=f"{float(np.mean(b_ious)):.4f}",
                n=len(ious),
            )
    pbar.close()

    miou  = float(np.mean(ious))   if ious   else 0.0
    mbiou = float(np.mean(b_ious)) if b_ious else 0.0
    print(f"\n  mIoU={miou:.4f}  mBoundary-IoU={mbiou:.4f}  (n={len(ious)})")
    return {
        "miou":             miou,
        "miou_std":         float(np.std(ious))   if ious   else 0.0,
        "boundary_iou":     mbiou,
        "boundary_iou_std": float(np.std(b_ious)) if b_ious else 0.0,
        "num_samples":      len(ious),
    }


def run_sweep(backend, algos, ratios, dataloader, num_samples, use_batch):
    valid = backend.valid_algos()
    for a in algos:
        if a not in valid:
            raise ValueError(f"backend={backend.name} does not support algo "
                             f"{a!r}. Valid: {valid}")

    RATIO_ALGOS = {"sparsesam"}
    configs = []
    for algo in algos:
        if algo in RATIO_ALGOS:
            for r in ratios:
                configs.append((algo, float(r)))
        else:
            configs.append((algo, None))

    all_results = []
    for algo, ratio in configs:
        tag = algo + (f"@r={ratio}" if ratio is not None else "")
        print(f"\n{'#'*80}\n# config: {tag}\n{'#'*80}")

        backend.remove_patch()
        if ratio is not None:
            backend.apply_patch(algo, ratio)

        results = eval_hq44k(backend, dataloader, num_samples=num_samples,
                             use_batch=use_batch)
        all_results.append({"algorithm": algo, "ratio": ratio, "results": results})

    print(f"\n{'='*80}\n  Summary  ({backend.name})\n{'='*80}")
    for r in all_results:
        tag = r["algorithm"] + (f"@r={r['ratio']}" if r["ratio"] is not None else "")
        print(f"  {tag:<24} mIoU={r['results']['miou']:.4f}  "
              f"mBIoU={r['results']['boundary_iou']:.4f}  "
              f"n={r['results']['num_samples']}")
    print(f"{'='*80}\n")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _build_backend(args):
    amp_dtype = {"fp32": None, "fp16": torch.float16,
                 "bf16": torch.bfloat16}[args.dtype]

    if args.backbone == "sam2":
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"SAM-2 checkpoint not found: {args.checkpoint}")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print(f"Loading SAM-2 cfg={args.model_cfg}  ckpt={args.checkpoint}")
        sam2_model = build_sam2(config_file=args.model_cfg,
                                ckpt_path=args.checkpoint, device=args.device)
        return Sam2Backend(SAM2ImagePredictor(sam2_model),
                           amp_dtype=amp_dtype, use_batch=args.use_batch,
                           group_size=args.group_size,
                           prune_mlp=not args.no_mlp_prune)

    if args.backbone == "sam3":
        from transformers.models.sam3.modeling_sam3 import Sam3Model
        from transformers.models.sam3.processing_sam3 import Sam3Processor
        print(f"Loading SAM-3 from {args.sam3_model} ...")
        processor = Sam3Processor.from_pretrained(args.sam3_processor or args.sam3_model)
        model = Sam3Model.from_pretrained(args.sam3_model).to(args.device).eval()
        if args.dtype != "fp32":
            model = model.to(dtype=torch.float16 if args.dtype == "fp16"
                                                 else torch.bfloat16)
        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"Loading fine-tune checkpoint: {args.checkpoint}")
            payload = torch.load(args.checkpoint, map_location="cpu")
            sd = payload.get("trainable_state_dict", payload) \
                 if isinstance(payload, dict) else payload
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"  applied {len(sd)} tensors  "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        return Sam3Backend(model, processor, text_prompt=args.text_prompt,
                           amp_dtype=amp_dtype, group_size=args.group_size,
                           prune_mlp=not args.no_mlp_prune,
                           device=args.device)

    raise ValueError(f"unknown backbone: {args.backbone!r}")


def main():
    p = argparse.ArgumentParser(description="Evaluate SAM-2 or SAM-3 on HQ44k",
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--backbone", type=str, default="sam2",
                   choices=["sam2", "sam3"],
                   help="Which model to evaluate.")

    # Common
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--batch-size",  type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="fp16",
                   choices=["fp32", "fp16", "bf16"])

    # SAM-2 args
    p.add_argument("--model-cfg",  type=str, default="sam2_hiera_l.yaml",
                   help="SAM-2 config (sam2_hiera_l.yaml etc.)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="SAM-2 ckpt path (required for --backbone sam2). For "
                        "sam3, optional fine-tune state dict.")
    p.add_argument("--use-batch", action="store_true",
                   help="(sam2) use SAM-2's native batch predict")

    # SAM-3 args
    p.add_argument("--sam3-model", type=str, default="facebook/sam3")
    p.add_argument("--sam3-processor", type=str, default=None)
    p.add_argument("--text-prompt", type=str, default="object")

    # Patch args
    p.add_argument("--algorithms", type=str, nargs="+", default=["none"],
                   choices=["none", "sparsesam"])
    p.add_argument("--ratios", type=float, nargs="+", default=[0.5])
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--no-mlp-prune", action="store_true")

    args = p.parse_args()

    if args.backbone == "sam2" and not args.checkpoint:
        p.error("--checkpoint is required for --backbone sam2")

    print(f"\n{'='*80}\n  HQ44k Evaluation: {args.backbone}\n"
          f"  device={args.device}  dtype={args.dtype}\n{'='*80}\n")

    backend = _build_backend(args)

    print("Loading dataset ...")
    datasets = get_default_datasets()
    valid_im_gt_list = get_im_gt_name_dict([datasets[0]], flag="valid")
    gos_dataset = OnlineDataset(
        [valid_im_gt_list[0]],
        transform=transforms.Compose([Resize([1024, 1024])]),
        eval_ori_resolution=True,
    )
    dataloader = DataLoader(
        gos_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        collate_fn=custom_collate_fn if args.batch_size > 1 else None,
    )
    print(f"  {len(gos_dataset)} samples  batch_size={args.batch_size}  "
          f"num_samples={args.num_samples or 'all'}\n")

    return run_sweep(
        backend, args.algorithms, args.ratios, dataloader,
        num_samples=args.num_samples, use_batch=args.use_batch,
    )


if __name__ == "__main__":
    main()
