#!/usr/bin/env python3
"""Evaluate SAM-HQ or SAM-3 on COCO val2017 with token-merging patches.

Protocol: GT-box prompt (per-instance) → predict mask → mIoU / Boundary IoU.
Optionally accumulates predictions and computes COCO segm AP via pycocotools.

Two backbones share the same eval loop:

  --backbone sam-hq   SAM-HQ ViT, full ToMe sweep (none/tome/pitome/sparsesam/gradtome)
  --backbone sam3     HuggingFace SAM-3, sparsesam patch only

Encoder runs once per image (SAM-HQ); decoder loops over all instance boxes.
For SAM-3 the processor + model run per box (per-image timing reported).

Usage:
    # SAM-HQ
    python eval_coco.py --backbone sam-hq \
        --algos none tome pitome sparsesam gradtome \
        --ratios 0.9 0.8 0.7 \
        --num-images 200

    # SAM-3
    python eval_coco.py --backbone sam3 \
        --sam3-model facebook/sam3 \
        --algos none sparsesam \
        --ratios 0.5 0.25 \
        --num-images 200
"""

import os
import sys
import gc
import time
import json
import argparse
import datetime
from contextlib import nullcontext
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# SAM
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sam-hq'))
from segment_anything import SamPredictor, sam_model_registry

# Reuse patch registry + helpers from the existing benchmark
from benchmark_tome import (
    ALGO_REGISTRY, VALID_ALGOS, apply_tome, remove_tome, reset_memory,
)
from coco_dataset import CocoInstanceDataset, coco_collate_fn
from train.train import compute_iou, compute_boundary_iou


SAM3_VALID_ALGOS = ("none", "sparsesam")


# ──────────────────────────────────────────────────────────────────────────────
# Backends — wrap each model behind the same predict_image(image, boxes) call
# ──────────────────────────────────────────────────────────────────────────────

class SamHQBackend:
    """SAM-HQ ViT: encoder once per image + per-instance decode."""

    name = "sam-hq"

    def __init__(self, predictor: SamPredictor):
        self.predictor = predictor
        self.device = predictor.device
        self.encoder = predictor.model.image_encoder

    def warmup(self, image_chw_uint8: torch.Tensor):
        with torch.no_grad():
            x = image_chw_uint8.to(self.device)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            t = self.predictor.model.preprocess(x).half()
            self.encoder(t)

    def predict_image(self, image_chw_uint8, ori_im_hwc, original_hw,
                      boxes_xyxy_orig, cat_names=None):
        """Returns dict with masks (N,H,W) bool at original res, scores (N,),
        encoder_ms (float). `ori_im_hwc` is unused for SAM-HQ."""
        del ori_im_hwc
        H, W = original_hw
        image = image_chw_uint8.to(self.device)

        transformed = self.predictor.model.preprocess(image.unsqueeze(0)).half()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            features, interm_features = self.encoder(transformed)
        torch.cuda.synchronize()
        encoder_ms = (time.perf_counter() - t0) * 1000

        self.predictor.features        = features
        self.predictor.interm_features = interm_features
        self.predictor.original_size   = (int(H), int(W))
        self.predictor.input_size      = tuple(transformed.shape[-2:])
        self.predictor.is_image_set    = True

        sx = 1024.0 / float(W)
        sy = 1024.0 / float(H)
        boxes_scaled = boxes_xyxy_orig.to(self.device).clone()
        boxes_scaled[:, 0::2] *= sx
        boxes_scaled[:, 1::2] *= sy
        boxes_scaled = boxes_scaled.to(torch.float16)

        with torch.no_grad():
            masks, scores, _ = self.predictor.predict_torch(
                point_coords=None, point_labels=None,
                boxes=boxes_scaled, multimask_output=False, hq_token_only=True,
            )  # (N, 1, H, W) bool

        return {
            "masks":      masks.squeeze(1).bool(),       # (N, H, W) at orig
            "scores":     scores.squeeze(-1).float(),    # (N,)
            "encoder_ms": encoder_ms,
        }

    def apply_patch(self, algo, ratio, margin):
        if algo == "none":
            return
        apply_tome(self.encoder, algo=algo, ratio=ratio, margin=margin)

    def remove_patch(self):
        remove_tome(self.encoder)

    def valid_algos(self):
        return VALID_ALGOS


class Sam3Backend:
    """HuggingFace SAM-3 with text + box prompt. One model call per box."""

    name = "sam3"

    def __init__(self, model, processor, text_prompt: str = "object",
                 amp_dtype: Optional[torch.dtype] = None,
                 use_category_text: bool = False,
                 device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.default_text = text_prompt
        self.amp_dtype = amp_dtype
        self.use_category_text = use_category_text
        self.device = device
        self._amp_ctx = (
            torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None
            else nullcontext()
        )

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

    def warmup(self, image_chw_uint8: torch.Tensor):
        img_np = image_chw_uint8.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        pil = Image.fromarray(img_np)
        H, W = pil.height, pil.width
        box = torch.tensor([0.5, 0.5, 1.0, 1.0])
        inputs = self.processor(
            images=pil, text=self.default_text,
            input_boxes=[[box.tolist()]], return_tensors="pt",
        )
        inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                  for k, v in inputs.items()}
        with torch.no_grad(), self._amp_ctx:
            self.model(**inputs)

    def _segment_one(self, pil_img: Image.Image, box_xyxy_pix: np.ndarray,
                     text: str):
        H, W = pil_img.height, pil_img.width
        box_t  = torch.as_tensor(box_xyxy_pix, dtype=torch.float32)
        cxcywh = self._xyxy_to_cxcywh_norm(box_t, H, W).tolist()
        inputs = self.processor(
            images=pil_img, text=text,
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
            return np.zeros((H, W), dtype=np.uint8), 0.0

        prompt_box = torch.as_tensor(box_xyxy_pix, dtype=torch.float32,
                                     device=results["boxes"].device)
        ious = self._box_iou_xyxy(results["boxes"], prompt_box)
        if ious.max().item() <= 0.0:
            best = int(results["scores"].argmax().item())
        else:
            best = int(ious.argmax().item())
        m = results["masks"][best].cpu().numpy().astype(np.uint8)
        s = float(results["scores"][best].item())
        return m, s

    def predict_image(self, image_chw_uint8, ori_im_hwc, original_hw,
                      boxes_xyxy_orig, cat_names=None):
        del image_chw_uint8, original_hw
        H, W = ori_im_hwc.shape[:2]
        pil = Image.fromarray(ori_im_hwc)
        N = boxes_xyxy_orig.shape[0]
        boxes_np = boxes_xyxy_orig.cpu().numpy()

        masks_out = np.zeros((N, H, W), dtype=np.uint8)
        scores_out = np.zeros((N,), dtype=np.float32)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for i in range(N):
            text = (cat_names[i] if (self.use_category_text and cat_names
                                     and cat_names[i]) else self.default_text)
            try:
                m, s = self._segment_one(pil, boxes_np[i], text)
            except Exception as e:
                print(f"  sam3 decode error: {e}")
                m, s = np.zeros((H, W), dtype=np.uint8), 0.0
            masks_out[i] = m
            scores_out[i] = s
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        per_img_ms = (time.perf_counter() - t0) * 1000

        masks_t = torch.from_numpy(masks_out).bool()
        scores_t = torch.from_numpy(scores_out)
        return {
            "masks":      masks_t,
            "scores":     scores_t,
            "encoder_ms": per_img_ms,    # whole-image cost (encoder+decoder*N)
        }

    def apply_patch(self, algo, ratio, margin):
        if algo == "none":
            return
        if algo != "sparsesam":
            raise ValueError(f"SAM-3 only supports algos {SAM3_VALID_ALGOS}, "
                             f"got {algo!r}")
        from PiToMe.algo.sparsesam.patch.sam3_hf import apply_patch
        apply_patch(self.model, ratio=ratio, group_size=4, prune_mlp=True,
                    verbose=False)

    def remove_patch(self):
        try:
            from PiToMe.algo.sparsesam.patch.sam3_hf import remove_patch
            remove_patch(self.model)
        except Exception:
            pass

    def valid_algos(self):
        return SAM3_VALID_ALGOS


# ──────────────────────────────────────────────────────────────────────────────
# Per-image evaluation
# ──────────────────────────────────────────────────────────────────────────────

def eval_image(backend, sample, coco_predictions: list | None = None,
               cat_id_to_name: dict | None = None) -> Dict:
    """Encode + decode one image. Returns timing / IoU lists."""
    image       = sample["image"][0]                          # (3, 1024, 1024)
    ori_im      = sample["ori_im"][0]                         # HWC uint8 numpy
    H, W        = sample["ori_size"][0]
    boxes_ori   = sample["boxes_ori"][0]                       # (N, 4) xyxy
    masks_ori   = sample["masks_ori"][0]                       # (N, H, W) uint8
    image_id    = sample["image_id"][0]
    ann_ids     = sample["ann_ids"][0]
    cat_ids     = sample["cat_ids"][0]
    det_scores  = sample.get("scores", [[1.0] * boxes_ori.shape[0]])[0]
    box_source  = sample.get("box_source", ["gt"])[0]
    n_inst      = boxes_ori.shape[0]

    cat_names = None
    if cat_id_to_name is not None:
        cat_names = [cat_id_to_name.get(int(c), None) for c in cat_ids]

    try:
        out = backend.predict_image(image, ori_im, (int(H), int(W)),
                                    boxes_ori, cat_names)
    except Exception as e:
        print(f"  decode error img {image_id}: {e}")
        return {"encoder_ms": 0.0, "n_inst": 0, "iou": [], "biou": []}

    masks      = out["masks"]              # (N, H, W) bool, original res
    pred_scores = out["scores"]            # (N,)
    encoder_ms = out["encoder_ms"]

    masks_ori_dev = masks_ori.to(masks.device)
    ious, b_ious = [], []

    for i in range(n_inst):
        pr = masks[i:i+1].unsqueeze(0).float()              # (1,1,H,W)

        if box_source == "gt" and ann_ids[i] != -1:
            gt = masks_ori_dev[i:i+1].unsqueeze(0).float()
            ious.append(compute_iou(pr, gt).item())
            b_ious.append(compute_boundary_iou(pr, gt).item())

        if coco_predictions is not None and cat_ids[i] != -1:
            from pycocotools import mask as mask_utils
            m_np = masks[i].cpu().numpy().astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(m_np))
            rle["counts"] = rle["counts"].decode("ascii")
            score = float(det_scores[i]) if box_source == "detections" \
                    else float(pred_scores[i].item())
            coco_predictions.append({
                "image_id":     int(image_id),
                "category_id":  int(cat_ids[i]),
                "segmentation": rle,
                "score":        score,
            })

    return {
        "encoder_ms": encoder_ms,
        "n_inst":     len(ious),
        "iou":        ious,
        "biou":       b_ious,
    }


def run_one(backend, dataloader, num_images, label="", warmup=2,
            collect_coco=False, cat_id_to_name=None) -> Dict:
    reset_memory()

    # Warmup
    it = iter(dataloader)
    for _ in range(warmup):
        try:
            s = next(it)
        except StopIteration:
            break
        backend.warmup(s["image"][0])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    del it

    coco_preds = [] if collect_coco else None
    enc_times, all_ious, all_bious = [], [], []
    n_imgs, n_inst = 0, 0

    pbar = tqdm(total=min(len(dataloader), num_images), desc=label, leave=False)
    t_overall = time.perf_counter()
    for sample in dataloader:
        if n_imgs >= num_images:
            break
        try:
            r = eval_image(backend, sample, coco_preds, cat_id_to_name)
        except Exception as e:
            print(f"  image error: {e}")
            continue
        enc_times.append(r["encoder_ms"])
        all_ious.extend(r["iou"])
        all_bious.extend(r["biou"])
        n_imgs += 1
        n_inst += r["n_inst"]
        pbar.update(1)
        if all_ious:
            pbar.set_postfix(
                miou=f"{float(np.mean(all_ious)):.4f}",
                biou=f"{float(np.mean(all_bious)):.4f}",
                n=len(all_ious),
            )
    pbar.close()
    overall_sec = time.perf_counter() - t_overall

    mem = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem = {
            "peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_memory_reserved_mb":  torch.cuda.max_memory_reserved()  / 1024**2,
        }

    return {
        "num_images":                n_imgs,
        "num_instances":             n_inst,
        "throughput_imgs_per_sec":   n_imgs / overall_sec,
        "encoder_mean_ms":           float(np.mean(enc_times)) if enc_times else 0.0,
        "encoder_std_ms":            float(np.std(enc_times))  if enc_times else 0.0,
        "miou":                      float(np.mean(all_ious))  if all_ious  else 0.0,
        "miou_std":                  float(np.std(all_ious))   if all_ious  else 0.0,
        "boundary_iou":              float(np.mean(all_bious)) if all_bious else 0.0,
        "boundary_iou_std":          float(np.std(all_bious))  if all_bious else 0.0,
        **mem,
        "_coco_preds": coco_preds,
        "timestamp":   datetime.datetime.now().isoformat(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# COCO segm AP
# ──────────────────────────────────────────────────────────────────────────────

def compute_coco_ap(ann_file: str, predictions: list) -> Dict:
    if not predictions:
        return {}
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(predictions)
    e = COCOeval(coco_gt, coco_dt, iouType="segm")
    img_ids = sorted({p["image_id"] for p in predictions})
    e.params.imgIds = img_ids
    e.evaluate(); e.accumulate(); e.summarize()
    s = e.stats
    return {
        "AP":        float(s[0]),
        "AP50":      float(s[1]),
        "AP75":      float(s[2]),
        "AP_small":  float(s[3]),
        "AP_medium": float(s[4]),
        "AP_large":  float(s[5]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_sweep(backend, algos, ratios, dataloader, num_images, margin,
              ann_file, ap=False, cat_id_to_name=None) -> List[Dict]:
    valid = backend.valid_algos()
    for a in algos:
        if a not in valid:
            raise ValueError(f"backend={backend.name} does not support algo "
                             f"{a!r}. Valid: {valid}")

    runs = []
    for algo in algos:
        if algo == "none":
            runs.append(("none", 1.0))
        else:
            for r in ratios:
                if r < 1.0:
                    runs.append((algo, r))
    seen = set(); deduped = []
    for r in runs:
        if r not in seen:
            seen.add(r); deduped.append(r)
    runs = deduped

    print(f"\n{'='*80}\n[{backend.name}] COCO sweep: {len(runs)} configs\n{'='*80}\n")

    results = []
    for algo, ratio in runs:
        backend.remove_patch()
        backend.apply_patch(algo=algo, ratio=ratio, margin=margin)

        label = f"{algo} r={ratio:.2f}"
        print(f"\n── {label} ──")
        out = run_one(backend, dataloader, num_images, label=label,
                      collect_coco=ap, cat_id_to_name=cat_id_to_name)
        coco_preds = out.pop("_coco_preds")

        ap_metrics = {}
        if ap and coco_preds:
            ap_metrics = compute_coco_ap(ann_file, coco_preds)

        out.update({
            "algo":  algo,
            "ratio": ratio,
            **{f"coco_{k}": v for k, v in ap_metrics.items()},
        })
        results.append(out)

        log = {
            f"{algo}_r{ratio:.2f}/throughput":   out["throughput_imgs_per_sec"],
            f"{algo}_r{ratio:.2f}/encoder_ms":   out["encoder_mean_ms"],
            f"{algo}_r{ratio:.2f}/miou":         out["miou"],
            f"{algo}_r{ratio:.2f}/boundary_iou": out["boundary_iou"],
            f"{algo}_r{ratio:.2f}/peak_memory_mb": out.get("peak_memory_allocated_mb", 0),
        }
        for k, v in ap_metrics.items():
            log[f"{algo}_r{ratio:.2f}/{k}"] = v
        wandb.log(log)

        msg = (f"  throughput={out['throughput_imgs_per_sec']:.2f} img/s  "
               f"enc={out['encoder_mean_ms']:.1f}ms  "
               f"mIoU={out['miou']:.4f}  B-IoU={out['boundary_iou']:.4f}")
        if ap_metrics:
            msg += f"  AP={ap_metrics['AP']:.4f}  AP50={ap_metrics['AP50']:.4f}"
        print(msg)

    backend.remove_patch()
    return results


def print_summary(results):
    """Paper-style table: model rows × COCO segm AP columns."""
    has_ap = any("coco_AP" in r for r in results)

    def fmt_model(r):
        return r["algo"] if r["algo"] == "none" else f"{r['algo']} r={r['ratio']:.2f}"

    if has_ap:
        col_w = 22
        ap_cols = [("AP", "coco_AP"), ("AP_50", "coco_AP50"), ("AP_75", "coco_AP75"),
                   ("AP_L", "coco_AP_large"), ("AP_M", "coco_AP_medium"),
                   ("AP_S", "coco_AP_small")]
        sep = "│"
        h_left  = f"{'Model':<{col_w}}"
        h_label = " " * col_w + sep + f"{'COCO val2017 (segm)':^{8 * len(ap_cols)}}"
        h_cols  = h_left + sep + "".join(f"{n:>8}" for n, _ in ap_cols)
        line_w  = len(h_cols)
        print("\n" + "=" * line_w)
        print(h_label)
        print(h_cols)
        print("─" * line_w)
        for r in results:
            row = f"{fmt_model(r):<{col_w}}" + sep + "".join(
                f"{r.get(k, 0) * 100:>8.1f}" for _, k in ap_cols
            )
            print(row)
        print("=" * line_w)
    else:
        # mIoU / B-IoU table (GT-box protocol)
        base = next((r for r in results if r["algo"] == "none"), None)
        col_w = 22
        sep = "│"
        h_cols = (f"{'Model':<{col_w}}" + sep +
                  f"{'mIoU':>8}{'ΔmIoU':>8}{'B-IoU':>8}{'Enc(ms)':>10}"
                  f"{'Thr(im/s)':>11}{'Mem(MB)':>10}")
        line_w = len(h_cols)
        print("\n" + "=" * line_w); print(h_cols); print("─" * line_w)
        for r in results:
            d_iou = (r["miou"] - base["miou"]) if base else 0
            print(f"{fmt_model(r):<{col_w}}" + sep +
                  f"{r['miou']*100:>8.2f}{d_iou*100:>+8.2f}"
                  f"{r['boundary_iou']*100:>8.2f}"
                  f"{r['encoder_mean_ms']:>10.2f}"
                  f"{r['throughput_imgs_per_sec']:>11.2f}"
                  f"{r.get('peak_memory_allocated_mb', 0):>10.0f}")
        print("=" * line_w)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _build_backend(args) -> "SamHQBackend | Sam3Backend":
    if args.backbone == "sam-hq":
        print(f"Loading SAM-HQ {args.model_type} from {args.model_ckt} ...")
        sam = sam_model_registry[args.model_type](
            checkpoint=args.model_ckt
        ).to("cuda").half()
        return SamHQBackend(SamPredictor(sam))

    if args.backbone == "sam3":
        from transformers.models.sam3.modeling_sam3 import Sam3Model
        from transformers.models.sam3.processing_sam3 import Sam3Processor
        print(f"Loading SAM-3 from {args.sam3_model} ...")
        processor = Sam3Processor.from_pretrained(args.sam3_processor or args.sam3_model)
        model = Sam3Model.from_pretrained(args.sam3_model).to("cuda").eval()
        if args.dtype != "fp32":
            model = model.to(dtype=torch.float16 if args.dtype == "fp16"
                                                 else torch.bfloat16)
        amp_dtype = {"fp32": None, "fp16": torch.float16,
                     "bf16": torch.bfloat16}[args.dtype]
        return Sam3Backend(
            model, processor,
            text_prompt=args.text_prompt,
            amp_dtype=amp_dtype,
            use_category_text=args.use_category_text,
        )

    raise ValueError(f"unknown backbone: {args.backbone!r}")


def main():
    p = argparse.ArgumentParser(description="Evaluate SAM-HQ or SAM-3 on COCO with ToMe patches",
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--backbone", type=str, default="sam-hq",
                   choices=["sam-hq", "sam3"],
                   help="Which model to evaluate.")

    p.add_argument("--coco-root", type=str, default="./data/coco",
                   help="Root with val2017/ and annotations/instances_val2017.json")
    p.add_argument("--split", type=str, default="val2017")
    p.add_argument("--num-images", type=int, default=200)
    p.add_argument("--max-instances", type=int, default=None,
                   help="Cap instances per image (default: no cap).")

    # SAM-HQ args
    p.add_argument("--model-ckt",  type=str, default="./ckts/sam_hq_vit_l.pth")
    p.add_argument("--model-type", type=str, default="vit_l",
                   choices=["vit_h", "vit_l", "vit_b", "vit_tiny"])

    # SAM-3 args
    p.add_argument("--sam3-model", type=str, default="facebook/sam3",
                   help="HF model id (e.g. facebook/sam3).")
    p.add_argument("--sam3-processor", type=str, default=None,
                   help="HF processor id (defaults to --sam3-model).")
    p.add_argument("--text-prompt", type=str, default="object",
                   help="Generic text prompt passed to SAM-3.")
    p.add_argument("--use-category-text", action="store_true",
                   help="Use the COCO category name as the text prompt for "
                        "each instance (instead of --text-prompt).")
    p.add_argument("--dtype", type=str, default="fp16",
                   choices=["fp32", "fp16", "bf16"],
                   help="SAM-3 inference precision (autocast).")

    p.add_argument("--algos", type=str, nargs="+",
                   default=["none", "tome", "pitome", "sparsesam", "gradtome"],
                   choices=VALID_ALGOS)
    p.add_argument("--ratios", type=float, nargs="+", default=[0.9, 0.8, 0.7])
    p.add_argument("--margin", type=float, default=0.5)

    p.add_argument("--ap", action="store_true",
                   help="Compute COCO segm AP (requires accumulating RLEs in RAM).")
    p.add_argument("--detections", type=str, default=None,
                   help="Path to COCO-results JSON with detector boxes "
                        "(e.g. FocalNet-DINO predictions). When set, switches "
                        "to detector-prompt protocol and forces --ap.")
    p.add_argument("--score-threshold", type=float, default=0.0,
                   help="Drop detections below this score (only used with --detections).")

    p.add_argument("--output-dir", type=str, default="./benchmark_results")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="sam-tome-coco")
    p.add_argument("--wandb-run-name", type=str, default=None)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    wandb.init(
        project=args.wandb_project, name=args.wandb_run_name,
        config=vars(args), mode="disabled" if args.no_wandb else "online",
    )

    image_dir = os.path.join(args.coco_root, args.split)
    ann_file  = os.path.join(args.coco_root, "annotations",
                             f"instances_{args.split}.json")
    for path in (image_dir, ann_file):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                f"Expected layout: {args.coco_root}/{{val2017/, annotations/instances_val2017.json}}"
            )

    backend = _build_backend(args)

    if args.detections is not None:
        if not os.path.exists(args.detections):
            raise FileNotFoundError(f"Detections file not found: {args.detections}")
        args.ap = True  # detector-prompt protocol → AP-only
        print(f"Using detector boxes from {args.detections} (forcing --ap)")

    print(f"Loading COCO {args.split} ...")
    dataset = CocoInstanceDataset(
        image_dir, ann_file,
        max_instances_per_image=args.max_instances,
        detections_file=args.detections,
        score_threshold=args.score_threshold,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2,
        pin_memory=True, collate_fn=coco_collate_fn,
    )
    print(f"  {len(dataset)} images, eval cap = {args.num_images}")

    cat_id_to_name = None
    if args.backbone == "sam3" and args.use_category_text:
        cat_id_to_name = {c["id"]: c["name"] for c in dataset.coco.cats.values()}

    results = run_sweep(
        backend, args.algos, args.ratios, loader, args.num_images,
        margin=args.margin, ann_file=ann_file, ap=args.ap,
        cat_id_to_name=cat_id_to_name,
    )

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv = os.path.join(args.output_dir, f"coco_eval_{ts}.csv")
    pd.DataFrame(results).to_csv(csv, index=False)
    print(f"\nResults → {csv}")
    print_summary(results)
    wandb.log({"results_table": wandb.Table(dataframe=pd.DataFrame(results))})
    wandb.finish()


if __name__ == "__main__":
    main()
