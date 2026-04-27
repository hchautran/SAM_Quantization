#!/usr/bin/env python3
"""Evaluate SAM-HQ on COCO val2017 with token-merging patches.

Protocol: GT-box prompt (per-instance) → predict mask → mIoU / Boundary IoU.
Optionally accumulates predictions and computes COCO segm AP via pycocotools.

Encoder runs once per image; decoder loops over all instance boxes for that
image (matches the cost structure of the standard SAM eval).

Usage:
    python eval_coco.py \
        --coco-root ./data/coco \
        --algos none tome pitome sparsesam gradtome \
        --ratios 0.9 0.8 0.7 \
        --num-images 200 \
        --model-ckt ./ckts/sam_hq_vit_l.pth \
        --model-type vit_l
"""

import os
import sys
import gc
import time
import json
import argparse
import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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


# ──────────────────────────────────────────────────────────────────────────────
# Per-image evaluation
# ──────────────────────────────────────────────────────────────────────────────

def eval_image(predictor: SamPredictor, sample, device,
               coco_predictions: list | None = None) -> Dict:
    """One encoder pass + per-instance decode. Returns timing / IoU lists."""
    image       = sample["image"][0].to(device)               # (3, 1024, 1024)
    ori_im      = sample["ori_im"][0]
    H, W        = sample["ori_size"][0]
    boxes_ori   = sample["boxes_ori"][0].to(device)            # (N, 4) xyxy
    masks_ori   = sample["masks_ori"][0].to(device)            # (N, H, W) uint8
    image_id    = sample["image_id"][0]
    ann_ids     = sample["ann_ids"][0]
    cat_ids     = sample["cat_ids"][0]
    det_scores  = sample.get("scores", [[1.0] * boxes_ori.shape[0]])[0]
    box_source  = sample.get("box_source", ["gt"])[0]
    n_inst      = boxes_ori.shape[0]

    # ── Encoder ───────────────────────────────────────────────────────────────
    transformed = predictor.model.preprocess(image.unsqueeze(0)).half()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        features, interm_features = predictor.model.image_encoder(transformed)
    torch.cuda.synchronize()
    encoder_ms = (time.perf_counter() - t0) * 1000

    # Set predictor state for this image
    predictor.features         = features
    predictor.interm_features  = interm_features
    predictor.original_size    = (int(H), int(W))
    predictor.input_size       = tuple(transformed.shape[-2:])
    predictor.is_image_set     = True

    # Scale boxes from original → 1024 input coordinates
    sx = 1024.0 / float(W)
    sy = 1024.0 / float(H)
    boxes_scaled = boxes_ori.clone()
    boxes_scaled[:, 0::2] *= sx
    boxes_scaled[:, 1::2] *= sy
    boxes_scaled = boxes_scaled.to(torch.float16)

    ious, b_ious = [], []
    with torch.no_grad():
        try:
            masks, scores, _ = predictor.predict_torch(
                point_coords=None, point_labels=None,
                boxes=boxes_scaled, multimask_output=False, hq_token_only=True,
            )  # (N, 1, H, W) bool
        except Exception as e:
            print(f"  decode error img {image_id}: {e}")
            return {"encoder_ms": encoder_ms, "n_inst": 0, "iou": [], "biou": []}

    for i in range(n_inst):
        pr = masks[i:i+1]                                       # (1,1,H,W)

        # Per-instance IoU only meaningful with GT-box prompts (1:1 match).
        if box_source == "gt" and ann_ids[i] != -1:
            gt = masks_ori[i:i+1].unsqueeze(0)
            ious.append(compute_iou(pr, gt).item())
            b_ious.append(compute_boundary_iou(pr, gt).item())

        if coco_predictions is not None and cat_ids[i] != -1:
            from pycocotools import mask as mask_utils
            m_np = pr.squeeze().cpu().numpy().astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(m_np))
            rle["counts"] = rle["counts"].decode("ascii")
            # Use detector score when available; otherwise SAM's mask score.
            score = float(det_scores[i]) if box_source == "detections" \
                    else float(scores[i].item())
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


def run_one(predictor, dataloader, num_images, label="", warmup=2,
            collect_coco=False) -> Dict:
    reset_memory()
    device = predictor.device

    # Warmup
    it = iter(dataloader)
    for _ in range(warmup):
        try:
            s = next(it)
        except StopIteration:
            break
        with torch.no_grad():
            t = predictor.model.preprocess(s["image"].to(device)).half()
            predictor.model.image_encoder(t)
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
            r = eval_image(predictor, sample, device, coco_preds)
        except Exception as e:
            print(f"  image error: {e}")
            continue
        enc_times.append(r["encoder_ms"])
        all_ious.extend(r["iou"])
        all_bious.extend(r["biou"])
        n_imgs += 1
        n_inst += r["n_inst"]
        pbar.update(1)
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

def run_sweep(predictor, algos, ratios, dataloader, num_images, margin,
              ann_file, ap=False) -> List[Dict]:
    encoder = predictor.model.image_encoder
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

    print(f"\n{'='*80}\nCOCO sweep: {len(runs)} configs\n{'='*80}\n")

    results = []
    for algo, ratio in runs:
        remove_tome(encoder)
        if algo != "none":
            apply_tome(encoder, algo=algo, ratio=ratio, margin=margin)

        label = f"{algo} r={ratio:.2f}"
        print(f"\n── {label} ──")
        out = run_one(predictor, dataloader, num_images, label=label, collect_coco=ap)
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

    remove_tome(encoder)
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

def main():
    p = argparse.ArgumentParser(description="Evaluate SAM-HQ on COCO with ToMe patches",
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--coco-root", type=str, default="./data/coco",
                   help="Root with val2017/ and annotations/instances_val2017.json")
    p.add_argument("--split", type=str, default="val2017")
    p.add_argument("--num-images", type=int, default=200)
    p.add_argument("--max-instances", type=int, default=None,
                   help="Cap instances per image (default: no cap).")

    p.add_argument("--model-ckt",  type=str, default="./ckts/sam_hq_vit_l.pth")
    p.add_argument("--model-type", type=str, default="vit_l",
                   choices=["vit_h", "vit_l", "vit_b", "vit_tiny"])

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

    print(f"Loading SAM {args.model_type} from {args.model_ckt} ...")
    sam = sam_model_registry[args.model_type](checkpoint=args.model_ckt).to("cuda").half()
    predictor = SamPredictor(sam)

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

    results = run_sweep(
        predictor, args.algos, args.ratios, loader, args.num_images,
        margin=args.margin, ann_file=ann_file, ap=args.ap,
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
