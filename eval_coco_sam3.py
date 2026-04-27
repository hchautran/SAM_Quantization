#!/usr/bin/env python3
"""Evaluate SAM3 on COCO val2017 with text prompts (open-vocabulary detection).

Protocol:
  1. For each image, encode it once using the image encoder.
  2. For each category, run grounding using pre-encoded text prompts (chunked to fit GPU).
  3. Dump predictions to JSON and use the official offline COCO evaluator
     (sam3.eval.coco_eval_offline.CocoEvaluatorOfflineWithPredFileEvaluators),
     which supports positive-split scoring and TIDE error analysis.
"""

import os
import sys
import gc
import json
import time
import datetime
import argparse
import traceback
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import wandb

from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

# Add sam3 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sam3'))
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.data_misc import FindStage, interpolate
from sam3.model import box_ops
from sam3.eval.coco_eval_offline import CocoEvaluatorOfflineWithPredFileEvaluators


# ──────────────────────────────────────────────────────────────────────────────
# Per-image inference
# ──────────────────────────────────────────────────────────────────────────────

def eval_image(processor, image_id: int, image_path: str,
               cat_ids: List[int],
               batched_text_features: Dict[str, torch.Tensor],
               confidence_threshold: float, top_k_per_cat: int,
               coco_predictions: dict, iou_types: List[str]) -> float:
    """One image: set_image → chunked grounding over all categories."""
    model  = processor.model
    device = processor.device
    image  = Image.open(image_path).convert("RGB")
    img_h, img_w = image.height, image.width

    has_segm = "segm" in iou_types

    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = processor.set_image(image)
        torch.cuda.synchronize()
        encoder_ms = (time.perf_counter() - t0) * 1000

        N = len(cat_ids)
        out_bbox_list = []
        out_probs_list = []
        out_masks_list = [] if has_segm else None

        chunk_size = min(processor.chunk_size, N)
        for chunk_start in range(0, N, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N)
            B = chunk_end - chunk_start

            chunk_text_features = {
                "language_features": batched_text_features["language_features"][:, chunk_start:chunk_end],
                "language_mask": batched_text_features["language_mask"][chunk_start:chunk_end]
            }
            state["backbone_out"].update(chunk_text_features)

            geometric_prompt = model._get_dummy_prompt(num_prompts=B)

            find_stage_chunk = FindStage(
                img_ids=torch.zeros(B, device=device, dtype=torch.long),
                text_ids=torch.arange(B, device=device, dtype=torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )

            outputs = model.forward_grounding(
                backbone_out=state["backbone_out"],
                find_input=find_stage_chunk,
                find_target=None,
                geometric_prompt=geometric_prompt,
            )

            out_bbox_list.append(outputs["pred_boxes"].detach())
            if has_segm:
                out_masks_list.append(outputs["pred_masks"].detach())

            out_logits = outputs["pred_logits"]
            out_probs  = (out_logits.sigmoid() * outputs["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
            out_probs_list.append(out_probs.detach())

            del outputs, out_logits, out_probs, geometric_prompt, find_stage_chunk

        out_bbox = torch.cat(out_bbox_list, dim=0) # [N, Q, 4]
        out_probs = torch.cat(out_probs_list, dim=0) # [N, Q]
        out_masks = torch.cat(out_masks_list, dim=0) if has_segm else None

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale = torch.tensor([img_w, img_h, img_w, img_h], device=boxes_xyxy.device)
        boxes_xyxy = boxes_xyxy * scale[None, None, :]

        Q = out_probs.shape[1]
        k = min(top_k_per_cat, Q)
        top_scores, top_idx = out_probs.topk(k, dim=1)
        keep_mask = top_scores > confidence_threshold

        if not keep_mask.any():
            return encoder_ms

        n_idx = torch.arange(N, device=out_probs.device).unsqueeze(1).expand(-1, k)
        flat_keep = keep_mask.flatten()
        flat_n    = n_idx.flatten()[flat_keep]
        flat_q    = top_idx.flatten()[flat_keep]
        flat_scores = top_scores.flatten()[flat_keep]
        flat_boxes  = boxes_xyxy[flat_n, flat_q]

        sel_masks = None
        if has_segm:
            sel_masks = out_masks[flat_n, flat_q].unsqueeze(1)

            decoded_masks = []
            mchunk = 64
            for start in range(0, sel_masks.shape[0], mchunk):
                chunk = sel_masks[start:start + mchunk]
                chunk = interpolate(chunk, (img_h, img_w), mode="bilinear", align_corners=False).sigmoid()
                decoded_masks.append((chunk > 0.5).squeeze(1).cpu().numpy())
            sel_masks = np.concatenate(decoded_masks, axis=0) if decoded_masks else np.zeros((0, img_h, img_w), dtype=np.uint8)
            del decoded_masks

    boxes_np  = flat_boxes.float().cpu().numpy()
    scores_np = flat_scores.float().cpu().numpy()
    n_arr     = flat_n.cpu().numpy()

    for i in range(boxes_np.shape[0]):
        cat_id = cat_ids[int(n_arr[i])]
        x1, y1, x2, y2 = boxes_np[i].tolist()
        payload = {
            "image_id":    int(image_id),
            "category_id": int(cat_id),
            "score":       float(scores_np[i]),
        }
        coco_predictions.setdefault("bbox", []).append({
            **payload,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
        })
        if has_segm:
            m = sel_masks[i].astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(m))
            rle["counts"] = rle["counts"].decode("ascii")
            coco_predictions.setdefault("segm", []).append({
                **payload,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "segmentation": rle,
            })

    del out_bbox, out_probs, out_masks, state, boxes_xyxy
    gc.collect()
    torch.cuda.empty_cache()

    return encoder_ms


# ──────────────────────────────────────────────────────────────────────────────
# Top-level evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def run_eval(processor, image_dir, ann_file, num_images,
             top_k_per_cat: int, iou_types: List[str],
             dump_dir: str, run_tag: str,
             positive_split: bool, use_tide: bool) -> Dict:

    coco_gt = COCO(ann_file)
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_id_to_name = [(c["id"], c["name"]) for c in cats]
    cat_ids = [c[0] for c in cat_id_to_name]

    print(f"  {len(cat_id_to_name)} COCO categories")

    print("  pre-encoding text prompts ...")
    names = [name for _, name in cat_id_to_name]
    text_cache = processor.model.backbone.forward_text(
        names, device=processor.device,
    )

    img_ids = sorted(coco_gt.getImgIds())[:num_images]
    coco_predictions: Dict[str, list] = {}
    enc_times = []

    t_overall = time.perf_counter()
    pbar = tqdm(img_ids, desc="SAM3 COCO Eval")
    for img_id in pbar:
        info = coco_gt.loadImgs(img_id)[0]
        path = os.path.join(image_dir, info["file_name"])
        try:
            ms = eval_image(processor, img_id, path,
                            cat_ids, text_cache,
                            processor.confidence_threshold, top_k_per_cat,
                            coco_predictions, iou_types)
            enc_times.append(ms)
        except Exception as e:
            print(f"\n  image error {img_id}: {repr(e)}")
            traceback.print_exc()
            if "CUDA driver error" in repr(e) or "out of memory" in repr(e).lower():
                torch.cuda.empty_cache()

    overall_sec = time.perf_counter() - t_overall

    # Dump predictions to JSON files (one per iou_type) and run offline evaluator
    os.makedirs(dump_dir, exist_ok=True)
    results: Dict[str, float] = {}
    for iou_type in iou_types:
        preds = coco_predictions.get(iou_type, [])
        if not preds:
            print(f"  {iou_type}: no predictions")
            continue
        pred_path = os.path.join(dump_dir, f"sam3_coco_{iou_type}_{run_tag}.json")
        with open(pred_path, "w") as f:
            json.dump(preds, f)
        print(f"\n── Offline COCOeval ({iou_type})  preds={len(preds)}  file={pred_path} ──")

        evaluator = CocoEvaluatorOfflineWithPredFileEvaluators(
            gt_path=ann_file,
            tide=use_tide,
            iou_type=iou_type,
            positive_split=positive_split,
        )
        outs = evaluator.evaluate(pred_path)
        results.update(outs)

    return {
        "num_images":              len(img_ids),
        "throughput_imgs_per_sec": len(img_ids) / overall_sec if overall_sec > 0 else 0,
        "encoder_mean_ms":         float(np.mean(enc_times)) if enc_times else 0.0,
        "positive_split":          positive_split,
        **results,
        "timestamp": datetime.datetime.now().isoformat(),
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate SAM3 on COCO val2017")
    p.add_argument("--coco-root", default="./data/coco")
    p.add_argument("--split", default="val2017")
    p.add_argument("--num-images", type=int, default=200)
    p.add_argument("--resolution", type=int, default=1008,
                   help="Must be 1008 — SAM3 RoPE freqs_cis is precomputed at native resolution.")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--bpe-path", default=None)
    p.add_argument("--confidence-threshold", type=float, default=0.0)
    p.add_argument("--top-k-per-cat", type=int, default=300)
    p.add_argument("--chunk-size", type=int, default=8,
                   help="Number of categories to process per grounding batch.")
    p.add_argument("--iou-types", nargs="+", default=["bbox"],
                   help="bbox and/or segm")
    p.add_argument("--positive-split", action="store_true",
                   help="Only evaluate detections whose category is GT for that image.")
    p.add_argument("--no-tide", action="store_true",
                   help="Disable TIDE error analysis (enabled by default if installed).")
    p.add_argument("--output-dir", default="./benchmark_results")
    p.add_argument("--dump-dir", default="./benchmark_results/sam3_coco_dumps")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.no_wandb:
        wandb.init(project="sam3-coco", config=vars(args))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    print("Building SAM3 image model ...")
    model = build_sam3_image_model(
        bpe_path=args.bpe_path,
        checkpoint_path=args.checkpoint,
        load_from_HF=(args.checkpoint is None),
        enable_inst_interactivity=False,
    )
    processor = Sam3Processor(model, resolution=args.resolution,
                              confidence_threshold=args.confidence_threshold)
    processor.chunk_size = args.chunk_size

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Evaluating SAM3 on COCO {args.split}: {args.num_images} images, res={args.resolution}")
    result = run_eval(
        processor,
        os.path.join(args.coco_root, args.split),
        os.path.join(args.coco_root, "annotations", f"instances_{args.split}.json"),
        args.num_images, args.top_k_per_cat, args.iou_types,
        dump_dir=args.dump_dir, run_tag=ts,
        positive_split=args.positive_split,
        use_tide=not args.no_tide,
    )

    csv_path = os.path.join(args.output_dir, f"coco_eval_sam3_{ts}.csv")
    pd.DataFrame([result]).to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    if not args.no_wandb:
        wandb.log(result)
        wandb.finish()


if __name__ == "__main__":
    main()
