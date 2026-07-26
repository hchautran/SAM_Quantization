#!/usr/bin/env python3
"""Evaluate MobileSAM and FastSAM on COCO using detector boxes."""

import argparse
import csv
import datetime
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mmcv
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.utils import compat_cfg, get_device, replace_cfg_vals, setup_multi_processes, update_data_root


ROOT = Path(__file__).resolve().parent
MOBILE_SAM_ROOT = ROOT / "MobileSAM"
FASTSAM_ROOT = ROOT / "FastSAM"
PTQ4SAM_ROOT = ROOT / "PTQ4SAM"

for path in (str(ROOT), str(PTQ4SAM_ROOT), str(MOBILE_SAM_ROOT), str(FASTSAM_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from mobile_sam import SamPredictor, sam_model_registry  # noqa: E402
from fastsam import FastSAM  # noqa: E402


DETECTOR_CONFIGS = {
    "yolox": "quant/configmmdet/yolox/yolo_l-sam-vit-l.py",
    "dino": "quant/configmmdet/focalnet_dino/focalnet-l-dino_sam-vit-l.py",
    "hdetr": "quant/configmmdet/hdetr/r50-hdetr_sam-vit-l.py",
}


def patch_torch_load_for_fastsam_checkpoint():
    original_torch_load = torch.load

    def torch_load_with_legacy_checkpoint_support(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = torch_load_with_legacy_checkpoint_support


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def import_mmdet_plugin(cfg):
    if not getattr(cfg, "plugin", False):
        return
    if hasattr(cfg, "plugin_dir"):
        plugin_dir = os.path.abspath(cfg.plugin_dir)
        parent_dir = os.path.dirname(os.path.dirname(plugin_dir))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        module_path = os.path.relpath(plugin_dir, parent_dir).replace("/", ".").strip(".")
    else:
        module_path = os.path.dirname(cfg.filename).replace("/", ".").strip(".")
    importlib.import_module(module_path)


def build_detector_and_loader(detector_name: str, batch_size: int, workers: int):
    cfg = Config.fromfile(DETECTOR_CONFIGS[detector_name])
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    import_mmdet_plugin(cfg)
    import quant.configmmdet.det_observer_instance_sam_  # noqa: F401
    cfg.model.train_cfg = None
    cfg.device = get_device()
    cfg.gpu_ids = [0]

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if batch_size > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    test_loader_cfg = {
        "samples_per_gpu": batch_size,
        "workers_per_gpu": workers,
        "dist": False,
        "shuffle": False,
        **cfg.data.get("test_dataloader", {}),
    }
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    model.det_model.to(cfg.device)
    model.to(cfg.device)
    model.eval()
    return model, dataset, data_loader


def unwrap_data_container(value):
    if hasattr(value, "data"):
        value = value.data
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return unwrap_data_container(value[0])
    return value


def normalize_batch_imgs(value):
    value = unwrap_data_container(value)
    if isinstance(value, torch.Tensor):
        if value.dim() == 3:
            return value.unsqueeze(0)
        return value
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
        if len(value) == 1 and value[0].dim() == 4:
            return value[0]
        return torch.stack([img if img.dim() == 3 else img.squeeze(0) for img in value], dim=0)
    raise TypeError(f"Unsupported img batch type: {type(value)}")


def normalize_batch_metas(value):
    value = unwrap_data_container(value)
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)) and (not value or isinstance(value[0], dict)):
        return list(value)
    raise TypeError(f"Unsupported img_metas batch type: {type(value)}")


def iter_batch_items(data: Dict):
    imgs = normalize_batch_imgs(data["img"])
    metas = normalize_batch_metas(data["img_metas"])
    if imgs.size(0) != len(metas):
        raise ValueError(f"Image/meta batch mismatch: {imgs.size(0)} images, {len(metas)} metas")
    for index, meta in enumerate(metas):
        meta = dict(meta)
        meta.setdefault("batch_input_shape", tuple(int(v) for v in imgs.shape[-2:]))
        yield imgs[index:index + 1], [meta]


@torch.no_grad()
def get_detector_predictions(detector_model, data: Dict, max_boxes: int, score_thr: float):
    predictions = []
    for img, img_metas in iter_batch_items(data):
        img = img.cuda(non_blocking=True) if torch.cuda.is_available() else img
        results = detector_model.simple_test(
            img=img,
            img_metas=img_metas,
            rescale=True,
            get_det_results=True,
        )
        result = results[0]
        boxes = result["boxes"]
        labels = result["labels"]
        scores = result["scores"]
        keep = scores >= score_thr
        boxes = boxes[keep][:max_boxes].detach().cpu().numpy()
        labels = labels[keep][:max_boxes].detach().cpu().numpy().astype(int)
        scores = scores[keep][:max_boxes].detach().cpu().numpy()
        predictions.append((img_metas[0], boxes, labels, scores))
    return predictions


def encode_binary_mask(mask: np.ndarray) -> Dict:
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    return {
        "size": [int(v) for v in rle["size"]],
        "counts": rle["counts"].decode("ascii"),
    }


@torch.no_grad()
def mobilesam_segment_image(predictor: SamPredictor, image: np.ndarray, boxes: np.ndarray):
    if len(boxes) == 0:
        return [], {"encoder_ms": 0.0, "decode_ms": 0.0, "total_ms": 0.0}

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_start = time.perf_counter()

    input_image = predictor.transform.apply_image(image)
    input_tensor = torch.as_tensor(input_image, device=predictor.device)
    input_tensor = input_tensor.permute(2, 0, 1).contiguous()[None, :, :, :]
    predictor.reset_image()
    predictor.original_size = image.shape[:2]
    predictor.input_size = tuple(input_tensor.shape[-2:])
    preprocessed = predictor.model.preprocess(input_tensor)

    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start = time.perf_counter()
    predictor.features = predictor.model.image_encoder(preprocessed)
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        encoder_ms = start_event.elapsed_time(end_event)
    else:
        encoder_ms = (time.perf_counter() - start) * 1000
    predictor.is_image_set = True

    transformed_boxes = predictor.transform.apply_boxes_torch(
        torch.as_tensor(boxes, dtype=torch.float32, device=predictor.device),
        image.shape[:2],
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_start = time.perf_counter()
    masks, sam_scores, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - decode_start) * 1000

    predictor.reset_image()
    total_ms = (time.perf_counter() - total_start) * 1000
    return masks[:, 0].detach().cpu().numpy(), {
        "encoder_ms": encoder_ms,
        "decode_ms": decode_ms,
        "total_ms": total_ms,
        "sam_scores": sam_scores[:, 0].detach().cpu().numpy(),
    }


def create_fastsam_layer_profiler(model, backbone_end_layer: int, encoder_end_layer: int):
    layers = model.model.model
    max_idx = len(layers) - 1
    if not (0 <= backbone_end_layer <= max_idx):
        raise ValueError(f"backbone_end_layer valid range: 0..{max_idx}")
    if not (0 <= encoder_end_layer <= max_idx):
        raise ValueError(f"encoder_end_layer valid range: 0..{max_idx}")
    if backbone_end_layer > encoder_end_layer:
        raise ValueError("backbone_end_layer must be <= encoder_end_layer")

    state = {}
    handles = []

    def start_hook(_module, _inputs):
        if torch.cuda.is_available():
            state["start_event"] = torch.cuda.Event(enable_timing=True)
            state["start_event"].record()
        else:
            state["start_time"] = time.perf_counter()

    def make_end_hook(name):
        def end_hook(_module, _inputs, _output):
            if torch.cuda.is_available():
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                state[f"{name}_event"] = event
            else:
                state[f"{name}_ms"] = (time.perf_counter() - state["start_time"]) * 1000
        return end_hook

    handles.append(layers[0].register_forward_pre_hook(start_hook))
    handles.append(layers[backbone_end_layer].register_forward_hook(make_end_hook("backbone")))
    handles.append(layers[encoder_end_layer].register_forward_hook(make_end_hook("encoder")))

    def finish():
        for handle in handles:
            handle.remove()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = state.get("start_event")
            if start_event is not None:
                if "backbone_event" in state:
                    state["backbone_ms"] = start_event.elapsed_time(state["backbone_event"])
                if "encoder_event" in state:
                    state["encoder_ms"] = start_event.elapsed_time(state["encoder_event"])
        return {
            "backbone_ms": float(state.get("backbone_ms", 0.0)),
            "encoder_ms": float(state.get("encoder_ms", 0.0)),
        }

    return finish


def select_fastsam_mask_for_box(result, box_xyxy, image_shape):
    if result is None or result.masks is None or len(result.masks.data) == 0:
        return None
    masks = result.masks.data
    h, w = masks.shape[1:]
    target_h, target_w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    if h != target_h or w != target_w:
        x1 = x1 * w / target_w
        x2 = x2 * w / target_w
        y1 = y1 * h / target_h
        y2 = y2 * h / target_h
    x1 = max(0, min(w, round(x1)))
    y1 = max(0, min(h, round(y1)))
    x2 = max(0, min(w, round(x2)))
    y2 = max(0, min(h, round(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    masks_area = torch.sum(masks[:, y1:y2, x1:x2], dim=(1, 2))
    orig_masks_area = torch.sum(masks, dim=(1, 2))
    ious = masks_area / (bbox_area + orig_masks_area - masks_area).clamp_min(1)
    return masks[int(torch.argmax(ious))].detach().cpu().numpy()


@torch.no_grad()
def fastsam_segment_image(model: FastSAM, image: Image.Image, boxes: np.ndarray, args):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    finish_profile = create_fastsam_layer_profiler(
        model,
        backbone_end_layer=args.fastsam_backbone_end_layer,
        encoder_end_layer=args.fastsam_encoder_end_layer,
    )
    start = time.perf_counter()
    results = model(
        image,
        device=args.device,
        retina_masks=args.retina,
        imgsz=args.imgsz,
        conf=args.fastsam_conf,
        iou=args.fastsam_iou,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - start) * 1000
    profile = finish_profile()
    result = results[0] if results else None
    image_np = np.array(image)
    masks = [select_fastsam_mask_for_box(result, box, image_np.shape) for box in boxes]
    return masks, {
        "backbone_ms": profile["backbone_ms"],
        "encoder_ms": profile["encoder_ms"],
        "total_ms": total_ms,
    }


def class_label_to_cat_id(dataset, label: int) -> int:
    return int(dataset.cat_ids[int(label)])


def build_detections(image_id, dataset, boxes, labels, det_scores, masks, sam_scores=None, score_mode="det"):
    detections = []
    for i, mask in enumerate(masks):
        if mask is None:
            continue
        score = float(det_scores[i])
        if score_mode == "det_sam" and sam_scores is not None:
            score *= float(sam_scores[i])
        detections.append(
            {
                "image_id": int(image_id),
                "category_id": class_label_to_cat_id(dataset, int(labels[i])),
                "segmentation": encode_binary_mask(mask),
                "score": score,
            }
        )
    return detections


def summarize_coco(coco_gt: COCO, detections: List[Dict], image_ids: List[int]):
    if not detections:
        return {}
    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval(coco_gt, coco_dt, "segm")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = [
        "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
    ]
    return {name: float(value) for name, value in zip(names, evaluator.stats)}


def append_summary_csv(summary_rows: List[Dict], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in summary_rows:
        fieldnames = list(dict.fromkeys(fieldnames + list(row.keys())))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def run_detector(detector_name: str, args, mobile_predictor, fastsam_model):
    reset_memory()
    detector_model, dataset, data_loader = build_detector_and_loader(
        detector_name, args.batch_size, args.workers
    )
    coco_gt = COCO(args.ann_file)

    mobile_dets = []
    fast_dets = []
    image_ids = []
    rows = []
    start = time.time()
    total_images = 0

    for data in tqdm(data_loader, desc=f"{detector_name} detector"):
        if total_images >= args.num_samples:
            break
        detector_predictions = get_detector_predictions(
            detector_model, data, args.max_boxes, args.det_score_thr
        )
        for meta, boxes, labels, scores in detector_predictions:
            if total_images >= args.num_samples:
                break
            image_id = int(meta["ori_filename"].split(".")[0])
            image_path = Path(meta["filename"])
            image_np = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_np)

            if "mobilesam" in args.models:
                masks, metrics = mobilesam_segment_image(mobile_predictor, image_np, boxes)
                mobile_dets.extend(
                    build_detections(
                        image_id, dataset, boxes, labels, scores, masks,
                        metrics.get("sam_scores"), score_mode=args.score_mode,
                    )
                )
                rows.append({
                    "detector": detector_name, "model": "MobileSAM", "image_id": image_id,
                    "num_boxes": len(boxes), "encoder_ms": metrics["encoder_ms"],
                    "decode_ms": metrics["decode_ms"], "total_ms": metrics["total_ms"],
                })

            if "fastsam" in args.models:
                masks, metrics = fastsam_segment_image(fastsam_model, image_pil, boxes, args)
                fast_dets.extend(build_detections(image_id, dataset, boxes, labels, scores, masks))
                rows.append({
                    "detector": detector_name, "model": "FastSAM", "image_id": image_id,
                    "num_boxes": len(boxes), "backbone_ms": metrics["backbone_ms"],
                    "encoder_ms": metrics["encoder_ms"], "total_ms": metrics["total_ms"],
                })

            image_ids.append(image_id)
            total_images += 1

    total_sec = time.time() - start
    summary_rows = []
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)

    for model_name, detections in (("MobileSAM", mobile_dets), ("FastSAM", fast_dets)):
        if model_name.lower() not in args.models:
            continue
        metrics = summarize_coco(coco_gt, detections, image_ids)
        model_rows = [r for r in rows if r["model"] == model_name]
        summary = {
            **metrics,
            "detector": detector_name,
            "model": model_name,
            "num_images": len(image_ids),
            "num_predictions": len(detections),
            "total_time_sec": total_sec,
            "throughput_imgs_per_sec": len(image_ids) / total_sec if total_sec else 0.0,
            "mean_total_ms": float(np.mean([r["total_ms"] for r in model_rows])) if model_rows else 0.0,
            "mean_encoder_ms": float(np.mean([r.get("encoder_ms", 0.0) for r in model_rows])) if model_rows else 0.0,
            "backbone_mean_ms": float(np.mean([r.get("backbone_ms", 0.0) for r in model_rows])) if model_rows else 0.0,
        }
        summary_rows.append(summary)
        if args.save_detections:
            save_json(
                output_dir / f"{model_name.lower()}_{detector_name}_coco_detections_{stamp}.json",
                detections,
            )

    del detector_model, data_loader, dataset
    reset_memory()
    return summary_rows, rows


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FastSAM/MobileSAM on COCO with detector boxes.")
    parser.add_argument("--detectors", nargs="+", default=["yolox", "dino", "hdetr"], choices=list(DETECTOR_CONFIGS))
    parser.add_argument("--models", nargs="+", default=["mobilesam", "fastsam"], choices=["mobilesam", "fastsam"])
    parser.add_argument("--ann-file", default=str(ROOT / "data/coco/annotations/instances_val2017.json"))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-boxes", type=int, default=120)
    parser.add_argument("--det-score-thr", type=float, default=0.05)
    parser.add_argument("--score-mode", choices=["det", "det_sam"], default="det")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mobilesam-checkpoint", default=str(MOBILE_SAM_ROOT / "weights/mobile_sam.pt"))
    parser.add_argument("--mobilesam-model-type", default="vit_t")
    parser.add_argument("--fastsam-checkpoint", default=str(FASTSAM_ROOT / ".weights/FastSAM-x.pt"))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--fastsam-conf", type=float, default=0.4)
    parser.add_argument("--fastsam-iou", type=float, default=0.9)
    parser.add_argument("--fastsam-backbone-end-layer", type=int, default=9)
    parser.add_argument("--fastsam-encoder-end-layer", type=int, default=21)
    parser.add_argument("--retina", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=str(ROOT / "benchmark_results"))
    parser.add_argument("--summary-csv", default=str(ROOT / "benchmark_results/coco_detector_sam_summary.csv"))
    parser.add_argument("--per-image-csv", default=str(ROOT / "benchmark_results/coco_detector_sam_per_image.csv"))
    parser.add_argument("--save-detections", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault(
        "PYTHONPATH",
        f"{ROOT / 'PTQ4SAM/projects/instance_segment_anything/ops'}:{os.environ.get('PYTHONPATH', '')}",
    )

    mobile_predictor = None
    if "mobilesam" in args.models:
        print("Loading MobileSAM...")
        sam = sam_model_registry[args.mobilesam_model_type](checkpoint=args.mobilesam_checkpoint).to(args.device)
        sam.eval()
        mobile_predictor = SamPredictor(sam)

    fastsam_model = None
    if "fastsam" in args.models:
        print("Loading FastSAM...")
        patch_torch_load_for_fastsam_checkpoint()
        fastsam_model = FastSAM(args.fastsam_checkpoint)

    all_summaries = []
    all_rows = []
    for detector in args.detectors:
        summaries, rows = run_detector(detector, args, mobile_predictor, fastsam_model)
        all_summaries.extend(summaries)
        all_rows.extend(rows)
        append_summary_csv(all_summaries, Path(args.summary_csv))
        append_summary_csv(all_rows, Path(args.per_image_csv))

    print("Summary CSV:", args.summary_csv)
    print("Per-image CSV:", args.per_image_csv)
    print(all_summaries)


if __name__ == "__main__":
    main()
