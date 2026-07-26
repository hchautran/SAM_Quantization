#!/usr/bin/env python3
"""Evaluate EfficientSAM and EfficientViT-SAM on COCO using detector boxes."""

import argparse
import csv
import datetime
import gc
import importlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import mmcv  # noqa: F401
import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.utils import compat_cfg, get_device, replace_cfg_vals, setup_multi_processes, update_data_root

ROOT = Path(__file__).resolve().parent
PTQ4SAM_ROOT = ROOT / "PTQ4SAM"
EFFICIENTSAM_ROOT = ROOT / "EfficientSAM"
EFFICIENTVIT_ROOT = ROOT / "efficientvit"

for path in (ROOT, PTQ4SAM_ROOT, EFFICIENTSAM_ROOT, EFFICIENTVIT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from efficient_sam.efficient_sam import build_efficient_sam  # noqa: E402


def install_efficientvit_optional_import_stubs():
    import importlib.machinery
    import importlib.util
    import types

    def missing_module(name: str) -> bool:
        return name not in sys.modules and importlib.util.find_spec(name) is None

    if missing_module("onnx"):
        onnx_stub = types.ModuleType("onnx")
        onnx_stub.__spec__ = importlib.machinery.ModuleSpec("onnx", loader=None)
        onnx_stub.load_model = lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("onnx is required only for export_onnx"))
        onnx_stub.save = lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("onnx is required only for export_onnx"))
        sys.modules["onnx"] = onnx_stub
    if missing_module("onnxsim"):
        onnxsim_stub = types.ModuleType("onnxsim")
        onnxsim_stub.__spec__ = importlib.machinery.ModuleSpec("onnxsim", loader=None)
        onnxsim_stub.simplify = lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("onnxsim is required only for export_onnx"))
        sys.modules["onnxsim"] = onnxsim_stub

DETECTOR_CONFIGS = {
    "yolox": "quant/configmmdet/yolox/yolo_l-sam-vit-l.py",
    "dino": "quant/configmmdet/focalnet_dino/focalnet-l-dino_sam-vit-l.py",
    "hdetr": "quant/configmmdet/hdetr/r50-hdetr_sam-vit-l.py",
}


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def cuda_timed_call(fn):
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        return out, float(start.elapsed_time(end))
    start_time = time.perf_counter()
    out = fn()
    return out, float((time.perf_counter() - start_time) * 1000)


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
        boxes = boxes[keep][:max_boxes].detach().cpu().numpy().astype(np.float32)
        labels = labels[keep][:max_boxes].detach().cpu().numpy().astype(int)
        scores = scores[keep][:max_boxes].detach().cpu().numpy().astype(np.float32)
        predictions.append((img_metas[0], boxes, labels, scores))
    return predictions


def encode_binary_mask(mask: np.ndarray) -> Dict:
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(mask)
    return {"size": [int(v) for v in rle["size"]], "counts": rle["counts"].decode("ascii")}


def boxes_to_efficientsam_prompts(boxes: np.ndarray, device: torch.device):
    boxes_t = torch.as_tensor(boxes, dtype=torch.float32, device=device)
    points = torch.stack([boxes_t[:, 0:2], boxes_t[:, 2:4]], dim=1).unsqueeze(0)
    labels = torch.tensor([2, 3], dtype=torch.int64, device=device).view(1, 1, 2).expand(1, boxes_t.shape[0], 2)
    return points, labels


def select_best_multimask(masks: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    best = scores[0].argmax(dim=1)
    return masks[0, torch.arange(masks.shape[1], device=masks.device), best]


@torch.no_grad()
def efficientsam_segment_image(model, image_np: np.ndarray, boxes: np.ndarray, device: torch.device):
    if len(boxes) == 0:
        return [], {"encoder_ms": 0.0, "decode_ms": 0.0, "total_ms": 0.0, "sam_scores": np.array([])}
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    points, labels = boxes_to_efficientsam_prompts(boxes, device)
    _, _, input_h, input_w = image_tensor.shape

    total_start = time.perf_counter()
    embeddings, encoder_ms = cuda_timed_call(lambda: model.get_image_embeddings(image_tensor))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_start = time.perf_counter()
    masks, scores = model.predict_masks(
        embeddings,
        points,
        labels,
        multimask_output=True,
        input_h=input_h,
        input_w=input_w,
        output_h=input_h,
        output_w=input_w,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - decode_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000
    selected = (select_best_multimask(masks, scores) > model.mask_threshold).detach().cpu().numpy()
    sam_scores = scores[0].max(dim=1).values.detach().cpu().numpy()
    return selected, {"encoder_ms": encoder_ms, "decode_ms": decode_ms, "total_ms": total_ms, "sam_scores": sam_scores}


@torch.no_grad()
def efficientvit_segment_image(predictor, image_np: np.ndarray, boxes: np.ndarray, device: torch.device):
    install_efficientvit_optional_import_stubs()
    from efficientvit.models.efficientvit.sam import SamResize
    if len(boxes) == 0:
        return [], {"encoder_ms": 0.0, "decode_ms": 0.0, "total_ms": 0.0, "sam_scores": np.array([])}
    predictor.reset_image()
    predictor.original_size = image_np.shape[:2]
    predictor.input_size = SamResize.get_preprocess_shape(
        *predictor.original_size,
        long_side_length=predictor.model.image_size[0],
    )
    torch_image = predictor.model.transform(image_np).unsqueeze(0).to(device)

    total_start = time.perf_counter()
    features, encoder_ms = cuda_timed_call(lambda: predictor.model.image_encoder(torch_image))
    predictor.features = features
    predictor.is_image_set = True

    box_torch = torch.as_tensor(predictor.apply_boxes(boxes), dtype=torch.float32, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_start = time.perf_counter()
    masks, scores, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=box_torch,
        multimask_output=True,
        return_logits=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - decode_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000

    selected = masks[torch.arange(masks.shape[0], device=device), scores.argmax(dim=1)].detach().cpu().numpy()
    sam_scores = scores.max(dim=1).values.detach().cpu().numpy()
    predictor.reset_image()
    return selected, {"encoder_ms": encoder_ms, "decode_ms": decode_ms, "total_ms": total_ms, "sam_scores": sam_scores}


def class_label_to_cat_id(dataset, label: int) -> int:
    return int(dataset.cat_ids[int(label)])


def build_detections(image_id, dataset, labels, det_scores, masks, sam_scores=None, score_mode="det"):
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


def append_csv(rows: List[Dict], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        fieldnames = list(dict.fromkeys(fieldnames + list(row.keys())))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))




def patch_efficientvit_mask_decoder_for_samhq(model):
    import inspect
    import types
    forward = model.mask_decoder.forward
    params = inspect.signature(forward).parameters
    if "hq_token_only" not in params:
        return model

    def forward_compat(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output):
        return forward(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=multimask_output,
            hq_token_only=False,
            interm_embeddings=None,
        )

    model.mask_decoder.forward = types.MethodType(forward_compat, model.mask_decoder)
    return model


def load_efficientsam(variant: str, checkpoint: Optional[str], device: str):
    if checkpoint is None:
        checkpoint_path = EFFICIENTSAM_ROOT / "weights" / f"efficient_sam_{variant}.pt"
        if variant == "vits" and not checkpoint_path.exists():
            zip_path = checkpoint_path.with_suffix(".pt.zip")
            if zip_path.exists():
                with zipfile.ZipFile(zip_path, "r") as archive:
                    archive.extract(checkpoint_path.name, checkpoint_path.parent)
        checkpoint = str(checkpoint_path)
    dims = {"vitt": (192, 3), "vits": (384, 6)}
    if variant not in dims:
        raise ValueError(f"Unsupported EfficientSAM variant: {variant}")
    return build_efficient_sam(*dims[variant], checkpoint=checkpoint).to(device).eval(), checkpoint


def load_efficientvit(model_name: str, weight_url: Optional[str], device: str, pretrained: bool = True):
    if pretrained and weight_url is None:
        checkpoint_name = model_name.replace("efficientvit-sam-", "efficientvit_sam_") + ".pt"
        weight_url = str(EFFICIENTVIT_ROOT / "assets/checkpoints/efficientvit_sam" / checkpoint_name)
    install_efficientvit_optional_import_stubs()
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model
    model = create_efficientvit_sam_model(model_name, pretrained=pretrained, weight_url=weight_url).to(device).eval()
    return patch_efficientvit_mask_decoder_for_samhq(model)


def run_detector(detector_name: str, args, loaded_models: Dict[str, object]):
    reset_memory()
    detector_model, dataset, data_loader = build_detector_and_loader(detector_name, args.batch_size, args.workers)
    coco_gt = COCO(args.ann_file)

    detections_by_model = {name: [] for name in loaded_models}
    image_ids = []
    rows = []
    start = time.time()
    total_images = 0
    device = torch.device(args.device)

    for data in tqdm(data_loader, desc=f"{detector_name} detector"):
        if total_images >= args.num_samples:
            break
        detector_predictions = get_detector_predictions(detector_model, data, args.max_boxes, args.det_score_thr)
        for meta, boxes, labels, scores in detector_predictions:
            if total_images >= args.num_samples:
                break
            image_id = int(meta["ori_filename"].split(".")[0])
            image_path = Path(meta["filename"])
            image_np = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)

            for model_name, model_obj in loaded_models.items():
                if model_name == "EfficientSAM":
                    masks, metrics = efficientsam_segment_image(model_obj, image_np, boxes, device)
                else:
                    masks, metrics = efficientvit_segment_image(model_obj, image_np, boxes, device)
                detections_by_model[model_name].extend(
                    build_detections(
                        image_id,
                        dataset,
                        labels,
                        scores,
                        masks,
                        metrics.get("sam_scores"),
                        score_mode=args.score_mode,
                    )
                )
                rows.append(
                    {
                        "detector": detector_name,
                        "model": model_name,
                        "image_id": image_id,
                        "num_boxes": int(len(boxes)),
                        "encoder_ms": float(metrics["encoder_ms"]),
                        "decode_ms": float(metrics["decode_ms"]),
                        "total_ms": float(metrics["total_ms"]),
                    }
                )

            image_ids.append(image_id)
            total_images += 1

    total_sec = time.time() - start
    memory_stats = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_stats = {
            "peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_memory_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        }

    summary_rows = []
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    for model_name, detections in detections_by_model.items():
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
            "mean_encoder_ms": float(np.mean([r["encoder_ms"] for r in model_rows])) if model_rows else 0.0,
            "mean_decode_ms": float(np.mean([r["decode_ms"] for r in model_rows])) if model_rows else 0.0,
            **memory_stats,
        }
        summary_rows.append(summary)
        if args.save_detections:
            save_json(output_dir / f"{model_name.lower().replace('-', '_')}_{detector_name}_coco_detections_{stamp}.json", detections)

    del detector_model, data_loader, dataset
    reset_memory()
    return summary_rows, rows


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EfficientSAM/EfficientViT-SAM on COCO with detector boxes.")
    parser.add_argument("--detectors", nargs="+", default=["yolox", "dino", "hdetr"], choices=list(DETECTOR_CONFIGS))
    parser.add_argument("--models", nargs="+", default=["efficientsam", "efficientvit"], choices=["efficientsam", "efficientvit"])
    parser.add_argument("--ann-file", default=str(ROOT / "data/coco/annotations/instances_val2017.json"))
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-boxes", type=int, default=120)
    parser.add_argument("--det-score-thr", type=float, default=0.05)
    parser.add_argument("--score-mode", choices=["det", "det_sam"], default="det")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--efficientsam-variant", default="vitt", choices=["vitt", "vits"])
    parser.add_argument("--efficientsam-checkpoint", default=None)
    parser.add_argument("--efficientvit-model", default="efficientvit-sam-l0")
    parser.add_argument("--efficientvit-weight-url", default=None)
    parser.add_argument("--efficientvit-no-pretrained", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "benchmark_results"))
    parser.add_argument("--summary-csv", default=str(ROOT / "benchmark_results/coco_detector_efficient_sams_summary.csv"))
    parser.add_argument("--per-image-csv", default=str(ROOT / "benchmark_results/coco_detector_efficient_sams_per_image.csv"))
    parser.add_argument("--save-detections", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault(
        "PYTHONPATH",
        f"{ROOT / 'PTQ4SAM/projects/instance_segment_anything/ops'}:{os.environ.get('PYTHONPATH', '')}",
    )

    loaded_models = {}
    if "efficientsam" in args.models:
        print("Loading EfficientSAM...")
        model, checkpoint = load_efficientsam(args.efficientsam_variant, args.efficientsam_checkpoint, args.device)
        args.efficientsam_checkpoint = checkpoint
        loaded_models["EfficientSAM"] = model
    if "efficientvit" in args.models:
        print("Loading EfficientViT-SAM...")
        model = load_efficientvit(args.efficientvit_model, args.efficientvit_weight_url, args.device, not args.efficientvit_no_pretrained)
        install_efficientvit_optional_import_stubs()
        from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
        loaded_models["EfficientViT-SAM"] = EfficientViTSamPredictor(model)

    all_summaries = []
    all_rows = []
    for detector in args.detectors:
        summaries, rows = run_detector(detector, args, loaded_models)
        all_summaries.extend(summaries)
        all_rows.extend(rows)
        append_csv(all_summaries, Path(args.summary_csv))
        append_csv(all_rows, Path(args.per_image_csv))

    print("Summary CSV:", args.summary_csv)
    print("Per-image CSV:", args.per_image_csv)
    print(all_summaries)


if __name__ == "__main__":
    main()
