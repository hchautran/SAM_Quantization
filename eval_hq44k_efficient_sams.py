#!/usr/bin/env python3
"""Evaluate EfficientSAM and EfficientViT-SAM on HQ44K with GT box prompts."""

import argparse
import csv
import datetime
import gc
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SAM_HQ_ROOT = ROOT / "sam-hq"
EFFICIENTSAM_ROOT = ROOT / "EfficientSAM"
EFFICIENTVIT_ROOT = ROOT / "efficientvit"

for path in (ROOT, SAM_HQ_ROOT, EFFICIENTSAM_ROOT, EFFICIENTVIT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data_utils import OnlineDataset  # noqa: E402
from efficient_sam.efficient_sam import build_efficient_sam  # noqa: E402
from train.train import compute_boundary_iou, compute_iou  # noqa: E402
from train.utils.dataloader import Resize, get_im_gt_name_dict  # noqa: E402
import train.utils.misc as misc  # noqa: E402


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


def get_default_datasets():
    return [
        {"name": "ThinObject5k-TR", "im_dir": "./data/thin_object_detection/ThinObject5K/images_train", "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_train", "im_ext": ".jpg", "gt_ext": ".png"},
        {"name": "DIS5K-TR", "im_dir": "./data/DIS5K/DIS-TR/im", "gt_dir": "./data/DIS5K/DIS-TR/gt", "im_ext": ".jpg", "gt_ext": ".png"},
        {"name": "DIS5K-VD", "im_dir": "./data/DIS5K/DIS-VD/im", "gt_dir": "./data/DIS5K/DIS-VD/gt", "im_ext": ".jpg", "gt_ext": ".png"},
        {"name": "COIFT", "im_dir": "./data/thin_object_detection/COIFT/images", "gt_dir": "./data/thin_object_detection/COIFT/masks", "im_ext": ".jpg", "gt_ext": ".png"},
        {"name": "HRSOD", "im_dir": "./data/thin_object_detection/HRSOD/images", "gt_dir": "./data/thin_object_detection/HRSOD/masks", "im_ext": ".jpg", "gt_ext": ".png"},
        {"name": "ThinObject5k-TE", "im_dir": "./data/thin_object_detection/ThinObject5K/images_test", "gt_dir": "./data/thin_object_detection/ThinObject5K/masks_test", "im_ext": ".jpg", "gt_ext": ".png"},
    ]


def custom_collate_fn(batch):
    ori_ims = [item["ori_im"] for item in batch]
    collated = {}
    for key in batch[0].keys():
        if key == "ori_im":
            collated[key] = ori_ims
        elif key in ("ori_im_path", "ori_gt_path"):
            collated[key] = [item[key] for item in batch]
        else:
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except Exception:
                collated[key] = [item[key] for item in batch]
    return collated


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


def build_dataloader(dataset_config: Dict, batch_size: int):
    valid_im_gt_list = get_im_gt_name_dict([dataset_config], flag="valid")
    dataset = OnlineDataset(
        [valid_im_gt_list[0]],
        transform=transforms.Compose([Resize([1024, 1024])]),
        eval_ori_resolution=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn,
    )


def tensor_image_to_uint8_np(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image_np, 0, 255).astype(np.uint8)


def prepare_efficientsam_image(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = image.float()
    if image.max().item() > 2.0:
        image = image / 255.0
    return image.unsqueeze(0).to(device)


def boxes_to_efficientsam_prompts(boxes: torch.Tensor, device: torch.device):
    boxes = boxes.to(device=device, dtype=torch.float32)
    points = torch.stack([boxes[:, 0:2], boxes[:, 2:4]], dim=1).unsqueeze(0)
    labels = torch.tensor([2, 3], dtype=torch.int64, device=device).view(1, 1, 2).expand(1, boxes.shape[0], 2)
    return points, labels


def select_best_multimask(masks: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    # masks: [1, N, C, H, W], scores: [1, N, C]
    best = scores[0].argmax(dim=1)
    return masks[0, torch.arange(masks.shape[1], device=masks.device), best]




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
    model = build_efficient_sam(*dims[variant], checkpoint=checkpoint).to(device).eval()
    return model, checkpoint


def load_efficientvit(model_name: str, weight_url: Optional[str], device: str, pretrained: bool = True):
    if pretrained and weight_url is None:
        checkpoint_name = model_name.replace("efficientvit-sam-", "efficientvit_sam_") + ".pt"
        weight_url = str(EFFICIENTVIT_ROOT / "assets/checkpoints/efficientvit_sam" / checkpoint_name)
    install_efficientvit_optional_import_stubs()
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model
    model = create_efficientvit_sam_model(model_name, pretrained=pretrained, weight_url=weight_url).to(device).eval()
    return patch_efficientvit_mask_decoder_for_samhq(model)


@torch.no_grad()
def efficientsam_process_one(model, image, label_box, label_ori, device):
    image_tensor = prepare_efficientsam_image(image, device)
    points, labels = boxes_to_efficientsam_prompts(label_box.view(1, 4), device)
    _, _, input_h, input_w = image_tensor.shape

    total_start = time.perf_counter()
    embeddings, encoder_ms = cuda_timed_call(lambda: model.get_image_embeddings(image_tensor))
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
    total_ms = (time.perf_counter() - total_start) * 1000

    pred = (select_best_multimask(masks, scores) > model.mask_threshold).float().unsqueeze(1)
    target = label_ori.unsqueeze(0).to(device)
    return {
        "encoder_ms": encoder_ms,
        "total_ms": total_ms,
        "iou": compute_iou(pred, target).item(),
        "boundary_iou": compute_boundary_iou(pred, target).item(),
    }


@torch.no_grad()
def efficientvit_process_one(predictor, image, label_box, label_ori, device):
    install_efficientvit_optional_import_stubs()
    from efficientvit.models.efficientvit.sam import SamResize
    image_np = tensor_image_to_uint8_np(image)
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

    box_np = label_box.detach().cpu().numpy().reshape(1, 4)
    box_torch = torch.as_tensor(predictor.apply_boxes(box_np), dtype=torch.float32, device=device)
    masks, scores, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=box_torch,
        multimask_output=True,
        return_logits=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_ms = (time.perf_counter() - total_start) * 1000

    pred = masks[torch.arange(masks.shape[0], device=device), scores.argmax(dim=1)].float().unsqueeze(1)
    target = label_ori.unsqueeze(0).to(device)
    predictor.reset_image()
    return {
        "encoder_ms": encoder_ms,
        "total_ms": total_ms,
        "iou": compute_iou(pred, target).item(),
        "boundary_iou": compute_boundary_iou(pred, target).item(),
    }


def evaluate_model(model_name, model_obj, batch_size, dataloader, num_samples, dataset_name, device):
    reset_memory()
    encoder_times, total_times, ious, boundary_ious = [], [], [], []
    total_images = 0
    total_batches = min(len(dataloader), (num_samples + batch_size - 1) // batch_size)
    progress_bar = tqdm(total=total_batches, desc=f"{dataset_name} {model_name} bs={batch_size}")
    overall_start = time.time()

    for data_val in dataloader:
        if total_images >= num_samples:
            break
        images = data_val["image"]
        labels_val = data_val["label"]
        labels_ori = data_val["ori_label"]
        labels_boxes = misc.masks_to_boxes(labels_val[:, 0, :, :] if labels_val.dim() == 4 else labels_val[0:1, :, :])

        for image, label_box, label_ori in zip(images, labels_boxes, labels_ori):
            if total_images >= num_samples:
                break
            if model_name == "EfficientSAM":
                metrics = efficientsam_process_one(model_obj, image, label_box, label_ori, device)
            else:
                metrics = efficientvit_process_one(model_obj, image, label_box, label_ori, device)
            encoder_times.append(metrics["encoder_ms"])
            total_times.append(metrics["total_ms"])
            ious.append(metrics["iou"])
            boundary_ious.append(metrics["boundary_iou"])
            total_images += 1
        progress_bar.update(1)

    progress_bar.close()
    overall_time = time.time() - overall_start
    memory_stats = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_stats = {
            "peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_memory_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
        }
    return {
        "dataset": dataset_name,
        "batch_size": batch_size,
        "num_images": total_images,
        "total_time_sec": overall_time,
        "throughput_imgs_per_sec": total_images / overall_time if overall_time else 0.0,
        "encoder_per_image_mean_ms": float(np.mean(encoder_times)) if encoder_times else 0.0,
        "encoder_per_image_std_ms": float(np.std(encoder_times)) if encoder_times else 0.0,
        "total_image_mean_ms": float(np.mean(total_times)) if total_times else 0.0,
        "total_image_std_ms": float(np.std(total_times)) if total_times else 0.0,
        "miou": float(np.mean(ious)) if ious else 0.0,
        "miou_std": float(np.std(ious)) if ious else 0.0,
        "boundary_iou": float(np.mean(boundary_ious)) if boundary_ious else 0.0,
        "boundary_iou_std": float(np.std(boundary_ious)) if boundary_ious else 0.0,
        **memory_stats,
        "timestamp": datetime.datetime.now().isoformat(),
    }


def save_results(results: List[Dict], output_dir: str) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = str(Path(output_dir) / f"efficient_sams_hq44k_results_{timestamp}.csv")
    if results:
        fieldnames = []
        for row in results:
            fieldnames = list(dict.fromkeys(fieldnames + list(row.keys())))
        with open(csv_filename, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    return csv_filename


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EfficientSAM/EfficientViT-SAM on HQ44K.")
    parser.add_argument("--models", nargs="+", default=["efficientsam", "efficientvit"], choices=["efficientsam", "efficientvit"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--dataset-idx", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--efficientsam-variant", default="vitt", choices=["vitt", "vits"])
    parser.add_argument("--efficientsam-checkpoint", default=None)
    parser.add_argument("--efficientvit-model", default="efficientvit-sam-l0")
    parser.add_argument("--efficientvit-weight-url", default=None)
    parser.add_argument("--efficientvit-no-pretrained", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "benchmark_results/efficient_sams_hq44k"))
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    loaded = {}
    if "efficientsam" in args.models:
        print("Loading EfficientSAM...")
        model, checkpoint = load_efficientsam(args.efficientsam_variant, args.efficientsam_checkpoint, args.device)
        loaded["EfficientSAM"] = model
        args.efficientsam_checkpoint = checkpoint
    if "efficientvit" in args.models:
        print("Loading EfficientViT-SAM...")
        model = load_efficientvit(args.efficientvit_model, args.efficientvit_weight_url, args.device, not args.efficientvit_no_pretrained)
        install_efficientvit_optional_import_stubs()
        from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
        loaded["EfficientViT-SAM"] = EfficientViTSamPredictor(model)

    datasets = get_default_datasets()
    dataset_indices = args.dataset_idx if args.dataset_idx is not None else range(2, 6)
    all_results = []
    for dataset_idx in dataset_indices:
        dataset_config = datasets[dataset_idx]
        for batch_size in args.batch_sizes:
            dataloader = build_dataloader(dataset_config, batch_size)
            for public_name, model_obj in loaded.items():
                result = evaluate_model(public_name, model_obj, batch_size, dataloader, args.num_samples, dataset_config["name"], device)
                result["model"] = public_name
                result["efficientsam_variant"] = args.efficientsam_variant if public_name == "EfficientSAM" else ""
                result["efficientsam_checkpoint"] = args.efficientsam_checkpoint if public_name == "EfficientSAM" else ""
                result["efficientvit_model"] = args.efficientvit_model if public_name == "EfficientViT-SAM" else ""
                result["efficientvit_weight_url"] = args.efficientvit_weight_url or ""
                result["efficientvit_pretrained"] = not args.efficientvit_no_pretrained
                all_results.append(result)
                print(
                    f"{public_name} {dataset_config['name']} bs={batch_size} "
                    f"enc/img={result['encoder_per_image_mean_ms']:.2f} ms "
                    f"total/img={result['total_image_mean_ms']:.2f} ms "
                    f"mIoU={result['miou']:.4f} B-IoU={result['boundary_iou']:.4f}"
                )
            del dataloader
            reset_memory()

    csv_filename = None if args.no_save else save_results(all_results, args.output_dir)
    if csv_filename:
        print(f"Results saved to: {csv_filename}")
    return all_results


if __name__ == "__main__":
    main()
