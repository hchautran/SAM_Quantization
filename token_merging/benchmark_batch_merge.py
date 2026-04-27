from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SAM_HQ_ROOT = REPO_ROOT / "sam-hq"
for path_item in (REPO_ROOT, SAM_HQ_ROOT, Path(__file__).resolve().parent):
    path_str = str(path_item)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data_utils import OnlineDataset
from merge_processor import (GradBipartiteMergeProcessor, IndAttnToMlp, 
                             ReUseMergeOps, BaseMergeProcessor, ReUseMergeOpsAttn, monkey_patch_merge_blocks)
from sam_engine import get_default_datasets
from segment_anything import SamPredictor, sam_model_registry
from train.train import compute_iou
from train.utils.dataloader import Resize, get_im_gt_name_dict
import train.utils.misc as misc



def custom_collate_fn(batch):
    ori_ims = [item["ori_im"] for item in batch]
    collated = {}
    for key in batch[0].keys():
        if key == "ori_im":
            collated[key] = ori_ims
        elif key in {"ori_im_path", "ori_gt_path"}:
            collated[key] = [item[key] for item in batch]
        else:
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except Exception:
                collated[key] = [item[key] for item in batch]
    return collated


class MinimalBatchMergeEvaluator:
    def reset_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def _time_encoder(self, predictor: SamPredictor, transformed_images: torch.Tensor):
        if transformed_images.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                features, interm_features = predictor.model.image_encoder(transformed_images)
            end.record()
            end.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            start_time = time.perf_counter()
            with torch.no_grad():
                features, interm_features = predictor.model.image_encoder(transformed_images)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return features, interm_features, elapsed_ms

    def process_batch(
        self,
        predictor: SamPredictor,
        images: torch.Tensor,
        labels_boxes: torch.Tensor,
        labels_ori: torch.Tensor,
    ) -> Dict[str, float]:
        device = predictor.device
        batch_size = images.shape[0]

        images = images.to(device)
        labels_boxes = labels_boxes.to(device)
        labels_ori = labels_ori.to(device)
        transformed_images = predictor.model.preprocess(images)

        features, interm_features, encoder_time_ms = self._time_encoder(
            predictor, transformed_images
        )

        all_ious = []
        with torch.no_grad():
            for idx in range(batch_size):
                predictor.features = features[idx : idx + 1]
                predictor.interm_features = [feat[idx : idx + 1] for feat in interm_features]
                predictor.original_size = (images.shape[2], images.shape[3])
                predictor.input_size = tuple(transformed_images.shape[-2:])
                predictor.is_image_set = True

                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=labels_boxes[idx : idx + 1],
                    hq_token_only=True,
                )
                all_ious.append(compute_iou(masks, labels_ori[idx : idx + 1]))

        return {
            "encoder_time_ms": encoder_time_ms,
            "encoder_time_per_image_ms": encoder_time_ms / batch_size,
            "miou": torch.mean(torch.stack(all_ious)).item(),
        }

    def benchmark(
        self,
        predictor: SamPredictor,
        dataloader: DataLoader,
        batch_size: int,
        num_samples: int,
    ) -> Dict[str, float]:
        self.reset_memory()
        encoder_times = []
        e2e_times = []
        mious = []
        total_images = 0

        total_steps = min(len(dataloader), max(1, (num_samples + batch_size - 1) // batch_size))
        progress = tqdm(total=total_steps, desc=f"batch={batch_size}")
        for data_val in dataloader:
            if total_images >= num_samples:
                break

            images = data_val["image"]
            labels_val = data_val["label"]
            labels_ori = data_val["ori_label"]
            remaining = num_samples - total_images
            if images.shape[0] > remaining:
                images = images[:remaining]
                labels_val = labels_val[:remaining]
                labels_ori = labels_ori[:remaining]

            labels_boxes = misc.masks_to_boxes(labels_val[:, 0, :, :])

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                metrics = self.process_batch(predictor, images, labels_boxes, labels_ori)
                end.record()
                end.synchronize()
                e2e_ms = start.elapsed_time(end)
            else:
                start_time = time.perf_counter()
                metrics = self.process_batch(predictor, images, labels_boxes, labels_ori)
                e2e_ms = (time.perf_counter() - start_time) * 1000.0

            current_batch = images.shape[0]
            encoder_times.append(metrics["encoder_time_per_image_ms"])
            e2e_times.append(e2e_ms / current_batch)
            mious.append(metrics["miou"])
            total_images += current_batch
            progress.update(1)

        progress.close()

        result = {
            "batch_size": batch_size,
            "num_images": total_images,
            "encoder_per_image_ms": float(np.mean(encoder_times)),
            "e2e_per_image_ms": float(np.mean(e2e_times)),
            "miou": float(np.mean(mious)),
        }
        print(
            f"batch_size={result['batch_size']} | "
            f"images={result['num_images']} | "
            f"encoder/img={result['encoder_per_image_ms']:.2f} ms | "
            f"e2e/img={result['e2e_per_image_ms']:.2f} ms | "
            f"mIoU={result['miou']:.4f}"
        )
        return result



def build_dataloader(dataset_idx: int, batch_size: int) -> DataLoader:
    datasets = get_default_datasets()
    dataset_cfg = datasets[dataset_idx]
    valid_im_gt_list = get_im_gt_name_dict([dataset_cfg], flag="valid")
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



def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal batch benchmark for SAM token merging")
    parser.add_argument("--model-type", type=str, default="vit_l")
    parser.add_argument("--model-ckt", type=str, default="./ckts/sam_hq_vit_l.pth")
    parser.add_argument("--dataset-idx", type=int, default=2)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4])
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--r", type=int, default=512)
    parser.add_argument("--sx", type=int, default=2)
    parser.add_argument("--sy", type=int, default=2)
    parser.add_argument(
        "--grad-method",
        type=str,
        default="sobel",
        choices=["sobel", "central_difference"],
    )
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--disable-mlp-merge", action="store_true")
    args = parser.parse_args()

    datasets = get_default_datasets()
    if args.dataset_idx < 0 or args.dataset_idx >= len(datasets):
        raise ValueError(f"dataset_idx must be in [0, {len(datasets) - 1}]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.model_ckt).to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    # ReUseMergeOps GradBipartiteMergeProcessor IndAttnToMlp BaseMergeProcessor ReUseMergeOpsAttn
    processor = ReUseMergeOpsAttn(
        r=args.r,
        sx=args.sx,
        sy=args.sy,
        grad_method=args.grad_method,
        layers=args.layers,
        merge_mlp=not args.disable_mlp_merge,
    )

    
    monkey_patch_merge_blocks(predictor.model, processor)

    print(f"dataset={datasets[args.dataset_idx]['name']}")
    print(
        f"merge: r={args.r}, sx={args.sx}, sy={args.sy}, grad_method={args.grad_method}, "
        f"merge_mlp={not args.disable_mlp_merge}, layers={args.layers if args.layers else 'all'}"
    )

    evaluator = MinimalBatchMergeEvaluator()
    results: List[Dict[str, float]] = []
    for batch_size in args.batch_sizes:
        dataloader = build_dataloader(args.dataset_idx, batch_size)
        results.append(evaluator.benchmark(predictor, dataloader, batch_size, args.num_samples))
        del dataloader
        evaluator.reset_memory()
    log_dir = Path("/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/token_merging/benchmark_results")
    log_file = log_dir / f"batch_merge_sx{args.sx}_sy{args.sy}.log"
    print("\nsummary")
    for result in results:
        print(
            f"batch_size={result['batch_size']} | "
            f"encoder/img={result['encoder_per_image_ms']:.2f} ms | "
            f"e2e/img={result['e2e_per_image_ms']:.2f} ms | "
            f"mIoU={result['miou']:.4f}"
        )
        # Save results to a log file
        with log_file.open("a") as f:
            f.write(
                f"r value={args.r} | "
                f"batch_size={result['batch_size']} | "
                f"encoder/img={result['encoder_per_image_ms']:.2f} ms | "
                f"e2e/img={result['e2e_per_image_ms']:.2f} ms | "
                f"mIoU={result['miou']:.4f}\n"
            )

if __name__ == "__main__":
    main()
