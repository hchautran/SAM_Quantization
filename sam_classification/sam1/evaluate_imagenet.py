import argparse
import json
import os
from pathlib import Path
from typing import Callable

import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from processor import SUPPORTED_PROCESSORS, get_encoder_processor, image_encoder_monkey_patch


VALID_IMAGE_SUFFIXES = (".JPEG", ".jpg", ".jpeg", ".png")
SPEED_WARMUP_SAMPLES = 100
SPEED_TIMED_SAMPLES = 100
SPEED_TOTAL_SAMPLES = SPEED_WARMUP_SAMPLES + SPEED_TIMED_SAMPLES


def load_ground_truth(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f.readlines() if line.strip()]


def format_mib(num_bytes):
    return f"{num_bytes / (1024 * 1024):.1f} MiB"


def get_available_shm_bytes():
    try:
        stats = os.statvfs("/dev/shm")
    except OSError:
        return None
    return stats.f_bavail * stats.f_frsize


def resolve_num_workers(batch_size, num_workers, input_size, prefetch_factor=1, label="evaluation"):
    if num_workers <= 0:
        return 0
    if not input_size or len(input_size) != 3:
        return num_workers

    channels, height, width = input_size
    batch_bytes = batch_size * channels * height * width * 4
    estimated_prefetch_bytes = batch_bytes * num_workers * prefetch_factor
    shm_bytes = get_available_shm_bytes()

    if shm_bytes is None:
        return num_workers

    print(
        f"{label.capitalize()} loader estimate: batch={format_mib(batch_bytes)}, "
        f"prefetch={format_mib(estimated_prefetch_bytes)}, /dev/shm={format_mib(shm_bytes)}"
    )
    if estimated_prefetch_bytes >= int(shm_bytes * 0.8):
        print(
            f"Disabling DataLoader workers for {label} because prefetched batches would exceed /dev/shm. "
            "Lower batch_size or increase shared memory to use workers."
        )
        return 0

    return num_workers


def collate_skip_errors(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    images, labels, image_files = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(image_files)


def build_loader_kwargs(batch_size, num_workers, pin_memory, prefetch_factor=1):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_skip_errors,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def run_speed_benchmark(
    eval_loader,
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    device,
    pin_memory,
    benchmark_name: str = "model",
):
    if device.type != "cuda":
        raise RuntimeError("--measure-speed requires CUDA timing with torch.cuda.Event.")

    timed_measurements_ms = []
    processed_samples = 0
    print(
        f"Running {benchmark_name} speed benchmark with {SPEED_WARMUP_SAMPLES} warm-up samples and "
        f"{SPEED_TIMED_SAMPLES} timed samples."
    )

    with torch.inference_mode():
        for batch in tqdm(eval_loader, total=min(len(eval_loader), SPEED_TOTAL_SAMPLES), desc="Speed Benchmark"):
            if batch is None:
                continue

            images, _, _ = batch
            batch_samples = images.size(0)
            if processed_samples >= SPEED_TOTAL_SAMPLES:
                break

            if processed_samples < SPEED_WARMUP_SAMPLES:
                images = images.to(device, non_blocking=pin_memory)
                _ = model_fn(images)
                torch.cuda.synchronize(device)
            else:
                torch.cuda.synchronize(device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                images = images.to(device, non_blocking=pin_memory)
                _ = model_fn(images)
                end_event.record()
                end_event.synchronize()
                timed_measurements_ms.append(start_event.elapsed_time(end_event))

            processed_samples += batch_samples
            if processed_samples >= SPEED_TOTAL_SAMPLES:
                break

    if len(timed_measurements_ms) != SPEED_TIMED_SAMPLES:
        raise RuntimeError(
            f"Expected {SPEED_TIMED_SAMPLES} timed samples but measured {len(timed_measurements_ms)}."
        )

    avg_inference_time_ms = sum(timed_measurements_ms) / len(timed_measurements_ms)
    print(f"{benchmark_name} Avg_Inference_Time_ms: {avg_inference_time_ms:.2f}")
    return {"avg_inference_time_ms": avg_inference_time_ms, "num_timed_samples": len(timed_measurements_ms)}


class ImageNetCalibDataset(Dataset):
    def __init__(self, data_path, transform=None, num_samples=16):
        self.data_path = Path(data_path)
        self.val_dir = self.data_path / "val"
        if not self.val_dir.exists():
            self.val_dir = self.data_path

        self.image_files = sorted(
            [f for f in os.listdir(self.val_dir) if f.endswith(VALID_IMAGE_SUFFIXES)]
        )[:num_samples]

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.val_dir / self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, -1, self.image_files[idx]


class ImageNetEvalDataset(Dataset):
    def __init__(self, data_path, transform, gt_file=None, num_images=None):
        self.data_path = Path(data_path)
        self.val_dir = self.data_path / "val"
        if not self.val_dir.exists():
            self.val_dir = self.data_path

        self.image_files = sorted(
            [f for f in os.listdir(self.val_dir) if f.endswith(VALID_IMAGE_SUFFIXES)]
        )
        if num_images is not None:
            self.image_files = self.image_files[:num_images]

        self.transform = transform
        self.labels = load_ground_truth(gt_file) if gt_file else None
        if self.labels is not None:
            if num_images is not None:
                self.labels = self.labels[:num_images]
            if len(self.labels) != len(self.image_files):
                raise ValueError(
                    f"Found {len(self.image_files)} images but {len(self.labels)} labels."
                )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = self.val_dir / img_file
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            label = -1 if self.labels is None else self.labels[idx]
            return img, label, img_file
        except Exception as exc:
            print(f"Error loading {img_file}: {exc}")
            return None


def get_model_name(model):
    if model == "vit_b":
        return "mvitv2_small.fb_in1k"
    if model == "vit_l":
        return "mvitv2_large.fb_in1k"
    if model == "vit_h":
        return "mvitv2_huge_cls.fb_inw21k"
    raise ValueError(f"Unsupported model: {model}")


def build_eval_loader(dataset, batch_size, num_workers, pin_memory, input_size):
    eval_num_workers = resolve_num_workers(
        batch_size=batch_size,
        num_workers=num_workers,
        input_size=input_size,
        prefetch_factor=1,
        label="evaluation",
    )
    return DataLoader(
        dataset,
        **build_loader_kwargs(
            batch_size=batch_size,
            num_workers=eval_num_workers,
            pin_memory=pin_memory,
            prefetch_factor=1,
        ),
    )


def evaluate_accuracy(eval_loader, model, device, pin_memory, labels_available):
    top1_correct = 0
    top5_correct = 0
    total = 0
    all_predictions = []

    print("Starting evaluation...")
    with torch.inference_mode():
        for batch in tqdm(eval_loader, total=len(eval_loader), desc="Evaluation"):
            if batch is None:
                continue

            images, labels, image_files = batch
            images = images.to(device, non_blocking=pin_memory)
            output = model(images)
            probabilities = output.softmax(dim=1)
            top5_prob, top5_indices = torch.topk(probabilities, k=5, dim=1)

            for j, img_file in enumerate(image_files):
                pred_class = top5_indices[j, 0].item()
                top5_preds = top5_indices[j].cpu().tolist()
                all_predictions.append(
                    {
                        "image": img_file,
                        "top1": pred_class,
                        "top5": top5_preds,
                        "top5_probs": top5_prob[j].cpu().tolist(),
                    }
                )

                if labels_available:
                    true_label = labels[j].item()
                    if pred_class == true_label:
                        top1_correct += 1
                    if true_label in top5_preds:
                        top5_correct += 1
                    total += 1

    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)

    metrics = {"top1_acc": None, "top5_acc": None, "total": total}
    if labels_available and total > 0:
        metrics["top1_acc"] = 100.0 * top1_correct / total
        metrics["top5_acc"] = 100.0 * top5_correct / total
        print(f"Total images evaluated: {total}")
        print(f"Top-1 Accuracy: {metrics['top1_acc']:.2f}%")
        print(f"Top-5 Accuracy: {metrics['top5_acc']:.2f}%")
    else:
        print(f"Inference completed on {len(all_predictions)} images")
        print("To compute accuracy, provide ground truth file with --gt_file")

    return metrics, all_predictions


def format_sparse_percent(processor, args):
    if processor is None:
        return "Sparse percent: n/a"
    if processor == "BASE":
        return "local:0.0000 global:0.0000"
    return f"local:{getattr(args, 'percent', 0.0):.4f} global:{getattr(args, 'percent_global', 0.0):.4f}"


def append_speed_and_acc_summary(summary_path: Path, model_name: str, processor, speed_stats, perf_stats, args):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    processor_name = processor or "NONE"
    inference_time = "n/a" if speed_stats is None else f"{speed_stats['avg_inference_time_ms']:.2f} ms"
    top1 = "n/a" if perf_stats["top1_acc"] is None else f"{perf_stats['top1_acc']:.2f}"
    top5 = "n/a" if perf_stats["top5_acc"] is None else f"{perf_stats['top5_acc']:.2f}"
    line = (
        f"Model name: {model_name} | "
        f"Processor: {processor_name} | "
        f"Inference time: {inference_time} | "
        f"{format_sparse_percent(processor, args)} | "
        f"Top1 Acc: {top1} | "
        f"Top5 Acc: {top5}"
    )
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(f"Saved summary to {summary_path}")


def evaluate_imagenet(
    data_path,
    modelsam="vit_l",
    batch_size=32,
    gt_file=None,
    num_images=None,
    processor=None,
    args=None,
    num_workers=4,
    measure_speed=False,
    results_dir=None,
    summary_path=None,
):
    model_name = get_model_name(modelsam)
    print(f"Loading model: {model_name}")

    if results_dir is None:
        results_dir = Path(__file__).parent / "results"
    if summary_path is None:
        summary_path = results_dir / "speed_and_Acc.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = timm.create_model(model_name, pretrained=True)
    model = model.eval().to(device)
    print(f"Using device: {device}")
    print(f"Model output classes: {model.num_classes}")

    data_config = timm.data.resolve_model_data_config(model)
    transforms_fn = timm.data.create_transform(**data_config, is_training=False)
    input_size = data_config.get("input_size")

    if processor is not None:
        print(f"Applying processor: {processor}")
        enc_processor = get_encoder_processor(processor)
        enc_processor.set_params(args)

        # calib_dataset = ImageNetCalibDataset(data_path, transform=transforms_fn, num_samples=args.num_calib_samples)
        # calib_batch_size = max(1, min(args.num_calib_samples, len(calib_dataset)))
        # calib_num_workers = resolve_num_workers(
        #     batch_size=calib_batch_size,
        #     num_workers=min(num_workers, 2),
        #     input_size=input_size,
        #     prefetch_factor=1,
        #     label="calibration",
        # )
        # calib_dataloader = DataLoader(
        #     calib_dataset,
        #     **build_loader_kwargs(
        #         batch_size=calib_batch_size,
        #         num_workers=calib_num_workers,
        #         pin_memory=pin_memory,
        #         prefetch_factor=1,
        #     ),
        # )

        # enc_processor.calibrate_sam_classification(
        #     model,
        #     dataloader=calib_dataloader,
        #     num_samples=args.num_calib_samples,
        #     device=device,
        # )

        image_encoder_monkey_patch(
            model,
            processor=enc_processor,
            n_bits=args.n_bits,
            weight_quant=args.en_weight_quant,
            act_quant=args.en_act_quant,
            device=device,
        )

    speed_stats = None
    if measure_speed:
        speed_dataset = ImageNetEvalDataset(
            data_path=data_path,
            transform=transforms_fn,
            gt_file=gt_file,
            num_images=SPEED_TOTAL_SAMPLES,
        )
        speed_loader = build_eval_loader(speed_dataset, 1, num_workers, pin_memory, input_size)
        # import ipdb; ipdb.set_trace()
        speed_stats = run_speed_benchmark(
            speed_loader,
            lambda images: model.forward_features(images),
            device,
            pin_memory,
            benchmark_name=f"{model_name} encoder",
        )

    eval_dataset = ImageNetEvalDataset(
        data_path=data_path,
        transform=transforms_fn,
        gt_file=gt_file,
        num_images=num_images,
    )
    eval_loader = build_eval_loader(eval_dataset, batch_size, num_workers, pin_memory, input_size)

    print(f"Found {len(eval_dataset)} images")
    if eval_dataset.labels is not None:
        print(f"Loaded ground truth with {len(eval_dataset.labels)} labels")
    else:
        print("No ground truth file provided. Running inference only.")

    perf_stats, predictions = evaluate_accuracy(
        eval_loader=eval_loader,
        model=model,
        device=device,
        pin_memory=pin_memory,
        labels_available=eval_dataset.labels is not None,
    )

    if measure_speed:
        append_speed_and_acc_summary(summary_path, modelsam, processor, speed_stats, perf_stats, args)

    return {"speed": speed_stats, "metrics": perf_stats, "predictions": predictions}


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM1 classification model on ImageNet")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/data/imagenet",
        help="Path to ImageNet dataset",
    )
    parser.add_argument("--model_type", type=str, default="vit_l", help="Model type: vit_b, vit_l, vit_h")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument(
        "--gt_file",
        type=str,
        default="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/data/imagenet/labels/val_labels_hf_0based.txt",
        help="Path to ground truth file",
    )
    parser.add_argument("--num_images", type=int, default=50000, help="Number of images to evaluate")
    parser.add_argument(
        "--output",
        type=str,
        default="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/sam_classification/sam1/results/results.json",
        help="Path to save predictions",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/sam_classification/sam1/results/speed_and_Acc.txt",
        help="Where to append speed/accuracy summaries when --measure-speed is enabled.",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default=None,
        choices=sorted(SUPPORTED_PROCESSORS),
        help="Processor to use",
    )
    parser.add_argument("--percent", type=float, default=0.0, help="Local sparse percent")
    parser.add_argument("--percent-global", type=float, default=0.0, help="Global sparse percent")
    parser.add_argument("--prune-global", action="store_true", help="Enable global pruning")
    parser.add_argument("--n-bits", type=int, default=16, help="Number of quantization bits")
    parser.add_argument("--high-entropy", action="store_true", help="Use high entropy")
    parser.add_argument("--num-calib-samples", type=int, default=16, help="Number of calibration samples")
    parser.add_argument("--en-weight-quant", type=str, default="per_channel", help="Encoder weight quantization method")
    parser.add_argument("--en-act-quant", type=str, default="per_token", help="Encoder activation quantization method")
    parser.add_argument(
        "--measure-speed",
        action="store_true",
        help="Measure image_encoder-only speed first, then run accuracy evaluation and append one summary line.",
    )

    args = parser.parse_args()

    results = evaluate_imagenet(
        data_path=args.data_path,
        modelsam=args.model_type,
        batch_size=args.batch_size,
        gt_file=args.gt_file,
        num_images=args.num_images,
        processor=args.processor,
        args=args,
        num_workers=args.num_workers,
        measure_speed=args.measure_speed,
        results_dir=Path(__file__).parent / "results",
        summary_path=Path(args.summary_path),
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results["predictions"], f, indent=2)
        print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
