import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from processor import SUPPORTED_PROCESSORS, get_encoder_processor, image_encoder_monkey_patch

VALID_IMAGE_SUFFIXES = (".JPEG", ".jpg", ".jpeg", ".png")
SPEED_WARMUP_SAMPLES = 100
SPEED_TIMED_SAMPLES = 100
SPEED_TOTAL_SAMPLES = SPEED_WARMUP_SAMPLES + SPEED_TIMED_SAMPLES


def _strip_facebook_prefix(name: str) -> str:
    name = name.strip()
    if name.startswith("facebook/"):
        name = name[len("facebook/") :]
    return name


def load_ground_truth(gt_path: str) -> List[int]:
    with open(gt_path, "r", encoding="utf-8") as handle:
        return [int(line.strip()) for line in handle.readlines() if line.strip()]


def format_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MiB"


def get_available_shm_bytes() -> Optional[int]:
    try:
        stats = os.statvfs("/dev/shm")
    except OSError:
        return None
    return stats.f_bavail * stats.f_frsize


def resolve_num_workers(
    batch_size: int,
    num_workers: int,
    input_size: Optional[Tuple[int, int, int]],
    prefetch_factor: int = 1,
    label: str = "evaluation",
) -> int:
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


def build_loader_kwargs(batch_size: int, num_workers: int, pin_memory: bool, prefetch_factor: int = 1):
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


def append_speed_benchmark(results_dir: Path, model_id: str, avg_inference_time_ms: float) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    speed_path = results_dir / "speed_benchmark.txt"
    with speed_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{model_id} Avg_Inference_Time_ms: {avg_inference_time_ms:.2f}\n")
    print(f"Avg_Inference_Time_ms: {avg_inference_time_ms:.2f}")
    print(f"Speed benchmark saved to: {speed_path}")
    return speed_path


def run_speed_benchmark(eval_loader, model_fn, device, pin_memory: bool, results_dir: Path, model_id: str):
    if device.type != "cuda":
        raise RuntimeError("--measure-speed requires CUDA for synchronized GPU timing.")

    timed_measurements_ms: List[float] = []
    processed_samples = 0
    print(
        f"Running speed benchmark with {SPEED_WARMUP_SAMPLES} warm-up samples and "
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
                images = images.to(device, non_blocking=pin_memory)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                start_event.record()
                _ = model_fn(images)
                end_event.record()
                torch.cuda.synchronize(device)
                timed_measurements_ms.append(start_event.elapsed_time(end_event))

            processed_samples += batch_samples
            if processed_samples >= SPEED_TOTAL_SAMPLES:
                break

    if len(timed_measurements_ms) != SPEED_TIMED_SAMPLES:
        raise RuntimeError(f"Expected {SPEED_TIMED_SAMPLES} timed samples but measured {len(timed_measurements_ms)}.")

    avg_inference_time_ms = sum(timed_measurements_ms) / len(timed_measurements_ms)
    append_speed_benchmark(results_dir, model_id, avg_inference_time_ms)
    return {"avg_inference_time_ms": avg_inference_time_ms, "num_timed_samples": len(timed_measurements_ms)}


def append_speed_and_acc_summary(
    summary_path: Path,
    model_name: str,
    processor: Optional[str],
    speed_stats: Optional[dict],
    perf_stats: dict,
    args,
):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    processor_name = processor or "NONE"
    inference_time = "n/a" if speed_stats is None else f"{speed_stats['avg_inference_time_ms']:.2f} ms"
    top1 = "n/a" if perf_stats["top1_acc"] is None else f"{perf_stats['top1_acc']:.2f}"
    top5 = "n/a" if perf_stats["top5_acc"] is None else f"{perf_stats['top5_acc']:.2f}"
    sparse_percent = "Sparse percent: n/a"
    if processor is not None:
        sparse_percent = f"Sparse percent: {getattr(args, 'percent', 0.0):.4f}"
    line = (
        f"Model name: {model_name} | "
        f"Processor: {processor_name} | "
        f"Inference time: {inference_time} | "
        f"{sparse_percent} | "
        f"Top1 Acc: {top1} | "
        f"Top5 Acc: {top5}"
    )
    with summary_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(f"Saved summary to {summary_path}")


class ImageNetEvalDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        transform,
        gt_file: Optional[str] = None,
        num_images: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.val_dir = self.data_path / "val"
        if not self.val_dir.exists():
            self.val_dir = self.data_path

        self.image_files = sorted([f for f in os.listdir(self.val_dir) if f.endswith(VALID_IMAGE_SUFFIXES)])
        if num_images is not None:
            self.image_files = self.image_files[:num_images]

        self.transform = transform
        self.labels = load_ground_truth(gt_file) if gt_file else None
        if self.labels is not None:
            if num_images is not None:
                self.labels = self.labels[:num_images]
            if len(self.labels) != len(self.image_files):
                raise ValueError(f"Found {len(self.image_files)} images but {len(self.labels)} labels.")

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


def build_pe_preprocess(image_size: int):
    # perception_models default mean/std is (0.5, 0.5, 0.5)
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def _load_imagenet_zeroshot_assets(language: str = "en"):
    assets_dir = (
        Path(__file__).resolve().parents[2]
        / "perception_models"
        / "apps"
        / "pe"
        / "clip_benchmark"
        / "datasets"
    )
    classnames_path = assets_dir / f"{language}_classnames.json"
    templates_path = assets_dir / f"{language}_zeroshot_classification_templates.json"

    with classnames_path.open("r", encoding="utf-8") as handle:
        classnames = json.load(handle)["imagenet1k"]
    with templates_path.open("r", encoding="utf-8") as handle:
        templates = json.load(handle)["imagenet1k"]

    return classnames, templates


def _tokenize_prompts(tokenizer, prompts: Iterable[str], context_length: int) -> torch.Tensor:
    sot = tokenizer.sot_token_id
    eot = tokenizer.eot_token_id

    all_tokens = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        tokens = [sot] + tokens[: max(0, context_length - 2)] + [eot]
        if len(tokens) < context_length:
            tokens = tokens + [0] * (context_length - len(tokens))
        all_tokens.append(tokens)
    return torch.tensor(all_tokens, dtype=torch.long)


@dataclass
class ZeroShotClassifier:
    classnames: List[str]
    templates: List[str]
    text_features: torch.Tensor  # [num_classes, embed_dim]


def build_zeroshot_classifier(model, device: torch.device, language: str = "en") -> ZeroShotClassifier:
    from core.vision_encoder.tokenizer import SimpleTokenizer

    classnames, templates = _load_imagenet_zeroshot_assets(language=language)
    context_length = int(getattr(model, "context_length", 77))
    tokenizer = SimpleTokenizer(context_length=context_length)

    print(f"Building zero-shot classifier: {len(classnames)} classes x {len(templates)} templates")
    text_features = []

    with torch.inference_mode():
        for classname in tqdm(classnames, desc="Text Embeddings"):
            prompts = [template.format(c=classname) for template in templates]
            token_ids = _tokenize_prompts(tokenizer, prompts, context_length=context_length).to(device)
            feats = model.encode_text(token_ids, normalize=True)  # [T, D]
            feats = feats.mean(dim=0)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            text_features.append(feats)

    text_features = torch.stack(text_features, dim=0)
    return ZeroShotClassifier(classnames=classnames, templates=templates, text_features=text_features)


def _compute_topk(logits: torch.Tensor, k: int = 5):
    probs = logits.softmax(dim=1)
    topk_prob, topk_idx = torch.topk(probs, k=k, dim=1)
    return topk_prob, topk_idx


def evaluate_imagenet(
    data_path: str,
    model_name: str,
    batch_size: int = 64,
    gt_file: Optional[str] = None,
    num_images: Optional[int] = None,
    num_workers: int = 4,
    mode: str = "clip_zeroshot",
    language: str = "en",
    measure_speed: bool = False,
    results_dir: Optional[Path] = None,
    save_features_path: Optional[str] = None,
    linear_head_path: Optional[str] = None,
    processor: Optional[str] = None,
    summary_path: Optional[Path] = None,
    args=None,
):
    model_key = _strip_facebook_prefix(model_name)
    if results_dir is None:
        results_dir = Path(__file__).parent / "results"
    if summary_path is None:
        summary_path = results_dir / "speed_and_Acc.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    eval_num_images = num_images
    if measure_speed:
        print("measure_speed enabled: running image-encoder speed benchmark before accuracy evaluation")

    if mode == "clip_zeroshot":
        from core.vision_encoder.pe import CLIP

        print(f"Loading PE CLIP model: {model_key} (pretrained=True)")
        model = CLIP.from_config(model_key, pretrained=True).eval().to(device)
        if processor:
            print(f"Applying processor to CLIP visual encoder: {processor}")
            enc_processor = get_encoder_processor(processor)
            if args is not None and hasattr(enc_processor, "set_params"):
                enc_processor.set_params(args)
            image_encoder_monkey_patch(model.visual, processor=enc_processor, device=device)
        image_size = int(getattr(model, "image_size", getattr(model.visual, "image_size")))
        preprocess = build_pe_preprocess(image_size=image_size)
        classifier = build_zeroshot_classifier(model, device=device, language=language)

        logit_scale = model.logit_scale.exp().detach()
        print(f"Using device: {device} image_size={image_size} logit_scale={logit_scale.item():.3f}")

        def forward_images(images: torch.Tensor) -> torch.Tensor:
            image_features = model.encode_image(images, normalize=True)
            return (logit_scale * image_features) @ classifier.text_features.T

        inference_model = forward_images

        def speed_model(images: torch.Tensor) -> torch.Tensor:
            return model.encode_image(images, normalize=True)

    elif mode == "vision_features":
        from core.vision_encoder import pe

        print(f"Loading PE vision encoder: {model_key} (pretrained=True)")
        # model = pe.VisionTransformer.from_config(model_key, pretrained=True).eval().to(device)
        print(model)
        # exit()
        if processor:
            print(f"Applying processor to vision encoder: {processor}")
            enc_processor = get_encoder_processor(processor)
            if args is not None and hasattr(enc_processor, "set_params"):
                enc_processor.set_params(args)
            image_encoder_monkey_patch(model, processor=enc_processor, device=device)
        image_size = int(getattr(model, "image_size", 224))
        preprocess = build_pe_preprocess(image_size=image_size)
        print(f"Using device: {device} image_size={image_size}")

        linear_head = None
        if linear_head_path:
            state = torch.load(linear_head_path, map_location="cpu")
            if isinstance(state, dict) and "weight" in state:
                weight = state["weight"]
                bias = state.get("bias", None)
            elif isinstance(state, dict) and "state_dict" in state:
                weight = state["state_dict"]["weight"]
                bias = state["state_dict"].get("bias", None)
            else:
                raise ValueError(f"Unsupported linear head checkpoint format: {linear_head_path}")

            linear_head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
            linear_head.weight.data.copy_(weight)
            if bias is not None:
                linear_head.bias.data.copy_(bias)
            linear_head = linear_head.to(device).eval()
            print(f"Loaded linear head from {linear_head_path} -> out_features={linear_head.out_features}")

        def forward_images(images: torch.Tensor) -> torch.Tensor:
            feats = model(images)
            if linear_head is None:
                raise RuntimeError(
                    "--mode vision_features requires --linear-head-path to produce logits/accuracy. "
                    "To only dump embeddings, pass --save-features-path and omit --linear-head-path."
                )
            return linear_head(feats)

        inference_model = forward_images
        speed_model = model

    else:
        raise ValueError(f"Unsupported --mode: {mode}")

    eval_dataset = ImageNetEvalDataset(
        data_path=data_path,
        transform=preprocess,
        gt_file=gt_file,
        num_images=eval_num_images,
    )

    input_size = (3, image_size, image_size)
    eval_num_workers = resolve_num_workers(
        batch_size=batch_size,
        num_workers=num_workers,
        input_size=input_size,
        prefetch_factor=1,
        label="evaluation",
    )
    eval_loader = DataLoader(
        eval_dataset,
        **build_loader_kwargs(
            batch_size=batch_size,
            num_workers=eval_num_workers,
            pin_memory=pin_memory,
            prefetch_factor=1,
        ),
    )

    speed_stats = None
    if measure_speed:
        speed_dataset = ImageNetEvalDataset(
            data_path=data_path,
            transform=preprocess,
            gt_file=gt_file,
            num_images=SPEED_TOTAL_SAMPLES,
        )
        speed_num_workers = resolve_num_workers(
            batch_size=1,
            num_workers=num_workers,
            input_size=input_size,
            prefetch_factor=1,
            label="speed",
        )
        speed_loader = DataLoader(
            speed_dataset,
            **build_loader_kwargs(
                batch_size=1,
                num_workers=speed_num_workers,
                pin_memory=pin_memory,
                prefetch_factor=1,
            ),
        )
        benchmark_id = f"{mode}:{model_key}:image_encoder"
        speed_stats = run_speed_benchmark(
            eval_loader=speed_loader,
            model_fn=speed_model,
            device=device,
            pin_memory=pin_memory,
            results_dir=results_dir,
            model_id=benchmark_id,
        )

    top1_correct = 0
    top5_correct = 0
    total = 0
    all_predictions = []

    feature_batches = []
    label_batches = []
    file_list: List[str] = []

    print(f"Found {len(eval_dataset)} images")
    if eval_dataset.labels is not None:
        print(f"Loaded ground truth with {len(eval_dataset.labels)} labels")
    else:
        print("No ground truth file provided. Running inference only.")

    print("Starting evaluation...")
    with torch.inference_mode():
        for batch in tqdm(eval_loader, total=len(eval_loader), desc="Evaluation"):
            if batch is None:
                continue

            images, labels, image_files = batch
            images = images.to(device, non_blocking=pin_memory)

            if mode == "vision_features" and save_features_path and not linear_head_path:
                feats = model(images)
                feature_batches.append(feats.detach().cpu())
                label_batches.append(labels.detach().cpu())
                file_list.extend(image_files)
                continue

            logits = inference_model(images)
            top5_prob, top5_idx = _compute_topk(logits, k=5)

            for j, img_file in enumerate(image_files):
                pred_class = top5_idx[j, 0].item()
                top5_preds = top5_idx[j].cpu().tolist()
                all_predictions.append(
                    {
                        "image": img_file,
                        "top1": pred_class,
                        "top5": top5_preds,
                        "top5_probs": top5_prob[j].cpu().tolist(),
                    }
                )

                if eval_dataset.labels is not None:
                    true_label = labels[j].item()
                    if pred_class == true_label:
                        top1_correct += 1
                    if true_label in top5_preds:
                        top5_correct += 1
                    total += 1

    if mode == "vision_features" and save_features_path and not linear_head_path:
        out_path = Path(save_features_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features = torch.cat(feature_batches, dim=0) if feature_batches else torch.empty((0,))
        labels = torch.cat(label_batches, dim=0) if label_batches else torch.empty((0,), dtype=torch.long)
        torch.save({"features": features, "labels": labels, "files": file_list, "model": model_key}, out_path)
        print(f"Saved features to {out_path}")
        result = {"saved_features": str(out_path), "num_images": int(features.shape[0]), "speed": speed_stats}
        append_speed_and_acc_summary(
            summary_path=summary_path,
            model_name=model_name,
            processor=processor,
            speed_stats=speed_stats,
            perf_stats={"top1_acc": None, "top5_acc": None},
            args=args,
        )
        return result

    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print("=" * 50)

    perf_stats = {"top1_acc": None, "top5_acc": None}
    if eval_dataset.labels is not None and total > 0:
        top1_acc = 100.0 * top1_correct / total
        top5_acc = 100.0 * top5_correct / total
        perf_stats["top1_acc"] = top1_acc
        perf_stats["top5_acc"] = top5_acc
        print(f"Total images evaluated: {total}")
        print(f"Top-1 Accuracy: {top1_acc:.2f}%")
        print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    else:
        print(f"Inference completed on {len(all_predictions)} images")
        print("To compute accuracy, provide ground truth file with --gt-file")

    append_speed_and_acc_summary(
        summary_path=summary_path,
        model_name=model_name,
        processor=processor,
        speed_stats=speed_stats,
        perf_stats=perf_stats,
        args=args,
    )

    return {"predictions": all_predictions, "metrics": perf_stats, "speed": speed_stats}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Perception Encoder (PE) models on ImageNet.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/data/imagenet",
        help="Path to ImageNet dataset (expects a `val/` folder or flat val folder).",
    )
    parser.add_argument(
        "--gt-file",
        type=str,
        default="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/data/imagenet/labels/val_labels_hf_0based.txt",
        help="Path to ground truth file (one 0-based int label per line, sorted to match val filenames).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/PE-Core-B16-224",
        help="Model name, e.g. facebook/PE-Core-B16-224, facebook/PE-Core-S16-384, facebook/PE-Core-T16-384.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--num_images", type=int, default=None, help="Limit number of val images")
    parser.add_argument(
        "--mode",
        type=str,
        default="clip_zeroshot",
        choices=["clip_zeroshot", "vision_features"],
        help="clip_zeroshot: PE CLIP zero-shot ImageNet-1k; vision_features: dump vision features (optionally with a linear head).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language for classnames/templates (uses perception_models clip_benchmark JSON assets).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "results" / "results.json"),
        help="Where to save predictions JSON (clip_zeroshot or vision_features+linear head).",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(Path(__file__).parent / "results" / "speed_and_Acc.txt"),
        help="Where to append speed/accuracy summaries when --measure-speed is enabled.",
    )
    parser.add_argument(
        "--save-features-path",
        type=str,
        default=None,
        help="When --mode vision_features and no --linear-head-path, save a .pt with features/labels/files.",
    )
    parser.add_argument(
        "--linear-head-path",
        type=str,
        default=None,
        help="Optional .pt checkpoint containing {weight, bias} to classify vision features (1000-way for ImageNet).",
    )
    parser.add_argument(
        "--measure-speed",
        action="store_true",
        help="Measure image-encoder speed on 200 samples with 100 warm-up and 100 timed iterations using CUDA events (CUDA only; excludes dataloading and classification head/text logits).",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default=None,
        choices=sorted(SUPPORTED_PROCESSORS),
        help="Optional processor to monkey-patch PE SelfAttention. The processor must implement `process_pe` or `process_sequence`.",
    )
    parser.add_argument("--percent", type=float, default=0.0, help="Processor parameter passthrough.")
    parser.add_argument("--percent-global", type=float, default=0.0, help="Processor parameter passthrough.")
    parser.add_argument("--prune-global", action="store_true", help="Processor parameter passthrough.")
    parser.add_argument("--n-bits", type=int, default=16, help="Processor parameter passthrough.")
    parser.add_argument("--high-entropy", action="store_true", help="Processor parameter passthrough.")
    parser.add_argument("--num-calib-samples", type=int, default=16, help="Processor parameter passthrough.")
    parser.add_argument("--en-weight-quant", type=str, default="per_channel", help="Processor parameter passthrough.")
    parser.add_argument("--en-act-quant", type=str, default="per_token", help="Processor parameter passthrough.")

    args = parser.parse_args()

    results = evaluate_imagenet(
        data_path=args.data_path,
        model_name=args.model,
        batch_size=args.batch_size,
        gt_file=args.gt_file,
        num_images=args.num_images,
        num_workers=args.num_workers,
        mode=args.mode,
        language=args.language,
        measure_speed=args.measure_speed,
        results_dir=Path(__file__).parent / "results",
        save_features_path=args.save_features_path,
        linear_head_path=args.linear_head_path,
        processor=args.processor,
        summary_path=Path(args.summary_path),
        args=args,
    )

    predictions = results["predictions"] if isinstance(results, dict) and "predictions" in results else None
    if args.output and isinstance(predictions, list):
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(predictions, handle, indent=2)
        print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
