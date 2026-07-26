#!/usr/bin/env python3
"""Fine-tune EfficientViT-SAM on DIS5K-TR with GT-box prompts."""

import argparse
import csv
import datetime
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
SAM_HQ_ROOT = ROOT / "sam-hq"
EFFICIENTVIT_ROOT = ROOT / "efficientvit"

for path in (ROOT, SAM_HQ_ROOT, EFFICIENTVIT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from data_utils import OnlineDataset  # noqa: E402
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


def dis_tr_config() -> Dict:
    return {
        "name": "DIS5K-TR",
        "im_dir": str(ROOT / "data/DIS5K/DIS-TR/im"),
        "gt_dir": str(ROOT / "data/DIS5K/DIS-TR/gt"),
        "im_ext": ".jpg",
        "gt_ext": ".png",
    }


def custom_collate_fn(batch):
    collated = {}
    for key in batch[0].keys():
        if key in ("ori_im", "ori_im_path", "ori_gt_path"):
            collated[key] = [item[key] for item in batch]
        else:
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except Exception:
                collated[key] = [item[key] for item in batch]
    return collated


def build_dataloader(batch_size: int, workers: int, max_samples: Optional[int]):
    valid_im_gt_list = get_im_gt_name_dict([dis_tr_config()], flag="valid")
    dataset = OnlineDataset(
        [valid_im_gt_list[0]],
        transform=transforms.Compose([Resize([1024, 1024])]),
        eval_ori_resolution=False,
    )
    if max_samples is not None and max_samples > 0:
        dataset.dataset = {key: value[:max_samples] for key, value in dataset.dataset.items()}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=custom_collate_fn,
    )


def default_weight_path(model_name: str) -> str:
    checkpoint_name = model_name.replace("efficientvit-sam-", "efficientvit_sam_") + ".pt"
    return str(EFFICIENTVIT_ROOT / "assets/checkpoints/efficientvit_sam" / checkpoint_name)


def load_model(args):
    install_efficientvit_optional_import_stubs()
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model

    weight_url = args.input_checkpoint or default_weight_path(args.model)
    model = create_efficientvit_sam_model(args.model, pretrained=not args.no_pretrained, weight_url=weight_url)
    model = patch_efficientvit_mask_decoder_for_samhq(model)
    model.to(args.device).train()
    return model, weight_url


def set_trainable(model, mode: str):
    for param in model.parameters():
        param.requires_grad = False
    if mode == "all":
        for param in model.parameters():
            param.requires_grad = True
    elif mode == "decoder":
        for module in (model.prompt_encoder, model.mask_decoder):
            for param in module.parameters():
                param.requires_grad = True
    elif mode == "mask_decoder":
        for param in model.mask_decoder.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported trainable mode: {mode}")


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def prepare_batch(images: torch.Tensor, labels: torch.Tensor, model, device: torch.device):
    images = images.float().to(device, non_blocking=True)
    labels = labels.float().to(device, non_blocking=True)
    target = int(model.image_size[1])
    _, _, h, w = images.shape

    masks = (labels > 128).float()
    boxes = misc.masks_to_boxes(labels[:, 0, :, :]).to(device=device, dtype=torch.float32)
    empty = ~torch.isfinite(boxes).all(dim=1) | (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
    if empty.any():
        boxes[empty] = torch.tensor([0, 0, w - 1, h - 1], device=device, dtype=torch.float32)

    if h != target or w != target:
        scale_x = target / float(w)
        scale_y = target / float(h)
        images = F.interpolate(images, size=(target, target), mode="bilinear", align_corners=False)
        masks = F.interpolate(masks, size=(target, target), mode="nearest")
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

    images = images / 255.0
    mean = torch.tensor([123.675 / 255, 116.28 / 255, 103.53 / 255], device=device).view(1, 3, 1, 1)
    std = torch.tensor([58.395 / 255, 57.12 / 255, 57.375 / 255], device=device).view(1, 3, 1, 1)
    images = (images - mean) / std
    return images, masks, boxes


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    probs = logits.sigmoid()
    numerator = 2 * (probs * targets).flatten(1).sum(dim=1)
    denominator = probs.flatten(1).sum(dim=1) + targets.flatten(1).sum(dim=1)
    return 1 - ((numerator + eps) / (denominator + eps)).mean()


def forward_box_logits(model, images: torch.Tensor, boxes: torch.Tensor):
    image_embeddings = model.image_encoder(images)
    logits_list = []
    iou_list = []
    for idx in range(images.shape[0]):
        box = boxes[idx : idx + 1]
        sparse_embeddings, dense_embeddings = model.prompt_encoder(points=None, boxes=box, masks=None)
        low_res_logits, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings[idx : idx + 1],
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        logits = model.postprocess_masks(low_res_logits, images.shape[-2:], images.shape[-2:])
        logits_list.append(logits)
        iou_list.append(iou_predictions)
    return torch.cat(logits_list, dim=0), torch.cat(iou_list, dim=0)


def compute_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor):
    pred = logits.sigmoid() > 0.5
    target = targets > 0.5
    inter = (pred & target).flatten(1).sum(dim=1).float()
    union = (pred | target).flatten(1).sum(dim=1).float().clamp_min(1)
    return (inter / union).mean()


def save_checkpoint(model, optimizer, args, epoch: int, step: int, path: Path, source_checkpoint: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "source_checkpoint": source_checkpoint,
            "args": vars(args),
            "timestamp": datetime.datetime.now().isoformat(),
        },
        path,
    )


def append_csv(rows: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def init_wandb(args):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb not installed in this environment. Install wandb or run without --wandb.") from exc
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or None,
        config=vars(args),
    )
    return run


def train(args):
    wandb_run = init_wandb(args)
    device = torch.device(args.device)
    model, source_checkpoint = load_model(args)
    set_trainable(model, args.trainable)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())
    dataloader = build_dataloader(args.batch_size, args.workers, args.max_samples)
    reset_memory()

    rows = []
    global_step = 0
    output_dir = Path(args.output_dir)
    final_path = Path(args.output_checkpoint) if args.output_checkpoint else output_dir / f"{args.model.replace('-', '_')}_dis_tr_{args.epochs}ep_{args.trainable}.pt"
    log_csv = output_dir / f"{args.model.replace('-', '_')}_dis_tr_train_log.csv"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        progress = tqdm(dataloader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in progress:
            global_step += 1
            images, masks, boxes = prepare_batch(batch["image"], batch["label"], model, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
                logits, _ = forward_box_logits(model, images, boxes)
                bce = F.binary_cross_entropy_with_logits(logits, masks)
                dice = dice_loss(logits, masks)
                loss = bce + args.dice_weight * dice

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if global_step % args.log_every == 0 or global_step == 1:
                with torch.no_grad():
                    iou = compute_iou_from_logits(logits.detach(), masks).item()
                row = {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": float(loss.detach().item()),
                    "bce": float(bce.detach().item()),
                    "dice": float(dice.detach().item()),
                    "train_iou": float(iou),
                    "lr": optimizer.param_groups[0]["lr"],
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                rows.append(row)
                if wandb_run is not None:
                    wandb_run.log(row, step=global_step)
                progress.set_postfix(loss=f"{row['loss']:.4f}", iou=f"{row['train_iou']:.4f}")
                append_csv(rows, log_csv)

        epoch_path = output_dir / f"{args.model.replace('-', '_')}_dis_tr_epoch{epoch}_{args.trainable}.pt"
        save_checkpoint(model, optimizer, args, epoch, global_step, epoch_path, source_checkpoint)
        epoch_sec = time.time() - epoch_start
        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "epoch_sec": epoch_sec, "checkpoint": str(epoch_path)}, step=global_step)
        print(f"epoch {epoch} done in {epoch_sec:.1f}s; saved {epoch_path}")

    save_checkpoint(model, optimizer, args, args.epochs, global_step, final_path, source_checkpoint)
    if torch.cuda.is_available():
        print(f"peak_memory_allocated_mb={torch.cuda.max_memory_allocated() / 1024**2:.1f}")
        print(f"peak_memory_reserved_mb={torch.cuda.max_memory_reserved() / 1024**2:.1f}")
    print(f"Final checkpoint: {final_path}")
    print(f"Train log CSV: {log_csv}")
    if wandb_run is not None:
        wandb_run.summary["final_checkpoint"] = str(final_path)
        wandb_run.summary["train_log_csv"] = str(log_csv)
        wandb_run.finish()
    return final_path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientViT-SAM on DIS5K-TR before HQ44K evaluation.")
    parser.add_argument("--model", default="efficientvit-sam-xl1")
    parser.add_argument("--input-checkpoint", default=None, help="Path to pretrained EfficientViT-SAM checkpoint. Default: efficientvit/assets/checkpoints/efficientvit_sam/<model>.pt")
    parser.add_argument("--no-pretrained", action="store_true", help="Debug only: initialize randomly instead of loading checkpoint.")
    parser.add_argument("--output-dir", default=str(ROOT / "benchmark_results/efficientvit_sam_finetune"))
    parser.add_argument("--output-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Debug limit. Omit for full DIS5K-TR.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--trainable", choices=["decoder", "mask_decoder", "all"], default="decoder")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--wandb", action="store_true", help="Log loss/IoU to Weights & Biases.")
    parser.add_argument("--wandb-project", default="efficientvit-sam-dis-tr")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
