#!/usr/bin/env python3
"""Profile SAM3 inference latency per component on COCO.

Walks a COCO val split and times each model stage during the actual
open-vocabulary detection pipeline used by ``eval_coco_sam3.py``:
  - Image preprocessing + image backbone (``backbone.forward_image``)
  - Text encoder (``backbone.forward_text``) — measured once on COCO categories
  - Prompt encoder (``model._encode_prompt``)
  - Fusion encoder (``model._run_encoder``)
  - DETR decoder (``model._run_decoder``)
  - Presence/class scoring (``model.dot_prod_scoring``)
  - Mask + semantic heads (``model._run_segmentation_heads``)
  - Total grounding pass (``model.forward_grounding``)

Usage:
  python profile_sam3.py --coco-root ./data/coco --num-images 20 --resolution 800
  python profile_sam3.py --coco-root ./data/coco --num-images 20 --disable-segmentation
  python profile_sam3.py --synthetic --resolution 800   # fall back to random images
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add sam3 to path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'sam3'))
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.data_misc import FindStage

from pycocotools.coco import COCO


class CUDATimer:
    def __init__(self):
        self.samples: List[float] = []
        self._start = None

    def start(self):
        self._start = torch.cuda.Event(enable_timing=True)
        self._start.record()

    def stop(self):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        torch.cuda.synchronize()
        self.samples.append(self._start.elapsed_time(end))

    def reset(self):
        self.samples.clear()

    def mean(self) -> float:
        return float(np.mean(self.samples)) if self.samples else 0.0

    def std(self) -> float:
        return float(np.std(self.samples)) if self.samples else 0.0

    def count(self) -> int:
        return len(self.samples)


def wrap_method(obj, method_name: str, timer: CUDATimer):
    orig = getattr(obj, method_name)

    def wrapper(*args, **kwargs):
        timer.start()
        try:
            return orig(*args, **kwargs)
        finally:
            timer.stop()

    setattr(obj, method_name, wrapper)
    return orig


def build_find_stage(batch_size: int, device: str):
    return FindStage(
        img_ids=torch.zeros(batch_size, device=device, dtype=torch.long),
        text_ids=torch.arange(batch_size, device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )


def print_report(name: str, timers: Dict[str, CUDATimer]):
    print(f"\n{name}")
    print(f"{'component':<32} {'calls':>8} {'mean(ms)':>10} {'std(ms)':>10} {'sum(ms)':>10}")
    print('-' * 74)
    for key, timer in timers.items():
        mean = timer.mean()
        std = timer.std()
        total = mean * timer.count()
        print(f"{key:<32} {timer.count():>8d} {mean:10.3f} {std:10.3f} {total:10.3f}")
    print('-' * 74)


# ──────────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────────

def load_coco_inputs(coco_root: str, split: str, num_images: int):
    ann_file = os.path.join(coco_root, "annotations", f"instances_{split}.json")
    image_dir = os.path.join(coco_root, split)
    coco = COCO(ann_file)
    cats = coco.loadCats(coco.getCatIds())
    cat_names = [c["name"] for c in cats]
    img_ids = sorted(coco.getImgIds())[:num_images]
    image_paths = [os.path.join(image_dir, coco.loadImgs(i)[0]["file_name"]) for i in img_ids]
    return image_paths, cat_names


def run_profile(args):
    assert torch.cuda.is_available(), 'CUDA is required for SAM3 profiling'

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.bf16:
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

    device = args.device

    print('Building SAM3 image model ...')
    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        load_from_HF=(args.checkpoint is None),
        enable_segmentation=not args.disable_segmentation,
        enable_inst_interactivity=False,
    )
    model.eval().to(device)

    processor = Sam3Processor(model, resolution=args.resolution,
                              device=device, confidence_threshold=0.0)
    processor.chunk_size = args.chunk_size

    # Load real COCO categories + images, or fall back to synthetic
    if args.synthetic:
        prompt_texts = args.prompts
        image_paths = None
        print(f"Synthetic mode: {len(prompt_texts)} prompts, {args.num_images} random images")
    else:
        image_paths, prompt_texts = load_coco_inputs(args.coco_root, args.split, args.num_images)
        print(f"COCO mode: {len(image_paths)} images × {len(prompt_texts)} categories "
              f"(chunk_size={args.chunk_size})")

    timers = {
        'Image Encoder (forward_image)': CUDATimer(),
        'Text Encoder (forward_text)':   CUDATimer(),
        'Prompt Encoder':                CUDATimer(),
        'Fusion Encoder':                CUDATimer(),
        'DETR Decoder':                  CUDATimer(),
        'Presence Head':                 CUDATimer(),
        'Mask/Semantic Heads':           CUDATimer(),
        'Grounding Total (per chunk)':   CUDATimer(),
        'Per-image Total':               CUDATimer(),
    }

    # Wrap model component methods
    orig_forward_image  = wrap_method(model.backbone, 'forward_image', timers['Image Encoder (forward_image)'])
    orig_forward_text   = wrap_method(model.backbone, 'forward_text',  timers['Text Encoder (forward_text)'])
    orig_encode_prompt  = wrap_method(model, '_encode_prompt',         timers['Prompt Encoder'])
    orig_run_encoder    = wrap_method(model, '_run_encoder',           timers['Fusion Encoder'])
    orig_run_decoder    = wrap_method(model, '_run_decoder',           timers['DETR Decoder'])
    orig_dot_prod       = wrap_method(model.dot_prod_scoring, '__call__', timers['Presence Head'])
    orig_run_seg_heads  = wrap_method(model, '_run_segmentation_heads', timers['Mask/Semantic Heads'])
    orig_forward_ground = wrap_method(model, 'forward_grounding',      timers['Grounding Total (per chunk)'])

    # Pre-encode all category text features once (uses Text Encoder timer once)
    with torch.no_grad():
        text_features = model.backbone.forward_text(prompt_texts, device=device)

    N = len(prompt_texts)
    chunk_size = min(args.chunk_size, N)

    def run_one_image(image: Image.Image):
        with torch.no_grad():
            timers['Per-image Total'].start()
            state = processor.set_image(image)

            for chunk_start in range(0, N, chunk_size):
                chunk_end = min(chunk_start + chunk_size, N)
                B = chunk_end - chunk_start

                state['backbone_out'].update({
                    "language_features": text_features["language_features"][:, chunk_start:chunk_end],
                    "language_mask":     text_features["language_mask"][chunk_start:chunk_end],
                })

                find_stage = build_find_stage(B, device)
                geo_prompt = model._get_dummy_prompt(num_prompts=B)

                _ = model.forward_grounding(
                    backbone_out=state['backbone_out'],
                    find_input=find_stage,
                    find_target=None,
                    geometric_prompt=geo_prompt,
                )
            timers['Per-image Total'].stop()

    def get_image(idx: int) -> Image.Image:
        if args.synthetic or image_paths is None:
            arr = np.random.randint(0, 255, (args.resolution, args.resolution, 3), dtype=np.uint8)
            return Image.fromarray(arr)
        return Image.open(image_paths[idx % len(image_paths)]).convert("RGB")

    # Warmup
    print(f'Warming up ({args.n_warmup} images) ...')
    for i in range(args.n_warmup):
        run_one_image(get_image(i))

    for t in timers.values():
        t.reset()

    # Re-time text encoder on warmed model
    with torch.no_grad():
        text_features = model.backbone.forward_text(prompt_texts, device=device)

    # Measured loop
    n_runs = len(image_paths) if (image_paths is not None) else args.num_images
    print(f'Profiling on {n_runs} images ...')
    for i in tqdm(range(n_runs)):
        run_one_image(get_image(i))

    # Restore original methods
    setattr(model.backbone, 'forward_image', orig_forward_image)
    setattr(model.backbone, 'forward_text',  orig_forward_text)
    setattr(model, '_encode_prompt',         orig_encode_prompt)
    setattr(model, '_run_encoder',           orig_run_encoder)
    setattr(model, '_run_decoder',           orig_run_decoder)
    setattr(model.dot_prod_scoring, '__call__', orig_dot_prod)
    setattr(model, '_run_segmentation_heads', orig_run_seg_heads)
    setattr(model, 'forward_grounding',      orig_forward_ground)

    print_report(f'SAM3 profiling — {"synthetic" if args.synthetic else "COCO " + args.split}', timers)
    if args.disable_segmentation:
        print('\nNote: segmentation head disabled for this run.')

    # Throughput
    total_ms = sum(t.mean() * t.count() for k, t in timers.items() if k == 'Per-image Total')
    if total_ms > 0:
        print(f"\nThroughput: {n_runs * 1000.0 / total_ms:.3f} img/s "
              f"(mean per-image latency: {timers['Per-image Total'].mean():.2f} ms)")


def parse_args():
    p = argparse.ArgumentParser(description='Profile SAM3 components during COCO inference')
    # Model
    p.add_argument('--checkpoint', default=None, help='SAM3 checkpoint path; if omitted, HF weights are downloaded')
    p.add_argument('--resolution', type=int, default=1008, help='Image resolution (1008 = native SAM3)')
    p.add_argument('--disable-segmentation', action='store_true', help='Disable segmentation heads (bbox-only timing)')
    p.add_argument('--bf16', action='store_true', help='Run under bfloat16 autocast (matches eval_coco_sam3.py)')
    p.add_argument('--device', default='cuda')
    # Data
    p.add_argument('--coco-root', default='./data/coco')
    p.add_argument('--split', default='val2017')
    p.add_argument('--num-images', type=int, default=20)
    p.add_argument('--chunk-size', type=int, default=8, help='Categories per grounding chunk')
    p.add_argument('--synthetic', action='store_true', help='Use random images + --prompts instead of COCO')
    p.add_argument('--prompts', nargs='+', default=['person', 'dog', 'car'],
                   help='Prompts for synthetic mode (ignored otherwise)')
    # Loop
    p.add_argument('--n-warmup', type=int, default=2)
    return p.parse_args()


def main():
    run_profile(parse_args())


if __name__ == '__main__':
    main()
