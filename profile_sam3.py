#!/usr/bin/env python3
"""Profile SAM3 image + grounding components.

This script measures the cost of SAM3 image encoding and grounding and
breaks down the time spent in key model stages:
  - image backbone / preprocessing
  - text encoding
  - prompt encoding
  - encoder
  - decoder
  - segmentation head

Usage:
  python profile_sam3.py --resolution 800
  python profile_sam3.py --checkpoint ./sam3_ckts/sam3.pt --resolution 800

If no checkpoint is provided, the script downloads HF SAM3 weights.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# Add sam3 to path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'sam3'))
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.data_misc import FindStage


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
    print(f"{'component':<32} {'mean(ms)':>10} {'std(ms)':>10}")
    print('-' * 54)
    total = 0.0
    for key, timer in timers.items():
        mean = timer.mean()
        std = timer.std()
        total += mean
        print(f"{key:<32} {mean:10.3f} {std:10.3f}")
    print('-' * 54)
    print(f"{'total':<32} {total:10.3f}")


def run_profile(
    checkpoint: Optional[str],
    resolution: int,
    n_warmup: int,
    n_runs: int,
    enable_segmentation: bool,
    device: str,
    prompt_texts: List[str],
):
    assert torch.cuda.is_available(), 'CUDA is required for SAM3 profiling'

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print('Building SAM3 image model ...')
    model = build_sam3_image_model(
        checkpoint_path=checkpoint,
        load_from_HF=True,
        enable_segmentation=enable_segmentation,
    )
    model.eval()
    model = model.to(device)

    processor = Sam3Processor(model, resolution=resolution, device=device, confidence_threshold=0.0)

    timers = {
        'Image Encoder': CUDATimer(),
        'Text Encoder': CUDATimer(),
        'Prompt Encoder': CUDATimer(),
        'Fusion Encoder': CUDATimer(),
        'DETR Decoder': CUDATimer(),
        'Presence Head': CUDATimer(),
        'Mask/Semantic Heads': CUDATimer(),
        'SAM3 Detector Total': CUDATimer(),
    }

    # Wrap the methods we want to time
    orig_forward_text = wrap_method(model.backbone, 'forward_text', timers['Text Encoder'])
    orig_encode_prompt = wrap_method(model, '_encode_prompt', timers['Prompt Encoder'])
    orig_run_encoder = wrap_method(model, '_run_encoder', timers['Fusion Encoder'])
    orig_run_decoder = wrap_method(model, '_run_decoder', timers['DETR Decoder'])
    orig_dot_prod_scoring = wrap_method(model.dot_prod_scoring, '__call__', timers['Presence Head'])
    orig_run_segmentation_heads = wrap_method(model, '_run_segmentation_heads', timers['Mask/Semantic Heads'])

    # Warm up one synthetic image for consistent timings
    image = Image.fromarray(np.random.randint(0, 255, size=(resolution, resolution, 3), dtype=np.uint8))
    dummy_texts = prompt_texts
    num_prompts = len(dummy_texts)
    state = None

    # Pre-encode prompts once to simulate evaluation flow
    with torch.no_grad():
        text_features = model.backbone.forward_text(dummy_texts, device=device)

    find_stage = build_find_stage(batch_size=num_prompts, device=device)
    geo_prompt = model._get_dummy_prompt(num_prompts=num_prompts)

    def run_once():
        nonlocal state
        with torch.no_grad():
            # Image encoding + preprocessing
            timers['Image Encoder'].start()
            state = processor.set_image(image)
            timers['Image Encoder'].stop()

            state['backbone_out'].update(text_features)

            timers['SAM3 Detector Total'].start()
            _ = model.forward_grounding(
                backbone_out=state['backbone_out'],
                find_input=find_stage,
                find_target=None,
                geometric_prompt=geo_prompt,
            )
            timers['SAM3 Detector Total'].stop()

    print(f'Warming up ({n_warmup} runs) ...')
    for _ in range(n_warmup):
        run_once()

    for timer in timers.values():
        timer.reset()

    print(f'Profiling ({n_runs} runs) ...')
    for _ in range(n_runs):
        run_once()

    # Restore original methods
    setattr(model.backbone, 'forward_text', orig_forward_text)
    setattr(model, '_encode_prompt', orig_encode_prompt)
    setattr(model, '_run_encoder', orig_run_encoder)
    setattr(model, '_run_decoder', orig_run_decoder)
    setattr(model.dot_prod_scoring, '__call__', orig_dot_prod_scoring)
    setattr(model, '_run_segmentation_heads', orig_run_segmentation_heads)

    print_report('SAM3 profiling results', timers)
    if not enable_segmentation:
        print('\nNote: segmentation head disabled for this run.')


def parse_args():
    p = argparse.ArgumentParser(description='Profile SAM3 image + grounding components')
    p.add_argument('--checkpoint', default=None, help='Optional SAM3 checkpoint path; if omitted, HF SAM3 weights are downloaded')
    p.add_argument('--resolution', type=int, default=1008, help='Image resolution for preprocessing (1008 is native SAM3 input size)')
    p.add_argument('--n-warmup', type=int, default=5, help='Warmup iterations')
    p.add_argument('--n-runs', type=int, default=20, help='Measured iterations')
    p.add_argument('--disable-segmentation', action='store_true', help='Disable segmentation head to measure bbox-only cost')
    p.add_argument('--device', default='cuda', help='Device to run on')
    p.add_argument('--prompts', nargs='+', default=['person', 'dog', 'car'], help='Text prompts to encode for grounding')
    return p.parse_args()


def main():
    args = parse_args()
    run_profile(
        checkpoint=args.checkpoint,
        resolution=args.resolution,
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        enable_segmentation=not args.disable_segmentation,
        device=args.device,
        prompt_texts=args.prompts,
    )


if __name__ == '__main__':
    main()
