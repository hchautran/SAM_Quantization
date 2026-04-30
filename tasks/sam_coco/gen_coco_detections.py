#!/usr/bin/env python3
"""Generate COCO val2017 box detections using a pretrained HF detector.

Output: COCO-results JSON (list of {"image_id", "category_id", "bbox", "score"})
that plugs into eval_coco.py via --detections.

The HQ-SAM paper uses FocalNet-DINO (~64 AP). Strong alternatives that work
out-of-the-box from `transformers`:

  jozhang97/deta-swin-large           ~62 box AP   (default — best/easy)
  facebook/detr-resnet-101            ~44 box AP   (light)
  microsoft/conditional-detr-resnet-50 ~43 box AP

Usage:
    python gen_coco_detections.py \
        --coco-root ./data/coco --split val2017 \
        --model jozhang97/deta-swin-large \
        --score-threshold 0.3 \
        --out ./focalnet_dino_coco_val2017.json
"""

import os
import json
import argparse
from tqdm import tqdm
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO


class CocoImages(Dataset):
    def __init__(self, image_dir, ann_file):
        self.image_dir = image_dir
        self.coco = COCO(ann_file)
        self.img_ids = sorted(self.coco.getImgIds())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.image_dir, info["file_name"])
        img = Image.open(path).convert("RGB")
        return {"image_id": img_id, "image": img,
                "size": (info["height"], info["width"])}


def collate(batch):
    return {
        "image_id": [b["image_id"] for b in batch],
        "image":    [b["image"]    for b in batch],
        "size":     [b["size"]     for b in batch],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", default="./data/coco")
    p.add_argument("--split", default="val2017")
    p.add_argument("--model", default="jozhang97/deta-swin-large",
                   help="HF model name. Must be COCO-pretrained.")
    p.add_argument("--out", default="./focalnet_dino_coco_val2017.json")
    p.add_argument("--score-threshold", type=float, default=0.3)
    p.add_argument("--top-k", type=int, default=100,
                   help="Keep top-K detections per image after threshold.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-images", type=int, default=None)
    args = p.parse_args()

    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    image_dir = os.path.join(args.coco_root, args.split)
    ann_file  = os.path.join(args.coco_root, "annotations",
                             f"instances_{args.split}.json")

    print(f"Loading detector {args.model} ...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model     = AutoModelForObjectDetection.from_pretrained(args.model).cuda().eval().half()

    # HF model id2label uses 0..N indices; we need actual COCO category_ids.
    # All these models are COCO-pretrained; their labels follow the COCO 80-class
    # ordering. Map back to original COCO ids (1..90 with gaps).
    coco_91 = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,
               27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,
               51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,
               76,77,78,79,80,81,82,84,85,86,87,88,89,90]
    n_labels = len(model.config.id2label)
    if n_labels == 80:
        idx_to_coco = {i: coco_91[i] for i in range(80)}
    elif n_labels >= 91:
        idx_to_coco = {i: i for i in range(n_labels)}  # already COCO ids
    else:
        raise RuntimeError(f"Unexpected #labels={n_labels} for {args.model}")

    ds = CocoImages(image_dir, ann_file)
    if args.num_images:
        ds.img_ids = ds.img_ids[:args.num_images]
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=2,
                        collate_fn=collate)

    predictions = []
    for batch in tqdm(loader, desc="detect"):
        inputs = processor(images=batch["image"], return_tensors="pt")
        inputs = {k: v.cuda().half() if v.dtype == torch.float32 else v.cuda()
                  for k, v in inputs.items()}
        target_sizes = torch.tensor(batch["size"], device="cuda")

        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes,
            threshold=args.score_threshold,
        )

        for img_id, r in zip(batch["image_id"], results):
            scores = r["scores"].float().cpu()
            labels = r["labels"].cpu()
            boxes  = r["boxes"].float().cpu()  # xyxy in original coords

            if len(scores) > args.top_k:
                top = scores.topk(args.top_k).indices
                scores, labels, boxes = scores[top], labels[top], boxes[top]

            for s, l, b in zip(scores, labels, boxes):
                x1, y1, x2, y2 = b.tolist()
                predictions.append({
                    "image_id":    int(img_id),
                    "category_id": int(idx_to_coco[int(l)]),
                    "bbox":        [x1, y1, x2 - x1, y2 - y1],  # COCO xywh
                    "score":       float(s),
                })

    print(f"Total detections: {len(predictions)}")
    with open(args.out, "w") as f:
        json.dump(predictions, f)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
