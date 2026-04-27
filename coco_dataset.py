"""COCO instance-segmentation dataset for box-prompted SAM evaluation.

Each item is one image with all of its annotations grouped together:
    {
        "image":      (3, 1024, 1024) uint8 tensor — resized image
        "ori_im":     (H, W, 3) uint8 array        — original image
        "ori_size":   (H, W)                       — original size
        "boxes_ori":  (N, 4) float32               — GT boxes in original coords (xyxy)
        "masks_ori":  (N, H, W) uint8              — GT instance masks at original res
        "image_id":   int
        "ann_ids":    list[int]
        "cat_ids":    list[int]
    }

This format lets the encoder run once per image and the decoder loop over
all N instances — which matches the cost structure of SAM evaluation.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


class CocoInstanceDataset(Dataset):
    """
    box_source:
      "gt"         — use GT bboxes as prompts (per-instance mIoU/B-IoU is meaningful)
      "detections" — use detector bboxes from `detections_file` (COCO-results JSON);
                     evaluation should be COCO segm AP (no 1:1 GT match)
    """
    def __init__(self, image_dir: str, ann_file: str, min_box_size: int = 4,
                 max_instances_per_image: int | None = None,
                 detections_file: str | None = None,
                 score_threshold: float = 0.0):
        import json as _json
        self.image_dir = image_dir
        self.coco = COCO(ann_file)
        self.min_box_size = min_box_size
        self.max_instances = max_instances_per_image
        self.box_source = "detections" if detections_file else "gt"

        self.dets_by_img: dict[int, list] = {}
        if detections_file is not None:
            with open(detections_file, "r") as f:
                preds = _json.load(f)
            for p in preds:
                if p.get("score", 1.0) < score_threshold:
                    continue
                self.dets_by_img.setdefault(int(p["image_id"]), []).append(p)
            for k in self.dets_by_img:
                self.dets_by_img[k].sort(key=lambda x: -x.get("score", 0.0))

        all_img_ids = sorted(self.coco.getImgIds())
        if self.box_source == "detections":
            self.img_ids = [i for i in all_img_ids if i in self.dets_by_img]
        else:
            self.img_ids = [
                i for i in all_img_ids
                if len(self.coco.getAnnIds(imgIds=i, iscrowd=False)) > 0
            ]
        self._resize = T.Resize((1024, 1024))

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        im_path = os.path.join(self.image_dir, info["file_name"])
        im = io.imread(im_path)
        if im.ndim == 2:
            im = np.stack([im] * 3, axis=-1)
        elif im.shape[2] == 4:
            im = im[:, :, :3]
        H, W = im.shape[:2]

        boxes, masks, kept_ann_ids, cat_ids, scores = [], [], [], [], []
        if self.box_source == "detections":
            for d in self.dets_by_img.get(img_id, []):
                x, y, w, h = d["bbox"]
                if w < self.min_box_size or h < self.min_box_size:
                    continue
                boxes.append([x, y, x + w, y + h])
                masks.append(np.zeros((H, W), dtype=np.uint8))  # placeholder; not used for AP
                kept_ann_ids.append(-1)
                cat_ids.append(int(d["category_id"]))
                scores.append(float(d.get("score", 1.0)))
        else:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            for a in self.coco.loadAnns(ann_ids):
                x, y, w, h = a["bbox"]
                if w < self.min_box_size or h < self.min_box_size:
                    continue
                m = self.coco.annToMask(a)
                if m.sum() == 0:
                    continue
                boxes.append([x, y, x + w, y + h])
                masks.append(m.astype(np.uint8))
                kept_ann_ids.append(a["id"])
                cat_ids.append(a["category_id"])
                scores.append(1.0)

        if self.max_instances is not None and len(boxes) > self.max_instances:
            boxes  = boxes[:self.max_instances]
            masks  = masks[:self.max_instances]
            kept_ann_ids = kept_ann_ids[:self.max_instances]
            cat_ids = cat_ids[:self.max_instances]
            scores  = scores[:self.max_instances]

        if len(boxes) == 0:
            boxes = [[0, 0, W, H]]
            masks = [np.zeros((H, W), dtype=np.uint8)]
            kept_ann_ids = [-1]
            cat_ids = [-1]
            scores  = [0.0]

        im_resized = self._resize(torch.from_numpy(im).permute(2, 0, 1))

        return {
            "image":     im_resized,
            "ori_im":    im,
            "ori_size":  (H, W),
            "boxes_ori": torch.tensor(boxes, dtype=torch.float32),
            "masks_ori": torch.from_numpy(np.stack(masks, axis=0)),
            "image_id":  int(img_id),
            "ann_ids":   kept_ann_ids,
            "cat_ids":   cat_ids,
            "scores":    scores,
            "box_source": self.box_source,
        }


def coco_collate_fn(batch):
    """Variable-length annotation lists per image — keep as Python lists."""
    out = {}
    for k in batch[0].keys():
        if k == "image":
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out
