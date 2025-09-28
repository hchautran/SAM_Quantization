# PTQ4SAM: Quantization Example

This repository provides a step-by-step guide to setting up and running the `seginw_engine_quan.py` file for segmentation-based tasks with quantization applied. Follow the instructions below to get started.

---

## Table of Contents

1. [Repository Cloning](#repository-cloning)
2. [Installation](#installation)
   - [Install MMCV](#install-mmcv)
   - [Compile CUDA Operators](#compile-cuda-operators)
   - [Install mmdet](#install-mmdet)
3. [Dataset Setup](#dataset-setup)
4. [Fix for Parallel Processing](#fix-for-parallel-processing)
5. [Visualization Settings](#visualization-settings)
6. [Running the Quantization Engine](#running-the-quantization-engine)

---

## Repository Cloning



git clone https://github.com/chengtao-lv/PTQ4SAM.git


---

## Installation

### Install MMCV
 mmcv must be lower than 2.0

+ pip install -U openmim
+ mim install "mmcv-full<2.0.0"

### Compile CUDA Operators


+ cd projects/instance_segment_anything/ops
+ python setup.py build install
+ cd ../../..

### Install mmdet



+ cd mmdetection/
+ python3 setup.py build develop
+ cd ..

---

## Dataset Setup

You can download the COCO dataset via Kaggle using the following snippet:


python
import kagglehub

Download latest version
path = kagglehub.dataset_download("awsaf49/coco-2017-dataset")

print("Path to dataset files:", path)

--coco
+  annotations
+  train2017
+  val2017
+  test2017

## Fix for Parallel Processing

A small fix is required to enable parallel processing. Locate the file:


/home/xc/anaconda3/envs/seg/lib/python3.10/site-packages/mmcv/parallel/distributed.py

At line 160, you will see:



+ module_to_run = self._replicated_tensor_module if self._use_replicated_tensor_module else self.module

Replace it with:


+ module_to_run = self.module

This will resolve the parallel processing issue.

---

## Visualization Settings

Open the file `quant/configmmdet/yolo_l-sam-vit-l.py`, then verify or modify the following parameters:


python
show_image = 10
result_coco_path = "/result/path"

- **show_image**: Controls how many images to display.
- **result_coco_path**: Specifies where to save the resulting images.

---

## Running the Quantization Engine


+ bash scripts/eval_seginw.sh




