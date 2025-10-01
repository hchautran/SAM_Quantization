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

## Quantization Methods

### Weight Quantization (RTN, UP, DOWN, RANDOM)

#### Basic Configuration
To quantize weights using RTN (Round to Nearest):
1. Set the config file to `rtn.yaml`
2. Configure the `up_down_RTN` parameter:
   - Set to `none` for standard RTN (full quantization)
   - Other options: `up`, `down`, `random`, or percentage-based RTN

#### Encoder Quantization
To enable RTN quantization for the encoder:
1. Open `hq44k_engine_quan.py`
2. Locate the `if self.quant_rtn:` block
3. Uncomment the encoder quantization section

#### Percentage-based RTN
- Default: 75%
- To modify: Edit the `quantize_weight_per_channel_random_round_up_down_absmax` function


In hq44k_engine_quan.py set:

if name == "main":
model_args = OmegaConf.load('quant/config/hq44k/rtn.yaml') # Note: Should be rtn.yaml for RTN
args = get_args_parser()
#engine = Hq44kSamEngine(Hq44kInferenceStrategy(model_args))
#engine.evaluate(args,model_args)
#exit()

---

### Activation Quantization (High/Low Density)

#### Basic Setup
1. Set the config file to `low_high.yaml` in `hq44k_engine_quan.py`
2. Run the script
3. **Important**: Add `exit()` after running to prevent further execution

#### Configuration Options

##### Density Selection
In the config file, set the `low_high_density` parameter:
- `high` - for high density quantization
- `low` - for low density quantization

##### Token Quantization Percentage
To modify the percentage of tokens to quantize:
1. Open `per_tensor_channel_group.py`
2. Locate the `quantize_activation_low_high_density_activation` function
3. Modify the `percent` variable (default: 50)

In hq44k_engine_quan.py set:

if name == "main":
model_args = OmegaConf.load('quant/config/hq44k/low_high.yaml')
args = get_args_parser()
engine = Hq44kSamEngine(Hq44kInferenceStrategy(model_args))
engine.evaluate(args, model_args)
exit() # Important: Exit after evaluation