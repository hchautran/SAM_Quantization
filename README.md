---
license: apache-2.0
---
# SAM Quantization 


## Installation
```
conda create -n sam_quant  python=3.10
pip install -r requirements.txt
cd sam-hq
pip install -e .
cd seginw 
pip install -e GroundDino/
cd ../../
```


## Evaluation
## SAM
### Preparation

``` 
mkdir data
mkdir pretrained_checkpoint
```

## 1. Data Preparation

HQSeg-44K can be downloaded from [hugging face link](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/data)

### Expected dataset structure for HQSeg-44K

```
data
|____DIS5K
|____cascade_psp
| |____DUTS-TE
| |____DUTS-TR
| |____ecssd
| |____fss_all
| |____MSRA_10K
|____thin_object_detection
| |____COIFT
| |____HRSOD
| |____ThinObject5K

```

## 2. Init Checkpoint
Init checkpoint can be downloaded from [hugging face link](https://huggingface.co/sam-hq-team/sam-hq-training/tree/main/pretrained_checkpoint)

### Expected checkpoint

```
pretrained_checkpoint
|____sam_vit_b_maskdecoder.pth
|____sam_vit_b.pth
|____sam_vit_l_maskdecoder.pth
|____sam_vit_l.pth
|____sam_vit_h_maskdecoder.pth
|____sam_vit_h.pth

```

## HQ-44k

### Run 


## Seginw 
### Prepare Data
### Run 

## SAM2
Comming soon !