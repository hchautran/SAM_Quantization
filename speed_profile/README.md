# Encoder speed benchmark

Measures image-encoder latency on synthetic 1024×1024 noise and reports
speedup against SAM-B. No dataset needed — nothing here touches COCO or HQ44K.

| model | what is timed |
|---|---|
| `sam_b` | SAM ViT-B `image_encoder` — the baseline every speedup divides by |
| `sparsesam_b` | same encoder, SparseSAM patch, sparse attention + MLP routing, keep-ratio 0.25 (75 % sparsity) |
| `fastsam` | **full forward** — FastSAM is YOLOv8-seg and has no encoder/decoder split |
| `mobilesam` | TinyViT `image_encoder` |
| `efficientsam` | EfficientSAM-vitt `image_encoder` |
| `efficientvit_sam` | EfficientViT-SAM-xl1 `image_encoder` |

## Setup on a fresh machine

```bash
git clone <this repo> && cd SAM_Quantization

# the four upstream repos, next to speed_profile/
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
git clone https://github.com/ChaoningZhang/MobileSAM.git
git clone https://github.com/yformer/EfficientSAM.git
git clone https://github.com/mit-han-lab/efficientvit.git
# plus sam-hq/ and PiToMe/, which SparseSAM needs

conda env create -f speed_profile/environment.yml
conda activate speed_profile
```

Checkpoints (paths are the defaults; override with `--<model>-checkpoint`):

| model | default path |
|---|---|
| `sam_b`, `sparsesam_b` | `ckts/sam_hq_vit_b.pth` |
| `fastsam` | `FastSAM/.weights/FastSAM-x.pt` |
| `mobilesam` | `MobileSAM/weights/mobile_sam.pt` |
| `efficientsam` | `EfficientSAM/weights/efficient_sam_vitt.pt` |
| `efficientvit_sam` | `ckts/efficientvit_sam_xl1.pt` |

## Run

```bash
python speed_profile/bench_encoder_speed.py --batch-sizes 1 2 4 8
```

That is the whole thing: 100 synthetic images per (model, batch size), 10 warmup
iterations, CUDA-event timing with a synchronize per iteration. Prints a table
and writes `speed_profile/encoder_speed_<gpu>_<timestamp>.csv`.

Useful flags:

```bash
--models sam_b sparsesam_b        # subset
--ratio 0.25                      # SparseSAM keep ratio (0.25 = 75 % sparsity)
--no-mlp-merge                    # SparseSAM: sparse attention only, full MLP
--num-samples 100 --warmup 10
--dtype fp16                      # fp16 is required for SparseSAM's cute kernel
--root /path/to/repos             # if the upstream repos live elsewhere
```

A model whose repo or checkpoint is missing is skipped with a printed reason;
the rest of the table still runs.

## Reading the output

`speedup_vs_sam_b` = SAM-B ms/image ÷ model ms/image, computed per batch size.
Everything runs in fp16 so the ratio is not inflated by the baseline sitting on
non-tensor-core paths.

Comparing GPUs: run the same command on each machine and diff the CSVs — `gpu`,
`torch`, `host` and `dtype` are recorded in every row.
