#!/bin/bash
# Usage: ./eval_port.sh <port>
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=$29506 quant/hq44k_engine_quan.py