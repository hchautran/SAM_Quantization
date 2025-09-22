#!/bin/bash
# Usage: ./eval_port.sh <port>
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=$29507 quant/hq44k_engine_quan.py