#!/bin/bash
# Master orchestrator for tasks/tasks.md §3 experiments.
# Runs SAM HQ44K and PE ImageNet sweeps sequentially. Continues past
# failures (playbook rule 8) and writes step markers to status.txt.

cd /media/volume/Chau/SAM_Quantization || exit 99
PY=/media/volume/Chau/miniconda3/envs/sam/bin/python
LOG=.outputs/run.log
STATUS=.outputs/status.txt

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
mark() {
    msg="[$(ts)] $*"
    echo "$msg" >> "$STATUS"
    echo "$msg" >> "$LOG"
}

mark "=== ORCHESTRATOR KICKOFF (pid=$$) ==="
mark "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)"

# ---------------- 3.1 SAM baseline ----------------
mark "STEP 3.1 START — SAM baseline (algo=none)"
$PY tasks/sam_hq44k/eval_hq44k.py \
    --algos none \
    --batch-sizes 8 \
    --num-samples 470 \
    --model-ckt ./ckts/sam_hq_vit_l.pth \
    --model-type vit_l \
    --output-dir .outputs/sam_hq44k/baseline \
    --no-wandb \
    >> "$LOG" 2>&1
mark "STEP 3.1 END (rc=$?)  csvs=$(ls .outputs/sam_hq44k/baseline/*.csv 2>/dev/null | wc -l)"

# ---------------- 3.2 SAM attn-only ----------------
mark "STEP 3.2 START — SAM attn-only (--no-mlp-merge)"
$PY tasks/sam_hq44k/eval_hq44k.py \
    --algos sparsesam tome gradtome \
    --ratios 0.75 0.6 0.5 0.4 0.3 0.25 \
    --batch-sizes 8 \
    --num-samples 470 \
    --model-ckt ./ckts/sam_hq_vit_l.pth \
    --model-type vit_l \
    --no-mlp-merge \
    --output-dir .outputs/sam_hq44k/attn_only \
    --no-wandb \
    >> "$LOG" 2>&1
mark "STEP 3.2 END (rc=$?)  csvs=$(ls .outputs/sam_hq44k/attn_only/*.csv 2>/dev/null | wc -l)"

# ---------------- 3.3 SAM attn + MLP merge ----------------
mark "STEP 3.3 START — SAM attn+MLP (--mlp-merge)"
$PY tasks/sam_hq44k/eval_hq44k.py \
    --algos sparsesam tome gradtome \
    --ratios 0.75 0.6 0.5 0.4 0.3 0.25 \
    --batch-sizes 8 \
    --num-samples 470 \
    --model-ckt ./ckts/sam_hq_vit_l.pth \
    --model-type vit_l \
    --mlp-merge \
    --output-dir .outputs/sam_hq44k/attn_plus_mlp \
    --no-wandb \
    >> "$LOG" 2>&1
mark "STEP 3.3 END (rc=$?)  csvs=$(ls .outputs/sam_hq44k/attn_plus_mlp/*.csv 2>/dev/null | wc -l)"

# ---------------- 3.4 PE baseline ----------------
mark "STEP 3.4 START — PE baseline (algorithm=none)"
$PY tasks/pe_imagenet/eval_pe_clip.py \
    --model PE-Core-L14-336 \
    --dataset imagenet1k \
    --dataset-root ./tasks/pe_imagenet/data/imagenet \
    --batch-size 128 --dtype fp16 \
    --algorithm none \
    --output-dir .outputs/pe_imagenet/baseline \
    >> "$LOG" 2>&1
mark "STEP 3.4 END (rc=$?)  csvs=$(ls .outputs/pe_imagenet/baseline/*.csv 2>/dev/null | wc -l)"

# ---------------- 3.5 PE attn-only ----------------
mark "STEP 3.5 START — PE attn-only (--no-mlp-merge)"
$PY tasks/pe_imagenet/eval_pe_clip.py \
    --model PE-Core-L14-336 \
    --dataset imagenet1k \
    --dataset-root ./tasks/pe_imagenet/data/imagenet \
    --batch-size 128 --dtype fp16 \
    --algorithm sparsesam_partial tome_partial gradtome_partial \
    --ratio 0.75 0.6 0.5 0.4 0.3 0.25 \
    --no-mlp-merge \
    --output-dir .outputs/pe_imagenet/attn_only \
    >> "$LOG" 2>&1
mark "STEP 3.5 END (rc=$?)  csvs=$(ls .outputs/pe_imagenet/attn_only/*.csv 2>/dev/null | wc -l)"

# ---------------- 3.6 PE attn + MLP ----------------
mark "STEP 3.6 START — PE attn+MLP (--mlp-merge)"
$PY tasks/pe_imagenet/eval_pe_clip.py \
    --model PE-Core-L14-336 \
    --dataset imagenet1k \
    --dataset-root ./tasks/pe_imagenet/data/imagenet \
    --batch-size 128 --dtype fp16 \
    --algorithm sparsesam_partial tome_partial gradtome_partial \
    --ratio 0.75 0.6 0.5 0.4 0.3 0.25 \
    --mlp-merge \
    --output-dir .outputs/pe_imagenet/attn_plus_mlp \
    >> "$LOG" 2>&1
mark "STEP 3.6 END (rc=$?)  csvs=$(ls .outputs/pe_imagenet/attn_plus_mlp/*.csv 2>/dev/null | wc -l)"

mark "=== ORCHESTRATOR DONE ==="
mark "All CSVs:"
find .outputs -name '*.csv' | sort >> "$STATUS"
