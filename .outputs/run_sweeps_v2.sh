#!/bin/bash
# Resumed orchestrator after fixing tome/merge clamp.
# Runs §3.2, §3.3, §3.4, §3.5, §3.6 per tasks/tasks.md.
# §3.2 and §3.3 are split per-(algo, ratio) so a single bad config
# doesn't lose the entire CSV (eval_hq44k writes CSV only at end).
#
# §3.1 baseline already done (.outputs/sam_hq44k/baseline/tome_benchmark_20260502_153219.csv).

cd /media/volume/Chau/SAM_Quantization || exit 99
PY=/media/volume/Chau/miniconda3/envs/sam/bin/python
LOG=.outputs/run.log
STATUS=.outputs/status.txt

ALGOS_SAM="sparsesam tome gradtome"
ALGOS_PE="sparsesam_partial tome_partial gradtome_partial"
RATIOS="0.75 0.6 0.5 0.4 0.3 0.25"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
mark() {
    msg="[$(ts)] $*"
    echo "$msg" >> "$STATUS"
    echo "$msg" >> "$LOG"
}

mark "=== ORCHESTRATOR-v2 KICKOFF (pid=$$) ==="
mark "GPU: $(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader)"

run_sam_cell() {
    # $1=algo, $2=ratio, $3=mode (no-mlp|mlp), $4=outdir
    local algo=$1 ratio=$2 mode=$3 outdir=$4
    local flag=""
    [ "$mode" = "no-mlp" ] && flag="--no-mlp-merge"
    [ "$mode" = "mlp" ] && flag="--mlp-merge"
    mark "    SAM cell START algo=$algo ratio=$ratio mode=$mode"
    $PY tasks/sam_hq44k/eval_hq44k.py \
        --algos "$algo" --ratios "$ratio" \
        --batch-sizes 8 --num-samples 470 \
        --model-ckt ./ckts/sam_hq_vit_l.pth --model-type vit_l \
        $flag --no-plot \
        --output-dir "$outdir" --no-wandb \
        >> "$LOG" 2>&1
    local rc=$?
    mark "    SAM cell END   algo=$algo ratio=$ratio mode=$mode rc=$rc"
}

# ---------------- 3.2 SAM attn-only ----------------
mark "STEP 3.2 START — SAM attn-only (per-cell)"
for algo in $ALGOS_SAM; do
    for r in $RATIOS; do
        run_sam_cell "$algo" "$r" "no-mlp" ".outputs/sam_hq44k/attn_only"
    done
done
mark "STEP 3.2 END   csvs=$(ls .outputs/sam_hq44k/attn_only/*.csv 2>/dev/null | wc -l)"

# ---------------- 3.3 SAM attn + MLP ----------------
mark "STEP 3.3 START — SAM attn+MLP (per-cell)"
for algo in $ALGOS_SAM; do
    for r in $RATIOS; do
        run_sam_cell "$algo" "$r" "mlp" ".outputs/sam_hq44k/attn_plus_mlp"
    done
done
mark "STEP 3.3 END   csvs=$(ls .outputs/sam_hq44k/attn_plus_mlp/*.csv 2>/dev/null | wc -l)"

# ---------------- 3.4 PE baseline ----------------
mark "STEP 3.4 START — PE baseline"
$PY tasks/pe_imagenet/eval_pe_clip.py \
    --model PE-Core-L14-336 \
    --dataset imagenet1k \
    --dataset-root ./tasks/pe_imagenet/data/imagenet \
    --batch-size 128 --dtype fp16 \
    --algorithm none \
    --output-dir .outputs/pe_imagenet/baseline \
    >> "$LOG" 2>&1
mark "STEP 3.4 END (rc=$?) csvs=$(ls .outputs/pe_imagenet/baseline/*.csv 2>/dev/null | wc -l)"

# ---------------- 3.5 PE attn-only (per algo to limit blast radius) ----------------
mark "STEP 3.5 START — PE attn-only (per-algo)"
for algo in $ALGOS_PE; do
    mark "    PE attn-only START algo=$algo"
    $PY tasks/pe_imagenet/eval_pe_clip.py \
        --model PE-Core-L14-336 \
        --dataset imagenet1k \
        --dataset-root ./tasks/pe_imagenet/data/imagenet \
        --batch-size 128 --dtype fp16 \
        --algorithm "$algo" \
        --ratio 0.75 0.6 0.5 0.4 0.3 0.25 \
        --no-mlp-merge \
        --output-dir .outputs/pe_imagenet/attn_only \
        >> "$LOG" 2>&1
    mark "    PE attn-only END   algo=$algo rc=$?"
done
mark "STEP 3.5 END   csvs=$(ls .outputs/pe_imagenet/attn_only/*.csv 2>/dev/null | wc -l)"

# ---------------- 3.6 PE attn + MLP ----------------
mark "STEP 3.6 START — PE attn+MLP (per-algo)"
for algo in $ALGOS_PE; do
    mark "    PE attn+MLP START algo=$algo"
    $PY tasks/pe_imagenet/eval_pe_clip.py \
        --model PE-Core-L14-336 \
        --dataset imagenet1k \
        --dataset-root ./tasks/pe_imagenet/data/imagenet \
        --batch-size 128 --dtype fp16 \
        --algorithm "$algo" \
        --ratio 0.75 0.6 0.5 0.4 0.3 0.25 \
        --mlp-merge \
        --output-dir .outputs/pe_imagenet/attn_plus_mlp \
        >> "$LOG" 2>&1
    mark "    PE attn+MLP END   algo=$algo rc=$?"
done
mark "STEP 3.6 END   csvs=$(ls .outputs/pe_imagenet/attn_plus_mlp/*.csv 2>/dev/null | wc -l)"

mark "=== ORCHESTRATOR-v2 DONE ==="
mark "All CSVs:"
find .outputs -name '*.csv' | sort >> "$STATUS"
