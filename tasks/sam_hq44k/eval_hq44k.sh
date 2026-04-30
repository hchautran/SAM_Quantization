#!/bin/bash
# SAM-HQ HQ44K throughput + mIoU sweep.
#
# Expected (relative to repo root): ./ckts/sam_hq_vit_l.pth and HQ44K-style
# datasets configured in sam_engine.py::get_default_datasets().
#
# Knobs (env-overridable):
#   ALGOS          space-separated algos (any of: tome pitome sparsesam
#                  sparsesam_pitome sparsesam_random gradtome
#                  gradtome_pitome gradtome_hilbert)
#   RATIOS         per-block keep fractions
#   BATCH_SIZES    space-separated batch sizes
#   NUM_SAMPLES    images per dataset
#   CKPT, MODEL    SAM-HQ checkpoint + arch (vit_l / vit_b / vit_h)
#   MLP_MERGE      "yes" (default) or "no". Sparsesam family only:
#                    yes → MLP runs on top floor(ratio·N) keep tokens
#                    no  → MLP on full token set; only K/V sparse attn
#                          compression remains. Use to isolate the
#                          partial-MLP shortcut from sparse attention.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

ALGOS=${ALGOS:-"sparsesam gradtome tome"}
RATIOS=${RATIOS:-"0.75 0.5 0.25"}
BATCH_SIZES=${BATCH_SIZES:-"8"}
NUM_SAMPLES=${NUM_SAMPLES:-470}
CKPT=${CKPT:-./ckts/sam_hq_vit_l.pth}
MODEL=${MODEL:-vit_l}
MLP_MERGE=${MLP_MERGE:-yes}

if [ "$MLP_MERGE" = "no" ] || [ "$MLP_MERGE" = "false" ] || [ "$MLP_MERGE" = "0" ]; then
    MLP_FLAG="--no-mlp-merge"
else
    MLP_FLAG="--mlp-merge"
fi

python "$SCRIPT_DIR/eval_hq44k.py" \
    --algos       $ALGOS \
    --ratios      $RATIOS \
    --batch-sizes $BATCH_SIZES \
    --num-samples "$NUM_SAMPLES" \
    --model-ckt   "$CKPT" \
    --model-type  "$MODEL" \
    "$MLP_FLAG"

# Compare partial-MLP vs full-MLP at the same compression ratio:
#   MLP_MERGE=yes sh tasks/sam_hq44k/eval_hq44k.sh
#   MLP_MERGE=no  sh tasks/sam_hq44k/eval_hq44k.sh
#
# CSVs land in ./benchmark_results/ — diff them to isolate the partial-MLP
# accuracy contribution from sparse K/V attention.
