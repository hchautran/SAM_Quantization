#!/bin/bash
# Profile PE vision encoder components, optionally comparing against a
# token-merging / fused-attention patch.
#
# Knobs (env-overridable):
#   MODEL=PE-Core-L14-336        which PE config (see eval_pe_clip.py --list-models)
#   ALGO=sparsesam               none | tome | gradtome | sparsesam | *_partial | flash_rope
#   RATIO=0.7                    per-stage-boundary keep fraction
#   NUM_STAGES=4                 split L blocks into N stages (N-1 merge points)
#   BATCH=128
#   DTYPE=fp16

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODEL=${MODEL:-PE-Core-L14-336}
ALGO=${ALGO:-sparsesam_partial}
RATIO=${RATIO:-0.5}
NUM_STAGES=${NUM_STAGES:-4}
BATCH=${BATCH:-128}
DTYPE=${DTYPE:-fp16}

python "$SCRIPT_DIR/profile_pe.py" \
    --model "$MODEL" \
    --batch-size "$BATCH" \
    --dtype "$DTYPE" \
    --tome-algo "$ALGO" \
    --tome-ratio "$RATIO" \
    --num-stages "$NUM_STAGES"
    # --use-flash-rope \
    # --fa2
