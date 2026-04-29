#!/usr/bin/env bash
# Profile SAM2 image-encoder components.
#
#   • Default: baseline + sparsesam comparison (block-sparse attn + local-block MLP prune).
#   • Set ALGO=none for baseline only.
#   • RATIO is the keep-bar fraction in (0, 1] — lower = more compression.

set -e

CONFIG=${CONFIG:-configs/sam2.1/sam2.1_hq_hiera_l.yaml}
CKPT=${CKPT:-./ckts/sam2.1_hq_hiera_large.pt}
ALGO=${ALGO:-sparsesam}
RATIO=${RATIO:-0.5}
BATCH=${BATCH:-1}
DTYPE=${DTYPE:-fp16}
N_RUNS=${N_RUNS:-20}

python profile_sam2.py \
    --sam2-config "$CONFIG" \
    --model-ckt   "$CKPT" \
    --batch-size  "$BATCH" \
    --dtype       "$DTYPE" \
    --n-runs      "$N_RUNS" \
    --algo        "$ALGO" \
    --ratio       "$RATIO"

# Examples:
#
#   # Baseline only
#   ALGO=none sh profile_sam2.sh
#
#   # Sparsesam at multiple ratios
#   for r in 0.75 0.5 0.25; do RATIO=$r sh profile_sam2.sh; done
#
#   # Ablate MLP pruning (attention-only sparsesam)
#   python profile_sam2.py --algo sparsesam --ratio 0.5 --no-mlp-prune
#
#   # Force SDPA fallback (no cute kernel) — useful for an A/B vs. the cute path
#   python profile_sam2.py --algo sparsesam --ratio 0.5 --no-cute
