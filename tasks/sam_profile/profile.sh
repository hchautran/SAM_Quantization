ALGO=${ALGO:-sparsesam}
BATCH_SIZE=${BATCH_SIZE:-8}
RATIO=${RATIO:-0.25}
GLOBAL_RATIO=${GLOBAL_RATIO:-0.25}

python profile_encoder.py --version sam1 \
    --batch-size=${BATCH_SIZE} \
    --model-ckt /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/SAM_Quantization/ckts/sam_hq_vit_l.pth \
    --model-type vit_l \
    --algo ${ALGO} \
    --tome-ratio ${RATIO} \
    --global-ratio ${GLOBAL_RATIO}
