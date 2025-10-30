
# CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=29508 quant/hq44k_engine_quan.py 

#!/bin/bash

YAML_FILE="./quant/config/hq44k/gptq_cuda.yaml"
PYTHON_SCRIPT="quant/hq44k_engine_quan.py"
CUDA_DEVICE=2
MASTER_PORT=29501

# Bit combinations to test
declare -a BIT_COMBINATIONS=("4,4" )

# Function to update YAML
update_yaml_bits() {
    sed -i "s/n_bits: [0-9]*/n_bits: $1/" "$YAML_FILE"
    sed -i "s/n_bits_mlp: [0-9]*/n_bits_mlp: $2/" "$YAML_FILE"
}

# Run experiments
for combination in "${BIT_COMBINATIONS[@]}"; do
    IFS=',' read -r n_bits n_bits_mlp <<< "$combination"
    echo "Running: n_bits=$n_bits, n_bits_mlp=$n_bits_mlp"
    
    update_yaml_bits $n_bits $n_bits_mlp
    
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE torchrun \
        --nproc_per_node=1 \
        --master_port=$MASTER_PORT \
        $PYTHON_SCRIPT --config $YAML_FILE

done