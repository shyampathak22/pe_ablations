#!/bin/bash
# Training launcher for PE Ablations
# Usage: ./scripts/run_training.sh <config_name> [additional_args]
# Examples:
#   ./scripts/run_training.sh nope
#   ./scripts/run_training.sh alibi --compile
#   ./scripts/run_training.sh fpope --wandb-run-name fpope-v2

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# CUDA 12.8 setup (required for Blackwell GPUs - RTX 50xx series)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

# Library paths for torch and CUDA
VENV_DIR="$PROJECT_DIR/.venv"
TORCH_LIB="$VENV_DIR/lib/python3.13/site-packages/torch/lib"
NVIDIA_LIBS="$VENV_DIR/lib/python3.13/site-packages/nvidia"

export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$TORCH_LIB:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/cuda_cupti/lib:${LD_LIBRARY_PATH:-}"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Parse arguments
CONFIG_NAME="${1:-base_config}"
shift 2>/dev/null || true

# Determine number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "No GPUs detected, using CPU"
    NUM_GPUS=1
fi

echo "=========================================="
echo "PE Ablations Training Launcher"
echo "=========================================="
echo "Config: configs/${CONFIG_NAME}.yaml"
echo "GPUs: $NUM_GPUS"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "=========================================="

# Run training
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running distributed training with $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS "$PROJECT_DIR/scripts/train.py" \
        --config "$PROJECT_DIR/configs/${CONFIG_NAME}.yaml" \
        "$@"
else
    echo "Running single GPU training..."
    python "$PROJECT_DIR/scripts/train.py" \
        --config "$PROJECT_DIR/configs/${CONFIG_NAME}.yaml" \
        "$@"
fi
