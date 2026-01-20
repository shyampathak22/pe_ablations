#!/bin/bash
# Evaluation launcher for PE Ablations
# Usage: ./scripts/run_eval.sh <checkpoint_path> [additional_args]
# Examples:
#   ./scripts/run_eval.sh checkpoints/fpope/best_model.pt
#   ./scripts/run_eval.sh checkpoints/alibi/best_model.pt --context-lengths 512 1024 2048 4096

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
CHECKPOINT_ARG="${1:?Error: checkpoint path or directory required}"
shift

# If given a directory, find the latest checkpoint
if [ -d "$CHECKPOINT_ARG" ]; then
    CHECKPOINT=$(ls -t "$CHECKPOINT_ARG"/checkpoint_step_*.pt 2>/dev/null | head -1)
    if [ -z "$CHECKPOINT" ]; then
        echo "Error: No checkpoint_step_*.pt files found in $CHECKPOINT_ARG"
        exit 1
    fi
    echo "Auto-selected latest checkpoint: $CHECKPOINT"
else
    CHECKPOINT="$CHECKPOINT_ARG"
fi

echo "=========================================="
echo "PE Ablations Evaluation Launcher"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',')"
echo "=========================================="

# Run evaluation
python "$PROJECT_DIR/scripts/evaluate.py" \
    --checkpoint "$CHECKPOINT" \
    "$@"
