#!/bin/bash
# Setup environment for PE Ablations with CUDA 12.8
# Source this file before running training/evaluation:
#   source setup_env.sh

# CUDA 12.8 paths (required for Blackwell GPUs - RTX 50xx series)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

# Library paths for torch and CUDA
VENV_DIR="$(dirname "${BASH_SOURCE[0]}")/.venv"
TORCH_LIB="$VENV_DIR/lib/python3.13/site-packages/torch/lib"
NVIDIA_LIBS="$VENV_DIR/lib/python3.13/site-packages/nvidia"

export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$TORCH_LIB:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/cuda_cupti/lib:${LD_LIBRARY_PATH:-}"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

echo "Environment configured:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  Python: $(which python)"
echo "  Torch: $(python -c 'import torch; print(torch.__version__)')"
