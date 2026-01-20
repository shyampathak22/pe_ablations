#!/bin/bash
# Run all PE ablation experiments
# Usage: ./scripts/run_all_experiments.sh [--compile]

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
EXTRA_ARGS="$@"

# Determine number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1
fi

echo "=========================================="
echo "PE Ablations - Full Experiment Suite"
echo "=========================================="
echo "GPUs: $NUM_GPUS"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ',')"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "Extra args: $EXTRA_ARGS"
echo "=========================================="

# Training configurations in order (fastest to slowest)
CONFIGS=(
    "nope"
    "alibi"
    "base_config"
    "rope_ntk"
    "rope_yarn"
    "fpope"
)

# Corresponding wandb run names
RUN_NAMES=(
    "nope"
    "alibi"
    "rope_baseline"
    "rope_ntk"
    "rope_yarn"
    "fpope"
)

DESCRIPTIONS=(
    "NoPE (No Positional Encoding baseline)"
    "ALiBi (Attention with Linear Biases)"
    "RoPE Baseline"
    "RoPE + NTK-aware scaling"
    "RoPE + YaRN scaling"
    "FPoPE (Fourier + Polar Position Embedding)"
)

# Run each experiment
for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    RUN_NAME="${RUN_NAMES[$i]}"
    DESC="${DESCRIPTIONS[$i]}"

    echo ""
    echo "=========================================="
    echo "[$((i+1))/${#CONFIGS[@]}] Training: $DESC"
    echo "  Config: configs/${CONFIG}.yaml"
    echo "  Run name: $RUN_NAME"
    echo "=========================================="

    if [ "$NUM_GPUS" -gt 1 ]; then
        torchrun --nproc_per_node=$NUM_GPUS "$PROJECT_DIR/scripts/train.py" \
            --config "$PROJECT_DIR/configs/${CONFIG}.yaml" \
            --wandb-run-name "$RUN_NAME" \
            $EXTRA_ARGS
    else
        python "$PROJECT_DIR/scripts/train.py" \
            --config "$PROJECT_DIR/configs/${CONFIG}.yaml" \
            --wandb-run-name "$RUN_NAME" \
            $EXTRA_ARGS
    fi

    echo "Completed: $DESC"
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

# Run evaluation on all trained models
echo ""
echo "Running evaluation on all models..."
echo "=========================================="

CONTEXT_LENGTHS="512 1024 2048 4096 8192"

# Checkpoint directories match the config checkpoint_dir values
CKPT_DIRS=(
    "checkpoints/nope"
    "checkpoints/alibi"
    "checkpoints/rope_baseline"
    "checkpoints/rope_ntk"
    "checkpoints/rope_yarn"
    "checkpoints/fpope"
)

for i in "${!RUN_NAMES[@]}"; do
    RUN_NAME="${RUN_NAMES[$i]}"
    CKPT_DIR="${CKPT_DIRS[$i]}"
    CKPT_PATH="$PROJECT_DIR/$CKPT_DIR"

    if [ -d "$CKPT_PATH" ]; then
        echo "Evaluating: $RUN_NAME (from $CKPT_DIR)"
        "$PROJECT_DIR/scripts/run_eval.sh" "$CKPT_PATH" \
            --output-dir "$PROJECT_DIR/eval_results/${RUN_NAME}" \
            --context-lengths $CONTEXT_LENGTHS \
            --benchmarks passkey niah
    else
        echo "Skipping $RUN_NAME - checkpoint dir not found at $CKPT_PATH"
    fi
done

echo ""
echo "=========================================="
echo "All experiments and evaluations completed!"
echo "=========================================="
