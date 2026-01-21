#!/bin/bash
# PoPE Ablation Study - DDP Training Script
# Runs pure_pope and pope_floor experiments sequentially
#
# Tests hypothesis: FoPE's Fourier mixing may be unnecessary with PoPE
# since PoPE already decouples content from position.

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Number of GPUs to use
NGPUS=${NGPUS:-2}

echo "=========================================="
echo "PoPE Ablation Study"
echo "Using $NGPUS GPUs"
echo "=========================================="

# Run 1: Pure PoPE (matching Cinnamon reference)
echo ""
echo "[$(date)] Starting pure_pope training..."
echo "Config: No Fourier mixing, no floor clipping (exact PoPE paper)"
echo ""

torchrun --nproc_per_node=$NGPUS --master_port=29500 \
    scripts/train.py --config configs/pure_pope.yaml --compile

echo ""
echo "[$(date)] pure_pope training completed!"
echo ""

# Run 2: PoPE with floor frequency clipping
echo "=========================================="
echo "[$(date)] Starting pope_floor training..."
echo "Config: No Fourier mixing, with floor clipping (undertrained freq → zero)"
echo ""

torchrun --nproc_per_node=$NGPUS --master_port=29500 \
    scripts/train.py --config configs/pope_floor.yaml --compile

echo ""
echo "[$(date)] pope_floor training completed!"
echo ""

echo "=========================================="
echo "[$(date)] All PoPE ablations completed!"
echo "=========================================="
echo ""
echo "Checkpoints saved to:"
echo "  - checkpoints/pure_pope/"
echo "  - checkpoints/pope_floor/"
echo ""
echo "To evaluate, run:"
echo "  python scripts/evaluate_ppl_degradation.py --models pure_pope pope_floor"
