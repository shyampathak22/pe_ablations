#!/bin/bash
# FPoPE Frequency Ablation Study - DDP Training Script
# Runs fpope_frozen and fpope_ceiling experiments sequentially

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Number of GPUs to use
NGPUS=${NGPUS:-2}

echo "=========================================="
echo "FPoPE Frequency Ablation Study"
echo "Using $NGPUS GPUs"
echo "=========================================="

# Run 1: FPoPE with frozen coefficients (like original FoPE paper)
echo ""
echo "[$(date)] Starting fpope_frozen training..."
echo "Config: freeze_coeffs=true, use_ceiling=false, normalize_coeffs=true"
echo ""

torchrun --nproc_per_node=$NGPUS --master_port=29500 \
    scripts/train.py --config configs/fpope_frozen.yaml --compile

echo ""
echo "[$(date)] fpope_frozen training completed!"
echo ""

# Run 2: FPoPE with learnable coefficients + ceiling clamp
echo "=========================================="
echo "[$(date)] Starting fpope_ceiling training..."
echo "Config: freeze_coeffs=false, use_ceiling=true, normalize_coeffs=true"
echo ""

torchrun --nproc_per_node=$NGPUS --master_port=29500 \
    scripts/train.py --config configs/fpope_ceiling.yaml --compile

echo ""
echo "[$(date)] fpope_ceiling training completed!"
echo ""

echo "=========================================="
echo "[$(date)] All ablations completed!"
echo "=========================================="
echo ""
echo "Checkpoints saved to:"
echo "  - checkpoints/fpope_frozen/"
echo "  - checkpoints/fpope_ceiling/"
echo ""
echo "To evaluate, run:"
echo "  python scripts/evaluate_ppl_degradation.py --checkpoint-dir checkpoints/fpope_frozen"
echo "  python scripts/evaluate_ppl_degradation.py --checkpoint-dir checkpoints/fpope_ceiling"
