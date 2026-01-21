#!/bin/bash
# 1B Token Ablation Study - DDP Training Script
# Runs fpope_frozen, pure_pope, and pope_floor at 1B tokens
# Expected runtime: ~7-8 hours total (2-3 hours each)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

NGPUS=${NGPUS:-2}

echo "=========================================="
echo "1B Token PE Ablation Study"
echo "Using $NGPUS GPUs"
echo "Expected runtime: ~7-8 hours"
echo "=========================================="

# Run 1: FPoPE frozen (baseline chimera)
echo ""
echo "[$(date)] Starting fpope_frozen_1b training..."
echo "Config: FoPE + PoPE with frozen coefficients"
echo ""

torchrun --nproc_per_node=$NGPUS --master_port=29500 \
    scripts/train.py --config configs/1b/fpope_frozen_1b.yaml --compile

echo ""
echo "[$(date)] fpope_frozen_1b completed!"
echo ""

# Run 2: Pure PoPE
echo "=========================================="
echo "[$(date)] Starting pure_pope_1b training..."
echo "Config: Pure PoPE (no FoPE, no floor clipping)"
echo ""

torchrun --nproc_per_node=$NGPUS --master_port=29500 \
    scripts/train.py --config configs/1b/pure_pope_1b.yaml --compile

echo ""
echo "[$(date)] pure_pope_1b completed!"
echo ""

# Run 3: PoPE with floor clipping
echo "=========================================="
echo "[$(date)] Starting pope_floor_1b training..."
echo "Config: PoPE with floor clipping (undertrained freq → zero)"
echo ""

torchrun --nproc_per_node=$NGPUS --master_port=29500 \
    scripts/train.py --config configs/1b/pope_floor_1b.yaml --compile

echo ""
echo "[$(date)] pope_floor_1b completed!"
echo ""

echo "=========================================="
echo "[$(date)] All 1B ablations completed!"
echo "=========================================="
echo ""
echo "Checkpoints saved to:"
echo "  - checkpoints/1b/fpope_frozen_1b/"
echo "  - checkpoints/1b/pure_pope_1b/"
echo "  - checkpoints/1b/pope_floor_1b/"
echo ""
echo "To evaluate, run:"
echo "  python scripts/evaluate_ppl_degradation.py --checkpoint-dir checkpoints/1b"
