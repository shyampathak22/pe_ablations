#!/bin/bash
# Run evaluation on all trained models (without retraining)
# Uses both GPUs in parallel for faster evaluation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$PROJECT_DIR/setup_env.sh"

CONTEXT_LENGTHS="512 1024 2048 4096 8192"

# Models to evaluate
MODELS=(nope alibi rope_baseline rope_ntk rope_yarn fpope)

echo "=== Running evaluations on 2 GPUs in parallel ==="

# Run pairs of models in parallel on GPU 0 and GPU 1
for ((i=0; i<${#MODELS[@]}; i+=2)); do
    model1="${MODELS[$i]}"
    model2="${MODELS[$((i+1))]:-}"  # May be empty if odd number

    echo ""
    echo "=== Batch $((i/2 + 1)): $model1 (GPU 0)" ${model2:+"+ $model2 (GPU 1)"} "==="

    # Run first model on GPU 0
    CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/scripts/evaluate.py" \
        --checkpoint "$PROJECT_DIR/checkpoints/$model1" \
        --output-dir "$PROJECT_DIR/eval_results/$model1" \
        --context-lengths $CONTEXT_LENGTHS \
        --benchmarks passkey niah &
    PID1=$!

    # Run second model on GPU 1 (if exists)
    if [ -n "$model2" ]; then
        CUDA_VISIBLE_DEVICES=1 python "$PROJECT_DIR/scripts/evaluate.py" \
            --checkpoint "$PROJECT_DIR/checkpoints/$model2" \
            --output-dir "$PROJECT_DIR/eval_results/$model2" \
            --context-lengths $CONTEXT_LENGTHS \
            --benchmarks passkey niah &
        PID2=$!
        wait $PID1 $PID2
    else
        wait $PID1
    fi
done

echo ""
echo "=== Generating comparison plots ==="
python "$PROJECT_DIR/scripts/plot_results.py" \
    --results-dir "$PROJECT_DIR/eval_results" \
    --output-dir "$PROJECT_DIR/plots"

echo ""
echo "=== Uploading to WandB ==="
python "$PROJECT_DIR/scripts/upload_to_wandb.py" \
    --results-dir "$PROJECT_DIR/eval_results" \
    --plots-dir "$PROJECT_DIR/plots" \
    --project pe-ablations

echo ""
echo "All evaluations complete!"
echo "  Results: $PROJECT_DIR/eval_results/"
echo "  Plots:   $PROJECT_DIR/plots/"
