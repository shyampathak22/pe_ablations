"""Long context PPL evaluation with sliding window.

Evaluates PPL at long context lengths using sliding window with overlap.
Each token's loss is computed with full preceding context up to window size.

Usage:
    python scripts/evaluate_long_context.py \
        --checkpoint-dir checkpoints/1b \
        --context-lengths 512 1024 2048 4096 8192 16384 32768 \
        --output-dir eval_results/long_context
"""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from src.data import get_tokenizer
from src.model import Transformer, TransformerConfig


def load_model(checkpoint_path: str, device: torch.device) -> tuple[Transformer, TransformerConfig, str]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_cfg = checkpoint.get("config", {}).get("model", checkpoint.get("model_config", {}))
    pe_mode = model_cfg.get("pe_mode", "rope")

    config = TransformerConfig(
        vocab_size=model_cfg.get("vocab_size", 50257),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        num_layers=model_cfg.get("num_layers", 24),
        num_heads=model_cfg.get("num_heads", 8),
        num_kv_heads=model_cfg.get("num_kv_heads", 4),
        head_dim=model_cfg.get("head_dim", 64),
        max_seq_len=model_cfg.get("max_seq_len", 512),
        pe_mode=pe_mode,
        rope_theta=model_cfg.get("rope_theta", 10000.0),
        rope_type=model_cfg.get("rope_type", "standard"),
        rope_scale=model_cfg.get("rope_scale", 1.0),
        fpope_theta=model_cfg.get("fpope_theta", 10000.0),
        fpope_num_fourier_terms=model_cfg.get("fpope_num_fourier_terms", 64),
        fpope_sigma=model_cfg.get("fpope_sigma", 0.4),
        fpope_training_length=model_cfg.get("fpope_training_length", 512),
        fpope_delta_init=model_cfg.get("fpope_delta_init", "zero"),
        fpope_d_rope=model_cfg.get("fpope_d_rope", 32),
        fpope_freeze_coeffs=model_cfg.get("fpope_freeze_coeffs", False),
        fpope_use_ceiling=model_cfg.get("fpope_use_ceiling", False),
        fpope_normalize_coeffs=model_cfg.get("fpope_normalize_coeffs", True),
        pope_theta=model_cfg.get("pope_theta", 10000.0),
        pope_training_length=model_cfg.get("pope_training_length", 512),
        pope_delta_init=model_cfg.get("pope_delta_init", "zero"),
        pope_d_rope=model_cfg.get("pope_d_rope", 32),
    )

    model = Transformer(config)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    return model, config, pe_mode


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@torch.inference_mode()
def compute_ppl_sliding_window(
    model: Transformer,
    tokens: list[int],
    context_length: int,
    device: torch.device,
    stride: int | None = None,
) -> tuple[float, bool]:
    """Compute PPL using sliding window - each position sees full context_length history.

    Returns (ppl, oom_flag)
    """
    model.extend_context_length(context_length)

    if stride is None:
        stride = context_length // 2

    total_loss = 0.0
    total_tokens = 0
    oom = False

    # Use enough data to get stable estimate
    seq_len = min(len(tokens), context_length * 4)
    tokens = tokens[:seq_len]

    for start in range(0, len(tokens) - context_length, stride):
        end = start + context_length
        chunk = tokens[start:end]

        input_ids = torch.tensor([chunk], device=device)

        try:
            output = model(input_ids, labels=input_ids)
            loss = output["loss"].item()

            # Only count tokens in the second half (they have full context)
            n_tokens = context_length // 2
            total_loss += loss * n_tokens
            total_tokens += n_tokens

            if total_tokens >= context_length * 2:
                break

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom = True
                clear_memory()
                break
            raise

        clear_memory()

    if total_tokens == 0:
        return float('inf'), oom

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')
    return ppl, oom


def main():
    parser = argparse.ArgumentParser(description="Long context PPL evaluation")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/1b",
                        help="Directory containing model checkpoints")
    parser.add_argument("--output-dir", type=str, default="eval_results/long_context",
                        help="Directory to save results")
    parser.add_argument("--context-lengths", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
                        help="Context lengths to test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Specific models to evaluate")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation data
    print("Loading evaluation data...")
    tokenizer = get_tokenizer("gpt2")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    long_text = " ".join([x["text"] for x in ds if x["text"].strip()])
    all_tokens = tokenizer.encode(long_text)
    print(f"Loaded {len(all_tokens):,} tokens")

    # Find checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    model_dirs = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir()])

    if args.models:
        model_dirs = [d for d in model_dirs if d.name in args.models]

    all_results = {}

    print(f"\nEvaluating {len(model_dirs)} models at context lengths: {args.context_lengths}")
    print("=" * 80)

    for model_dir in model_dirs:
        ckpts = sorted(model_dir.glob("checkpoint_step_*.pt"))
        if not ckpts:
            continue

        ckpt_path = ckpts[-1]
        name = model_dir.name

        print(f"\n{name}:")

        model, config, pe_mode = load_model(str(ckpt_path), device)
        print(f"  PE mode: {pe_mode}, params: {config.num_params:,}")

        results = {"ppl": {}, "ratio": {}, "pe_mode": pe_mode}
        base_ppl = None

        for ctx_len in args.context_lengths:
            ppl, oom = compute_ppl_sliding_window(model, all_tokens, ctx_len, device)

            if oom:
                print(f"  {ctx_len:>6}: OOM")
                break

            results["ppl"][ctx_len] = ppl

            if base_ppl is None:
                base_ppl = ppl

            ratio = ppl / base_ppl if base_ppl > 0 else float('inf')
            results["ratio"][ctx_len] = ratio

            status = "baseline" if ctx_len == args.context_lengths[0] else \
                     "excellent" if ratio < 1.2 else \
                     "good" if ratio < 1.5 else \
                     "degraded" if ratio < 2.0 else \
                     "poor" if ratio < 5.0 else "BROKEN"

            print(f"  {ctx_len:>6}: PPL={ppl:>8.1f}, ratio={ratio:.2f}x [{status}]")

        all_results[name] = results
        del model
        clear_memory()

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: PPL Degradation Ratio (lower is better)")
    print("=" * 80)

    # Header
    print(f"{'Model':<20}", end="")
    for ctx in args.context_lengths:
        print(f" | {ctx:>7}", end="")
    print("")
    print("-" * (22 + 10 * len(args.context_lengths)))

    # Sort by performance at longest tested context
    def best_ratio(name):
        ratios = list(all_results[name]["ratio"].values())
        return ratios[-1] if ratios else 999

    for name in sorted(all_results.keys(), key=best_ratio):
        print(f"{name:<20}", end="")
        for ctx in args.context_lengths:
            ratio = all_results[name]["ratio"].get(ctx, float('nan'))
            if math.isnan(ratio):
                print(f" | {'OOM':>7}", end="")
            else:
                marker = "" if ratio < 1.5 else "*" if ratio < 3 else "**"
                print(f" | {ratio:>6.2f}{marker}", end="")
        print("")

    print("\n(* = degraded >1.5x, ** = broken >3x)")

    # Save results
    output_file = output_dir / "long_context_ppl.json"
    with open(output_file, "w") as f:
        json_results = {}
        for name, res in all_results.items():
            json_results[name] = {
                "pe_mode": res["pe_mode"],
                "ppl": {str(k): v for k, v in res["ppl"].items()},
                "ratio": {str(k): v for k, v in res["ratio"].items()},
            }
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Generate comparison plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = {'pope_floor_1b': 'green', 'pure_pope_1b': 'blue', 'fpope_frozen_1b': 'red'}

        for name in sorted(all_results.keys(), key=best_ratio):
            res = all_results[name]
            ctx_lens = sorted([int(k) for k in res["ratio"].keys()])
            ratios = [res["ratio"][c] for c in ctx_lens]
            color = colors.get(name, None)
            ax.plot(ctx_lens, ratios, marker='o', label=name, color=color, linewidth=2)

        ax.set_xlabel("Context Length", fontsize=12)
        ax.set_ylabel("PPL Ratio (vs training length)", fontsize=12)
        ax.set_title("PPL Degradation: PoPE vs FPoPE at Long Contexts", fontsize=14)
        ax.set_xscale("log", base=2)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No degradation')
        ax.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='1.5x threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = output_dir / "long_context_comparison.png"
        plt.savefig(plot_file, dpi=150)
        print(f"Plot saved to {plot_file}")
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
