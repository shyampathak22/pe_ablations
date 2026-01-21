"""Evaluate PE methods by measuring perplexity degradation as context length increases.

This is a better metric than generation-based benchmarks for undertrained models because:
1. It directly measures what PE is supposed to do: maintain model performance at longer contexts
2. Doesn't require good generation capability
3. Fast and deterministic

Usage:
    python scripts/evaluate_ppl_degradation.py --output-dir eval_results/ppl_degradation
    python scripts/evaluate_ppl_degradation.py --context-lengths 512 1024 2048 4096 8192 16384
"""

import argparse
import json
import math
from pathlib import Path

import torch
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
        # FPoPE config
        fpope_theta=model_cfg.get("fpope_theta", 10000.0),
        fpope_num_fourier_terms=model_cfg.get("fpope_num_fourier_terms", 64),
        fpope_sigma=model_cfg.get("fpope_sigma", 0.4),
        fpope_training_length=model_cfg.get("fpope_training_length", 512),
        fpope_delta_init=model_cfg.get("fpope_delta_init", "zero"),
        fpope_d_rope=model_cfg.get("fpope_d_rope", 32),
        fpope_freeze_coeffs=model_cfg.get("fpope_freeze_coeffs", False),
        fpope_use_ceiling=model_cfg.get("fpope_use_ceiling", False),
        fpope_normalize_coeffs=model_cfg.get("fpope_normalize_coeffs", True),
        # Pure PoPE config
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


def compute_ppl(
    model: Transformer,
    tokens: list[int],
    context_length: int,
    device: torch.device,
    stride: int | None = None,
) -> float:
    """Compute perplexity using sliding window for long sequences."""
    model.extend_context_length(context_length)

    if stride is None:
        stride = context_length // 2

    # Use multiple windows and average
    total_loss = 0.0
    total_tokens = 0

    seq_len = min(len(tokens), context_length * 4)  # Use up to 4x context worth of data
    tokens = tokens[:seq_len]

    for start in range(0, len(tokens) - context_length, stride):
        end = start + context_length
        chunk = tokens[start:end]

        input_ids = torch.tensor([chunk], device=device)

        with torch.no_grad():
            output = model(input_ids, labels=input_ids)
            loss = output["loss"].item()

        # Weight by number of tokens (excluding first token which has no context)
        n_tokens = context_length - 1
        total_loss += loss * n_tokens
        total_tokens += n_tokens

        if total_tokens >= context_length * 2:  # Enough samples
            break

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return math.exp(avg_loss) if avg_loss < 20 else float('inf')


def evaluate_model(
    model: Transformer,
    tokens: list[int],
    context_lengths: list[int],
    device: torch.device,
) -> dict:
    """Evaluate model at multiple context lengths."""
    results = {"ppl": {}, "ratio": {}, "loss": {}}
    base_ppl = None

    for ctx_len in context_lengths:
        try:
            ppl = compute_ppl(model, tokens, ctx_len, device)
            loss = math.log(ppl) if ppl < float('inf') else float('inf')

            results["ppl"][ctx_len] = ppl
            results["loss"][ctx_len] = loss

            if base_ppl is None:
                base_ppl = ppl

            results["ratio"][ctx_len] = ppl / base_ppl if base_ppl > 0 else float('inf')

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    OOM at {ctx_len}, stopping")
                break
            raise

        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PE methods by PPL degradation")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--output-dir", type=str, default="eval_results/ppl_degradation",
                        help="Directory to save results")
    parser.add_argument("--context-lengths", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096, 8192],
                        help="Context lengths to test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Specific models to evaluate (default: all)")
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
    print("=" * 70)

    for model_dir in model_dirs:
        # Find latest checkpoint
        ckpts = sorted(model_dir.glob("checkpoint_step_*.pt"))
        if not ckpts:
            continue

        ckpt_path = ckpts[-1]
        name = model_dir.name

        print(f"\n{name}:")

        # Load model
        model, config, pe_mode = load_model(str(ckpt_path), device)
        print(f"  PE mode: {pe_mode}, params: {config.num_params:,}")

        # Evaluate
        results = evaluate_model(model, all_tokens, args.context_lengths, device)
        results["pe_mode"] = pe_mode
        results["checkpoint"] = str(ckpt_path)
        all_results[name] = results

        # Print results
        print(f"  {'Ctx':>6} | {'PPL':>10} | {'Ratio':>8} | Status")
        print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+--------")

        for ctx_len in args.context_lengths:
            if ctx_len not in results["ppl"]:
                break
            ppl = results["ppl"][ctx_len]
            ratio = results["ratio"][ctx_len]

            if ctx_len == args.context_lengths[0]:
                status = "baseline"
            elif ratio < 1.2:
                status = "excellent"
            elif ratio < 1.5:
                status = "good"
            elif ratio < 2.0:
                status = "degraded"
            elif ratio < 5.0:
                status = "poor"
            else:
                status = "BROKEN"

            print(f"  {ctx_len:>6} | {ppl:>10.1f} | {ratio:>7.2f}x | {status}")

        del model
        torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: PPL Degradation Ratio (lower is better)")
    print("=" * 70)

    # Header
    print(f"{'Model':<18}", end="")
    for ctx in args.context_lengths:
        print(f" | {ctx:>7}", end="")
    print(" | Score")
    print("-" * (20 + 10 * len(args.context_lengths) + 8))

    # Rows sorted by average ratio
    def avg_ratio(item):
        name = item[0] if isinstance(item, tuple) else item
        ratios = [all_results[name]["ratio"].get(c, 999) for c in args.context_lengths[1:]]
        return sum(ratios) / len(ratios) if ratios else 999

    for name in sorted(all_results.keys(), key=avg_ratio):
        print(f"{name:<18}", end="")
        for ctx in args.context_lengths:
            ratio = all_results[name]["ratio"].get(ctx, float('nan'))
            if math.isnan(ratio):
                print(f" | {'N/A':>7}", end="")
            else:
                marker = "" if ratio < 1.5 else "*" if ratio < 3 else "**"
                print(f" | {ratio:>6.2f}{marker}", end="")

        # Overall score (average ratio excluding baseline)
        score = avg_ratio(name)
        print(f" | {score:.2f}x")

    print("\n(* = degraded >1.5x, ** = broken >3x)")
    print("Lower score = better length extrapolation")

    # Save results
    output_file = output_dir / "ppl_degradation_results.json"
    with open(output_file, "w") as f:
        # Convert keys to strings for JSON
        json_results = {}
        for name, res in all_results.items():
            json_results[name] = {
                "pe_mode": res["pe_mode"],
                "checkpoint": res["checkpoint"],
                "ppl": {str(k): v for k, v in res["ppl"].items()},
                "ratio": {str(k): v for k, v in res["ratio"].items()},
                "loss": {str(k): v for k, v in res["loss"].items()},
            }
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Generate plot
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        colors = plt.cm.tab10.colors

        # Plot 1: PPL vs context length
        for i, (name, res) in enumerate(sorted(all_results.items(), key=avg_ratio)):
            ctx_lens = sorted([int(k) for k in res["ppl"].keys()])
            ppls = [res["ppl"][c] for c in ctx_lens]
            ax1.plot(ctx_lens, ppls, marker='o', label=name, color=colors[i % len(colors)])

        ax1.set_xlabel("Context Length")
        ax1.set_ylabel("Perplexity")
        ax1.set_title("Perplexity vs Context Length")
        ax1.set_xscale("log", base=2)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Ratio vs context length
        for i, (name, res) in enumerate(sorted(all_results.items(), key=avg_ratio)):
            ctx_lens = sorted([int(k) for k in res["ratio"].keys()])
            ratios = [res["ratio"][c] for c in ctx_lens]
            ax2.plot(ctx_lens, ratios, marker='o', label=name, color=colors[i % len(colors)])

        ax2.set_xlabel("Context Length")
        ax2.set_ylabel("PPL Ratio (vs training length)")
        ax2.set_title("PPL Degradation Ratio")
        ax2.set_xscale("log", base=2)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='1.5x threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = output_dir / "ppl_degradation_plot.png"
        plt.savefig(plot_file, dpi=150)
        print(f"Plot saved to {plot_file}")
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
