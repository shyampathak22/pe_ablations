"""Evaluation script for context length extrapolation benchmarks."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.data import get_tokenizer
from src.evaluation import PasskeyRetrievalBenchmark, NIAHBenchmark, RULERBenchmark
from src.model import Transformer, TransformerConfig


def load_model(checkpoint_path: str, device: torch.device) -> tuple[Transformer, TransformerConfig]:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in checkpoint and "model" in checkpoint["config"]:
        model_cfg = checkpoint["config"]["model"]
    else:
        model_cfg = checkpoint.get("model_config", {})

    config = TransformerConfig(
        vocab_size=model_cfg.get("vocab_size", 50257),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        num_layers=model_cfg.get("num_layers", 24),
        num_heads=model_cfg.get("num_heads", 8),
        num_kv_heads=model_cfg.get("num_kv_heads", 4),
        head_dim=model_cfg.get("head_dim", 64),
        max_seq_len=model_cfg.get("max_seq_len", 512),
        rope_type=model_cfg.get("rope_type", "standard"),
        rope_theta=model_cfg.get("rope_theta", 10000.0),
        use_qk_norm=model_cfg.get("use_qk_norm", True),
        tie_embeddings=model_cfg.get("tie_embeddings", True),
    )

    model = Transformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def run_passkey(
    model: Transformer,
    tokenizer,
    context_lengths: list[int],
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Run passkey retrieval benchmark."""
    print("Running Passkey Retrieval benchmark...")

    benchmark = PasskeyRetrievalBenchmark(
        tokenizer=tokenizer,
        num_samples=5,
        passkey_length=5,
    )

    results = benchmark.evaluate(model, context_lengths, device)

    print(f"\nPasskey Retrieval Results:")
    print(f"  Overall accuracy: {results['overall_accuracy']:.2%}")
    for ctx_len, acc in results["accuracy_by_length"].items():
        print(f"  Context {ctx_len}: {acc:.2%}")

    output_file = output_dir / "passkey_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "overall_accuracy": results["overall_accuracy"],
                "accuracy_by_length": {str(k): v for k, v in results["accuracy_by_length"].items()},
                "accuracy_by_position": {str(k): v for k, v in results["accuracy_by_position"].items()},
            },
            f,
            indent=2,
        )

    return results


def run_niah(
    model: Transformer,
    tokenizer,
    context_lengths: list[int],
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Run Needle in a Haystack benchmark."""
    print("Running Needle in a Haystack benchmark...")

    benchmark = NIAHBenchmark(
        tokenizer=tokenizer,
        num_samples=1,
    )

    results = benchmark.evaluate(model, context_lengths, device)

    print(f"\nNIAH Results:")
    print(f"  Overall score: {results['overall_score']:.2%}")

    output_file = output_dir / "niah_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "overall_score": results["overall_score"],
                "depths": results["depths"],
                "context_lengths": results["context_lengths"],
                "score_matrix": results["score_matrix"].tolist(),
            },
            f,
            indent=2,
        )

    heatmap_file = output_dir / "niah_heatmap.png"
    benchmark.plot_heatmap(results, save_path=str(heatmap_file))
    print(f"  Heatmap saved to {heatmap_file}")

    return results


def run_ruler(
    model: Transformer,
    tokenizer,
    context_lengths: list[int],
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Run RULER benchmark."""
    print("Running RULER benchmark...")

    benchmark = RULERBenchmark(tokenizer=tokenizer)

    results = benchmark.evaluate(
        model,
        context_lengths,
        device,
        samples_per_task=5,
    )

    print(f"\nRULER Results:")
    print(f"  Overall accuracy: {results['overall_accuracy']:.2%}")
    for task, acc in results["accuracy_by_task"].items():
        print(f"  {task}: {acc:.2%}")

    output_file = output_dir / "ruler_results.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "overall_accuracy": results["overall_accuracy"],
                "accuracy_by_task": results["accuracy_by_task"],
                "accuracy_by_length": {str(k): v for k, v in results["accuracy_by_length"].items()},
                "accuracy_by_task_length": {
                    task: {str(k): v for k, v in lengths.items()}
                    for task, lengths in results["accuracy_by_task_length"].items()
                },
            },
            f,
            indent=2,
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PE ablations model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["passkey", "niah"],
        choices=["passkey", "niah", "ruler"],
        help="Benchmarks to run",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
        help="Context lengths to test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    print(f"Model loaded: {config.num_params:,} parameters")

    tokenizer = get_tokenizer("gpt2")

    print(f"Testing context lengths: {args.context_lengths}")

    for benchmark in args.benchmarks:
        if benchmark == "passkey":
            run_passkey(model, tokenizer, args.context_lengths, device, output_dir)
        elif benchmark == "niah":
            run_niah(model, tokenizer, args.context_lengths, device, output_dir)
        elif benchmark == "ruler":
            run_ruler(model, tokenizer, args.context_lengths, device, output_dir)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
