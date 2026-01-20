"""Generate comparison plots from evaluation results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(eval_dir: Path) -> dict:
    """Load all results from an evaluation directory."""
    results = {"name": eval_dir.name}

    passkey_file = eval_dir / "passkey_results.json"
    if passkey_file.exists():
        with open(passkey_file) as f:
            results["passkey"] = json.load(f)

    niah_file = eval_dir / "niah_results.json"
    if niah_file.exists():
        with open(niah_file) as f:
            results["niah"] = json.load(f)

    return results


def plot_passkey_comparison(all_results: list[dict], output_path: Path) -> None:
    """Plot passkey accuracy vs context length for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    markers = ['o', 's', '^', 'D', 'v', 'p']

    for i, results in enumerate(all_results):
        if "passkey" not in results:
            continue

        passkey = results["passkey"]
        ctx_lengths = sorted([int(k) for k in passkey["accuracy_by_length"].keys()])
        accuracies = [passkey["accuracy_by_length"][str(k)] for k in ctx_lengths]

        ax.plot(ctx_lengths, accuracies,
                marker=markers[i % len(markers)],
                color=colors[i],
                label=results["name"],
                linewidth=2,
                markersize=8)

    ax.set_xlabel("Context Length", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Passkey Retrieval: Accuracy vs Context Length", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    ax.axvline(x=512, color='red', linestyle=':', alpha=0.5, label='Training length (512)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_niah_comparison(all_results: list[dict], output_path: Path) -> None:
    """Plot NIAH overall scores for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    names = []
    scores = []

    for results in all_results:
        if "niah" in results:
            names.append(results["name"])
            scores.append(results["niah"]["overall_score"])

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.bar(names, scores, color=colors, edgecolor='black')

    ax.set_ylabel("Overall Score", fontsize=12)
    ax.set_title("Needle in a Haystack: Overall Scores", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.1%}', ha='center', va='bottom', fontsize=10)

    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_extrapolation_summary(all_results: list[dict], output_path: Path) -> None:
    """Plot extrapolation performance summary."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Passkey at different extrapolation ratios
    ax1 = axes[0]
    training_len = 512
    extrapolation_ratios = [1, 2, 4, 8, 16, 32]
    target_lengths = [training_len * r for r in extrapolation_ratios]

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for i, results in enumerate(all_results):
        if "passkey" not in results:
            continue

        passkey = results["passkey"]
        accs = []
        for tgt in target_lengths:
            acc = passkey["accuracy_by_length"].get(str(tgt), None)
            accs.append(acc)

        valid_ratios = [r for r, a in zip(extrapolation_ratios, accs) if a is not None]
        valid_accs = [a for a in accs if a is not None]

        if valid_accs:
            ax1.plot(valid_ratios, valid_accs,
                    marker='o', color=colors[i],
                    label=results["name"], linewidth=2, markersize=8)

    ax1.set_xlabel("Extrapolation Ratio (Context / Training Length)", fontsize=11)
    ax1.set_ylabel("Passkey Accuracy", fontsize=11)
    ax1.set_title("Extrapolation Performance", fontsize=12)
    ax1.set_xscale("log", base=2)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Summary table as bar chart
    ax2 = axes[1]

    # Calculate metrics for each model
    model_names = []
    in_dist_acc = []  # At training length
    extrapolation_4x = []  # At 4x
    extrapolation_8x = []  # At 8x

    for results in all_results:
        if "passkey" not in results:
            continue

        passkey = results["passkey"]
        model_names.append(results["name"])
        in_dist_acc.append(passkey["accuracy_by_length"].get("512", 0))
        extrapolation_4x.append(passkey["accuracy_by_length"].get("2048", 0))
        extrapolation_8x.append(passkey["accuracy_by_length"].get("4096", 0))

    x = np.arange(len(model_names))
    width = 0.25

    ax2.bar(x - width, in_dist_acc, width, label='1x (512)', color='green', alpha=0.8)
    ax2.bar(x, extrapolation_4x, width, label='4x (2048)', color='orange', alpha=0.8)
    ax2.bar(x + width, extrapolation_8x, width, label='8x (4096)', color='red', alpha=0.8)

    ax2.set_ylabel("Accuracy", fontsize=11)
    ax2.set_title("Accuracy at Key Extrapolation Points", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot PE ablation results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="eval_results",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    all_results = []
    for subdir in sorted(results_dir.iterdir()):
        if subdir.is_dir():
            results = load_results(subdir)
            if "passkey" in results or "niah" in results:
                all_results.append(results)
                print(f"Loaded: {subdir.name}")

    if not all_results:
        print("No results found!")
        return

    print(f"\nGenerating plots for {len(all_results)} models...")

    # Generate plots
    plot_passkey_comparison(all_results, output_dir / "passkey_comparison.png")
    plot_niah_comparison(all_results, output_dir / "niah_comparison.png")
    plot_extrapolation_summary(all_results, output_dir / "extrapolation_summary.png")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
