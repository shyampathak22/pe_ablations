"""Upload evaluation results and plots to WandB."""

import argparse
import json
from pathlib import Path

import wandb


def upload_results(results_dir: Path, plots_dir: Path, project: str) -> None:
    """Upload all evaluation results to WandB."""

    # Initialize a summary run
    run = wandb.init(
        project=project,
        name="evaluation-summary",
        job_type="evaluation",
    )

    # Load and log results for each model
    all_passkey = {}
    all_niah = {}

    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue

        model_name = subdir.name

        # Load passkey results
        passkey_file = subdir / "passkey_results.json"
        if passkey_file.exists():
            with open(passkey_file) as f:
                passkey = json.load(f)
            all_passkey[model_name] = passkey

            # Log individual metrics
            for ctx_len, acc in passkey["accuracy_by_length"].items():
                wandb.log({
                    f"passkey/{model_name}/ctx_{ctx_len}": acc,
                })

            wandb.log({
                f"passkey/{model_name}/overall": passkey["overall_accuracy"],
            })

        # Load NIAH results
        niah_file = subdir / "niah_results.json"
        if niah_file.exists():
            with open(niah_file) as f:
                niah = json.load(f)
            all_niah[model_name] = niah

            wandb.log({
                f"niah/{model_name}/overall": niah["overall_score"],
            })

            # Upload heatmap if exists
            heatmap_file = subdir / "niah_heatmap.png"
            if heatmap_file.exists():
                wandb.log({
                    f"niah/{model_name}/heatmap": wandb.Image(str(heatmap_file)),
                })

    # Create comparison tables
    if all_passkey:
        # Passkey comparison table
        ctx_lengths = set()
        for passkey in all_passkey.values():
            ctx_lengths.update(passkey["accuracy_by_length"].keys())
        ctx_lengths = sorted([int(c) for c in ctx_lengths])

        table_data = []
        for model_name, passkey in all_passkey.items():
            row = [model_name]
            for ctx in ctx_lengths:
                row.append(passkey["accuracy_by_length"].get(str(ctx), None))
            row.append(passkey["overall_accuracy"])
            table_data.append(row)

        columns = ["Model"] + [f"Ctx {c}" for c in ctx_lengths] + ["Overall"]
        table = wandb.Table(data=table_data, columns=columns)
        wandb.log({"passkey/comparison_table": table})

    if all_niah:
        # NIAH comparison table
        table_data = [[name, niah["overall_score"]] for name, niah in all_niah.items()]
        table = wandb.Table(data=table_data, columns=["Model", "Overall Score"])
        wandb.log({"niah/comparison_table": table})

    # Upload comparison plots
    if plots_dir.exists():
        for plot_file in plots_dir.glob("*.png"):
            wandb.log({
                f"plots/{plot_file.stem}": wandb.Image(str(plot_file)),
            })

    # Log summary artifact
    artifact = wandb.Artifact("evaluation-results", type="results")
    artifact.add_dir(str(results_dir))
    if plots_dir.exists():
        artifact.add_dir(str(plots_dir))
    run.log_artifact(artifact)

    wandb.finish()
    print(f"Results uploaded to WandB project: {project}")


def main():
    parser = argparse.ArgumentParser(description="Upload results to WandB")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="eval_results",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="plots",
        help="Directory containing plots",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="pe-ablations",
        help="WandB project name",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    upload_results(results_dir, plots_dir, args.project)


if __name__ == "__main__":
    main()
