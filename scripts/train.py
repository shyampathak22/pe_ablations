"""Main training script for PE ablations."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data import create_dataloader, pretokenize_dataset
from src.model import Transformer, TransformerConfig
from src.training import Trainer, setup_distributed, cleanup_distributed, wrap_model_ddp
from src.training.trainer import TrainerConfig
from src.training.distributed import is_main_process, print_rank0, get_rank, get_world_size


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PE ablations model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--pretokenize-only",
        action="store_true",
        help="Only pretokenize the dataset, don't train",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster training",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="WandB run name",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    set_seed(config["training"]["seed"] + rank)

    data_dir = Path(args.data_dir or config["data"]["data_dir"])

    if args.pretokenize_only or not (data_dir / "train.bin").exists():
        if is_main_process():
            print_rank0("Pre-tokenizing dataset...")
            token_counts = pretokenize_dataset(
                output_dir=data_dir,
                dataset_name=config["data"]["dataset_name"],
                dataset_config=config["data"]["dataset_config"],
                tokenizer_name=config["data"]["tokenizer_name"],
            )
            print_rank0(f"Token counts: {token_counts}")

        if world_size > 1:
            torch.distributed.barrier()

        if args.pretokenize_only:
            cleanup_distributed()
            return

    print_rank0("Creating model...")

    # Get positional encoding mode
    pe_mode = config["model"].get("pe_mode", "rope")

    # Build model config with PE-specific parameters
    model_kwargs = {
        "vocab_size": config["model"]["vocab_size"],
        "hidden_dim": config["model"]["hidden_dim"],
        "num_layers": config["model"]["num_layers"],
        "num_heads": config["model"]["num_heads"],
        "num_kv_heads": config["model"]["num_kv_heads"],
        "head_dim": config["model"]["head_dim"],
        "ffn_dim": config["model"]["ffn_dim"],
        "ffn_multiple_of": config["model"]["ffn_multiple_of"],
        "max_seq_len": config["model"]["max_seq_len"],
        "norm_eps": config["model"]["norm_eps"],
        "use_qk_norm": config["model"]["use_qk_norm"],
        "dropout": config["model"]["dropout"],
        "tie_embeddings": config["model"]["tie_embeddings"],
        "gradient_checkpointing": config["model"]["gradient_checkpointing"],
        "pe_mode": pe_mode,
    }

    if pe_mode == "rope":
        # RoPE-specific parameters
        model_kwargs.update({
            "rope_theta": config["model"].get("rope_theta", 10000.0),
            "rope_type": config["model"].get("rope_type", "default"),
            "rope_scale": config["model"].get("rope_scale", 1.0),
        })
    elif pe_mode == "fpope":
        # FPoPE-specific parameters
        model_kwargs.update({
            "fpope_theta": config["model"].get("fpope_theta", 10000.0),
            "fpope_num_fourier_terms": config["model"].get("fpope_num_fourier_terms", 64),
            "fpope_sigma": config["model"].get("fpope_sigma", 0.4),
            "fpope_training_length": config["model"].get("fpope_training_length", 512),
            "fpope_delta_init": config["model"].get("fpope_delta_init", "zero"),
            "fpope_d_rope": config["model"].get("fpope_d_rope", 32),
            "fpope_freeze_coeffs": config["model"].get("fpope_freeze_coeffs", False),
            "fpope_use_ceiling": config["model"].get("fpope_use_ceiling", False),
            "fpope_normalize_coeffs": config["model"].get("fpope_normalize_coeffs", True),
        })
        print_rank0(f"Using FPoPE positional encoding with theta={model_kwargs['fpope_theta']}, "
                    f"num_fourier_terms={model_kwargs['fpope_num_fourier_terms']}, "
                    f"freeze_coeffs={model_kwargs['fpope_freeze_coeffs']}, "
                    f"use_ceiling={model_kwargs['fpope_use_ceiling']}, "
                    f"normalize_coeffs={model_kwargs['fpope_normalize_coeffs']}")

    model_config = TransformerConfig(**model_kwargs)

    model = Transformer(model_config)
    model = model.to(device)

    print_rank0(f"Model parameters: {model_config.num_params:,}")
    print_rank0(f"Actual parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.compile:
        print_rank0("Compiling model with torch.compile...")
        model = torch.compile(model)

    if world_size > 1:
        model = wrap_model_ddp(
            model,
            local_rank,
            find_unused_parameters=config["distributed"]["find_unused_parameters"],
            bucket_cap_mb=config["distributed"]["bucket_cap_mb"],
        )

    print_rank0("Creating data loaders...")
    train_loader = create_dataloader(
        data_path=data_dir,
        seq_len=config["training"]["seq_len"],
        batch_size=config["training"]["batch_size"],
        split="train",
        num_workers=config["data"]["num_workers"],
        distributed=(world_size > 1),
        world_size=world_size,
        rank=rank,
    )

    val_loader = create_dataloader(
        data_path=data_dir,
        seq_len=config["training"]["seq_len"],
        batch_size=config["training"]["batch_size"],
        split="validation",
        num_workers=config["data"]["num_workers"],
        distributed=(world_size > 1),
        world_size=world_size,
        rank=rank,
    )

    checkpoint_dir = args.checkpoint_dir or config["training"]["checkpoint_dir"]
    wandb_run_name = args.wandb_run_name or config["training"]["wandb_run_name"]

    trainer_config = TrainerConfig(
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=tuple(config["training"]["betas"]),
        grad_clip=config["training"]["grad_clip"],
        warmup_steps=config["training"]["warmup_steps"],
        total_tokens=config["training"]["total_tokens"],
        batch_size=config["training"]["batch_size"],
        seq_len=config["training"]["seq_len"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        mixed_precision=config["training"]["mixed_precision"],
        validation_interval=config["training"]["validation_interval"],
        validation_samples=config["training"]["validation_samples"],
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=config["training"]["checkpoint_interval"],
        keep_last_n=config["training"]["keep_last_n"],
        log_interval=config["training"]["log_interval"],
        wandb_project=config["training"]["wandb_project"],
        wandb_run_name=wandb_run_name,
        loss_spike_threshold=config["training"]["loss_spike_threshold"],
        seed=config["training"]["seed"],
        gpu_peak_tflops=config["training"]["gpu_peak_tflops"],
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    print_rank0("Starting training...")
    trainer.train(model_config=model_config)

    cleanup_distributed()


if __name__ == "__main__":
    main()
