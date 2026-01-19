"""Training loop with logging, checkpointing, and validation."""

import math
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.distributed import (
    all_reduce,
    barrier,
    get_rank,
    get_world_size,
    is_main_process,
    print_rank0,
)
from src.training.optimizer import create_optimizer, create_scheduler, get_grad_norm


def estimate_flops_per_token(
    num_layers: int,
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    ffn_dim: int,
    vocab_size: int,
    seq_len: int,
) -> int:
    """Estimate FLOPs per token for forward pass.

    Based on the Chinchilla/PaLM methodology for transformer FLOPs estimation.
    For training (fwd + bwd + optimizer), multiply by ~3.
    """
    # Embedding lookup: negligible FLOPs

    # Per-layer attention FLOPs:
    # Q projection: 2 * hidden * (num_heads * head_dim)
    # K projection: 2 * hidden * (num_kv_heads * head_dim)
    # V projection: 2 * hidden * (num_kv_heads * head_dim)
    # Attention scores: 2 * seq_len * (num_heads * head_dim)
    # Attention output: 2 * seq_len * (num_heads * head_dim)
    # Output projection: 2 * (num_heads * head_dim) * hidden
    qkv_flops = 2 * hidden_dim * (num_heads + 2 * num_kv_heads) * head_dim
    attn_flops = 4 * seq_len * num_heads * head_dim  # scores + weighted sum
    out_proj_flops = 2 * num_heads * head_dim * hidden_dim
    attn_total = qkv_flops + attn_flops + out_proj_flops

    # Per-layer FFN FLOPs (SwiGLU has 3 projections):
    # Up: 2 * hidden * ffn_dim
    # Gate: 2 * hidden * ffn_dim
    # Down: 2 * ffn_dim * hidden
    ffn_total = 6 * hidden_dim * ffn_dim

    # Total per layer
    layer_flops = attn_total + ffn_total

    # All layers
    total_flops = num_layers * layer_flops

    # Output projection: 2 * hidden * vocab
    total_flops += 2 * hidden_dim * vocab_size

    return total_flops


@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 1000

    # Training
    total_tokens: int = 100_000_000
    batch_size: int = 64
    seq_len: int = 512
    gradient_accumulation_steps: int = 1

    # Mixed precision
    mixed_precision: str = "bf16"

    # Validation
    validation_interval: int = 500
    validation_samples: int = 1000

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 1000
    keep_last_n: int = 3

    # Logging
    log_interval: int = 10
    wandb_project: str = "pe-ablations"
    wandb_run_name: str | None = None

    # Robustness
    loss_spike_threshold: float = 10.0
    seed: int = 42

    # Hardware (for MFU calculation)
    # Peak TFLOPS for your GPU in BF16/FP16
    # Examples: A100=312, H100=989, RTX 4090=165, RTX 5060 Ti=23.7
    gpu_peak_tflops: float = 23.7


class Trainer:
    """Training loop with all bells and whistles."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: TrainerConfig,
        device: torch.device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        tokens_per_step = (
            config.batch_size
            * config.seq_len
            * get_world_size()
            * config.gradient_accumulation_steps
        )
        self.total_steps = config.total_tokens // tokens_per_step

        self.optimizer = create_optimizer(
            model,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
        self.scheduler = create_scheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=self.total_steps,
        )

        self.scaler = None
        if config.mixed_precision == "fp16":
            self.scaler = GradScaler()

        self.step = 0
        self.tokens_seen = 0
        self.epoch = 0

        self.loss_history: deque[float] = deque(maxlen=100)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.wandb_run = None
        self.model_config = None  # Set in train() for FLOPs calculation
        self.flops_per_token = None

    def _init_wandb(self, model_config: Any) -> None:
        """Initialize WandB logging."""
        if not is_main_process():
            return

        try:
            import wandb

            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "model": model_config.__dict__ if hasattr(model_config, "__dict__") else model_config,
                    "training": self.config.__dict__,
                },
                resume="allow",
            )
        except ImportError:
            print_rank0("WandB not available, skipping logging")

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to WandB."""
        if self.wandb_run is not None:
            import wandb
            wandb.log(metrics, step=self.step)

    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save a training checkpoint."""
        if not is_main_process():
            return

        checkpoint = {
            "model_state_dict": self.model.module.state_dict()
            if hasattr(self.model, "module")
            else self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "epoch": self.epoch,
            "config": self.config.__dict__,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        if self.wandb_run is not None:
            import wandb
            checkpoint["wandb_run_id"] = wandb.run.id

        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.step}.pt"
        torch.save(checkpoint, checkpoint_path)
        print_rank0(f"Saved checkpoint to {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        for ckpt in checkpoints[: -self.config.keep_last_n]:
            ckpt.unlink()

    def load_checkpoint(self, checkpoint_path: str | Path | None = None) -> bool:
        """Load a checkpoint to resume training.

        Args:
            checkpoint_path: Path to checkpoint. If None, loads latest.

        Returns:
            True if checkpoint was loaded, False otherwise.
        """
        if checkpoint_path is None:
            checkpoints = sorted(
                self.checkpoint_dir.glob("checkpoint_step_*.pt"),
                key=lambda p: int(p.stem.split("_")[-1]),
            )
            if not checkpoints:
                return False
            checkpoint_path = checkpoints[-1]

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return False

        print_rank0(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        model = self.model.module if hasattr(self.model, "module") else self.model
        model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.step = checkpoint["step"]
        self.tokens_seen = checkpoint["tokens_seen"]
        self.epoch = checkpoint.get("epoch", 0)

        print_rank0(f"Resumed from step {self.step}, tokens seen: {self.tokens_seen:,}")
        return True

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation and return metrics."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        max_batches = self.config.validation_samples // self.config.batch_size

        for batch in self.val_loader:
            if num_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16,
                enabled=self.config.mixed_precision in ["bf16", "fp16"],
            ):
                output = self.model(input_ids, labels=labels)
                loss = output["loss"]

            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            self.model.train()
            return {}

        avg_loss = total_loss / num_batches

        if get_world_size() > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            all_reduce(loss_tensor)
            avg_loss = loss_tensor.item() / get_world_size()

        self.model.train()

        return {
            "val/loss": avg_loss,
            "val/perplexity": math.exp(min(avg_loss, 20)),
        }

    def _check_loss_spike(self, loss: float) -> bool:
        """Check if loss has spiked abnormally."""
        if len(self.loss_history) < 10:
            return False

        avg_loss = sum(self.loss_history) / len(self.loss_history)
        return loss > avg_loss * self.config.loss_spike_threshold

    def _train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute a single training step."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16,
            enabled=self.config.mixed_precision in ["bf16", "fp16"],
        ):
            output = self.model(input_ids, labels=labels)
            loss = output["loss"] / self.config.gradient_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {"loss": loss.item() * self.config.gradient_accumulation_steps}

    def _optimizer_step(self) -> float:
        """Execute optimizer step with gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip,
        )

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

    def _compute_mfu(self, tokens_per_sec: float) -> float:
        """Compute Model FLOPs Utilization."""
        if self.flops_per_token is None:
            return 0.0

        # Training FLOPs = 3x forward (fwd + bwd + optimizer)
        training_flops_per_token = 3 * self.flops_per_token
        achieved_flops = tokens_per_sec * training_flops_per_token

        # Peak FLOPs across all GPUs
        peak_flops = self.config.gpu_peak_tflops * 1e12 * get_world_size()

        return achieved_flops / peak_flops

    def train(self, model_config: Any = None) -> None:
        """Run the full training loop."""
        self._init_wandb(model_config)
        self.model_config = model_config

        # Compute FLOPs per token for MFU calculation
        if model_config is not None and hasattr(model_config, "num_layers"):
            ffn_dim = model_config.ffn_dim
            if ffn_dim is None:
                ffn_dim = int(4 * model_config.hidden_dim * 2 / 3)
                ffn_dim = model_config.ffn_multiple_of * (
                    (ffn_dim + model_config.ffn_multiple_of - 1) // model_config.ffn_multiple_of
                )

            self.flops_per_token = estimate_flops_per_token(
                num_layers=model_config.num_layers,
                hidden_dim=model_config.hidden_dim,
                num_heads=model_config.num_heads,
                num_kv_heads=model_config.num_kv_heads,
                head_dim=model_config.head_dim,
                ffn_dim=ffn_dim,
                vocab_size=model_config.vocab_size,
                seq_len=self.config.seq_len,
            )
            print_rank0(f"FLOPs per token (fwd): {self.flops_per_token / 1e9:.2f}B")

        self.load_checkpoint()

        self.model.train()
        tokens_per_batch = self.config.batch_size * self.config.seq_len * get_world_size()

        print_rank0(f"Starting training from step {self.step}")
        print_rank0(f"Total steps: {self.total_steps:,}")
        print_rank0(f"Tokens per step: {tokens_per_batch * self.config.gradient_accumulation_steps:,}")
        print_rank0(f"GPU peak TFLOPS: {self.config.gpu_peak_tflops} x {get_world_size()} GPUs")

        start_time = time.time()
        step_start_time = time.time()
        accumulated_loss = 0.0
        accumulation_count = 0

        data_iter = iter(self.train_loader)

        pbar = tqdm(
            initial=self.step,
            total=self.total_steps,
            desc="Training",
            disable=not is_main_process(),
        )

        while self.step < self.total_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                if hasattr(self.train_loader.sampler, "set_epoch"):
                    self.train_loader.sampler.set_epoch(self.epoch)
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            step_metrics = self._train_step(batch)
            accumulated_loss += step_metrics["loss"]
            accumulation_count += 1

            if accumulation_count >= self.config.gradient_accumulation_steps:
                grad_norm = self._optimizer_step()
                self.step += 1
                self.tokens_seen += (
                    tokens_per_batch * self.config.gradient_accumulation_steps
                )

                avg_loss = accumulated_loss / accumulation_count
                self.loss_history.append(avg_loss)

                if torch.isnan(torch.tensor(avg_loss)):
                    print_rank0("NaN loss detected! Stopping training.")
                    break

                if self._check_loss_spike(avg_loss):
                    print_rank0(f"Warning: Loss spike detected at step {self.step}: {avg_loss:.4f}")

                if self.step % self.config.log_interval == 0:
                    step_time = time.time() - step_start_time
                    tokens_per_sec = (
                        tokens_per_batch
                        * self.config.gradient_accumulation_steps
                        * self.config.log_interval
                        / step_time
                    )
                    mfu = self._compute_mfu(tokens_per_sec)

                    metrics = {
                        "train/loss": avg_loss,
                        "train/perplexity": math.exp(min(avg_loss, 20)),
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "train/grad_norm": grad_norm,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/mfu": mfu,
                        "train/gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                        "train/step": self.step,
                        "train/tokens_seen": self.tokens_seen,
                        "train/epoch": self.epoch,
                    }
                    self._log_metrics(metrics)

                    # Format TPS for display (e.g., 125.4K)
                    if tokens_per_sec >= 1e6:
                        tps_str = f"{tokens_per_sec / 1e6:.1f}M"
                    elif tokens_per_sec >= 1e3:
                        tps_str = f"{tokens_per_sec / 1e3:.1f}K"
                    else:
                        tps_str = f"{tokens_per_sec:.0f}"

                    pbar.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        tps=tps_str,
                        mfu=f"{mfu:.1%}",
                    )

                    step_start_time = time.time()

                if (
                    self.step % self.config.validation_interval == 0
                    and self.val_loader is not None
                ):
                    val_metrics = self.validate()
                    self._log_metrics(val_metrics)
                    print_rank0(
                        f"Step {self.step}: val_loss={val_metrics.get('val/loss', 0):.4f}, "
                        f"val_ppl={val_metrics.get('val/perplexity', 0):.2f}"
                    )

                if self.step % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()

                accumulated_loss = 0.0
                accumulation_count = 0
                pbar.update(1)

        pbar.close()

        self._save_checkpoint()

        total_time = time.time() - start_time
        print_rank0(f"\nTraining complete!")
        print_rank0(f"Total time: {total_time / 3600:.2f} hours")
        print_rank0(f"Final tokens seen: {self.tokens_seen:,}")
        print_rank0(f"Average tokens/sec: {self.tokens_seen / total_time:.0f}")

        if self.wandb_run is not None:
            import wandb
            wandb.finish()
