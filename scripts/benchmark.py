"""Memory and speed profiling script."""

import argparse
import time

import torch
import yaml

from src.model import Transformer, TransformerConfig


def profile_memory(
    model: Transformer,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Profile memory usage during forward and backward pass."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    initial_memory = torch.cuda.memory_allocated() / 1e9

    with torch.autocast(device_type="cuda", dtype=dtype):
        output = model(input_ids, labels=labels)
        loss = output["loss"]

    forward_memory = torch.cuda.memory_allocated() / 1e9
    forward_peak = torch.cuda.max_memory_allocated() / 1e9

    loss.backward()

    backward_memory = torch.cuda.memory_allocated() / 1e9
    backward_peak = torch.cuda.max_memory_allocated() / 1e9

    return {
        "initial_memory_gb": initial_memory,
        "forward_memory_gb": forward_memory,
        "forward_peak_gb": forward_peak,
        "backward_memory_gb": backward_memory,
        "backward_peak_gb": backward_peak,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def profile_speed(
    model: Transformer,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    num_warmup: int = 5,
    num_iterations: int = 20,
) -> dict:
    """Profile training speed."""
    model.train()

    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)

    for _ in range(num_warmup):
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = model(input_ids, labels=labels)
            loss = output["loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = model(input_ids, labels=labels)
            loss = output["loss"]
        loss.backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time

    time_per_iter = elapsed_time / num_iterations
    tokens_per_sec = (batch_size * seq_len) / time_per_iter
    samples_per_sec = batch_size / time_per_iter

    return {
        "time_per_iter_ms": time_per_iter * 1000,
        "tokens_per_sec": tokens_per_sec,
        "samples_per_sec": samples_per_sec,
        "batch_size": batch_size,
        "seq_len": seq_len,
    }


def find_max_batch_size(
    model: Transformer,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    max_memory_fraction: float = 0.9,
) -> int:
    """Find maximum batch size that fits in GPU memory."""
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
    target_memory = total_memory * max_memory_fraction

    batch_size = 1
    max_batch_size = 1

    while True:
        try:
            torch.cuda.empty_cache()

            input_ids = torch.randint(
                0, model.config.vocab_size, (batch_size, seq_len), device=device
            )
            labels = torch.randint(
                0, model.config.vocab_size, (batch_size, seq_len), device=device
            )

            with torch.autocast(device_type="cuda", dtype=dtype):
                output = model(input_ids, labels=labels)
                loss = output["loss"]
            loss.backward()

            peak_memory = torch.cuda.max_memory_allocated() / 1e9

            if peak_memory < target_memory:
                max_batch_size = batch_size
                batch_size *= 2
            else:
                break

        except torch.cuda.OutOfMemoryError:
            break

        finally:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    low, high = max_batch_size, batch_size
    while low < high - 1:
        mid = (low + high) // 2
        try:
            torch.cuda.empty_cache()

            input_ids = torch.randint(
                0, model.config.vocab_size, (mid, seq_len), device=device
            )
            labels = torch.randint(
                0, model.config.vocab_size, (mid, seq_len), device=device
            )

            with torch.autocast(device_type="cuda", dtype=dtype):
                output = model(input_ids, labels=labels)
                loss = output["loss"]
            loss.backward()

            peak_memory = torch.cuda.max_memory_allocated() / 1e9

            if peak_memory < target_memory:
                low = mid
            else:
                high = mid

        except torch.cuda.OutOfMemoryError:
            high = mid

        finally:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    return low


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile PE ablations model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (auto-detect if not specified)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length (default: from config)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile",
    )
    parser.add_argument(
        "--grad-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config["training"]["mixed_precision"] == "bf16" else torch.float16

    print("Creating model...")
    pe_mode = config["model"].get("pe_mode", "rope")
    model_config = TransformerConfig(
        vocab_size=config["model"]["vocab_size"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        num_kv_heads=config["model"]["num_kv_heads"],
        head_dim=config["model"]["head_dim"],
        max_seq_len=config["model"]["max_seq_len"],
        pe_mode=pe_mode,
        # RoPE parameters
        rope_type=config["model"].get("rope_type", "standard"),
        rope_theta=config["model"].get("rope_theta", 10000.0),
        rope_scale=config["model"].get("rope_scale", 1.0),
        # FPoPE parameters
        fpope_theta=config["model"].get("fpope_theta", 10000.0),
        fpope_num_fourier_terms=config["model"].get("fpope_num_fourier_terms", 64),
        fpope_sigma=config["model"].get("fpope_sigma", 0.4),
        fpope_training_length=config["model"].get("fpope_training_length", 512),
        fpope_delta_init=config["model"].get("fpope_delta_init", "zero"),
        # Other parameters
        use_qk_norm=config["model"]["use_qk_norm"],
        gradient_checkpointing=args.grad_checkpointing or config["model"]["gradient_checkpointing"],
    )

    model = Transformer(model_config)
    model = model.to(device)

    print(f"Model parameters: {model_config.num_params:,}")
    print(f"PE mode: {model_config.pe_mode}", end="")
    if model_config.pe_mode == "rope":
        print(f" ({model_config.rope_type})")
    else:
        print(f" (fourier_terms={model_config.fpope_num_fourier_terms})")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    print(f"Mixed precision: {dtype}")
    print(f"Gradient checkpointing: {model_config.gradient_checkpointing}")

    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)

    seq_len = args.seq_len or config["training"]["seq_len"]

    print(f"\n{'='*60}")
    print(f"Finding maximum batch size for seq_len={seq_len}...")
    max_batch = find_max_batch_size(model, seq_len, device, dtype)
    print(f"Maximum batch size: {max_batch}")

    # Use 80% of max batch size for safe profiling (leave headroom)
    batch_size = args.batch_size or min(int(max_batch * 0.8), 64)
    batch_size = max(batch_size, 1)  # Ensure at least 1
    print(f"\n{'='*60}")
    print(f"Profiling with batch_size={batch_size}, seq_len={seq_len}")

    print("\nMemory profile:")
    mem_results = profile_memory(model, batch_size, seq_len, device, dtype)
    print(f"  Forward peak: {mem_results['forward_peak_gb']:.2f} GB")
    print(f"  Backward peak: {mem_results['backward_peak_gb']:.2f} GB")

    # Aggressive memory cleanup between profiles
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\nSpeed profile:")
    speed_results = profile_speed(model, batch_size, seq_len, device, dtype)
    print(f"  Time per iteration: {speed_results['time_per_iter_ms']:.1f} ms")
    print(f"  Tokens per second: {speed_results['tokens_per_sec']:,.0f}")
    print(f"  Samples per second: {speed_results['samples_per_sec']:.1f}")

    # Compute MFU
    from src.training.trainer import estimate_flops_per_token

    ffn_dim = model_config.ffn_dim
    if ffn_dim is None:
        ffn_dim = int(4 * model_config.hidden_dim * 2 / 3)
        ffn_dim = model_config.ffn_multiple_of * (
            (ffn_dim + model_config.ffn_multiple_of - 1) // model_config.ffn_multiple_of
        )

    flops_per_token = estimate_flops_per_token(
        num_layers=model_config.num_layers,
        hidden_dim=model_config.hidden_dim,
        num_heads=model_config.num_heads,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
        ffn_dim=ffn_dim,
        vocab_size=model_config.vocab_size,
        seq_len=seq_len,
    )

    gpu_peak_tflops = config["training"]["gpu_peak_tflops"]
    training_flops = speed_results['tokens_per_sec'] * 3 * flops_per_token
    mfu = training_flops / (gpu_peak_tflops * 1e12)

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Config: {args.config}")
    print(f"  Batch size: {batch_size} (max: {max_batch})")
    print(f"  Sequence length: {seq_len}")
    print(f"  Peak memory: {mem_results['backward_peak_gb']:.2f} GB")
    print(f"  Throughput: {speed_results['tokens_per_sec']:,.0f} tokens/sec")
    print(f"  MFU: {mfu:.1%} (vs {gpu_peak_tflops} TFLOPS peak)")

    if torch.cuda.device_count() > 1:
        multi_gpu_tps = speed_results['tokens_per_sec'] * torch.cuda.device_count()
        multi_gpu_flops = multi_gpu_tps * 3 * flops_per_token
        multi_gpu_mfu = multi_gpu_flops / (gpu_peak_tflops * 1e12 * torch.cuda.device_count())
        print(f"\nWith {torch.cuda.device_count()} GPUs (estimated):")
        print(f"  Effective batch size: {batch_size * torch.cuda.device_count()}")
        print(f"  Throughput: {multi_gpu_tps:,.0f} tokens/sec")
        print(f"  MFU: {multi_gpu_mfu:.1%}")


if __name__ == "__main__":
    main()
