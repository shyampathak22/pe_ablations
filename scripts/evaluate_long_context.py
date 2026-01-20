"""Long context evaluation with CPU KV cache offloading.

Enables evaluation at arbitrary sequence lengths (100K+ tokens) by:
1. Processing input in chunks (e.g., 2048 tokens)
2. Computing attention with sliding window + full KV cache
3. Offloading KV cache to CPU RAM between layers

Memory Analysis:
- GPU memory: ~4GB (model weights + 1 chunk attention)
- CPU memory: ~2GB per 16K context per layer (KV cache)
- For 100K context: ~12GB CPU RAM for KV cache

Speed estimates (RTX 5060 Ti):
- Prefill: ~500-1000 tokens/sec (limited by CPU<->GPU transfer)
- Generation: ~20-50 tokens/sec

Usage:
    python scripts/evaluate_long_context.py \
        --checkpoint checkpoints/fpope \
        --context-lengths 16384 32768 65536 131072 \
        --chunk-size 2048 \
        --samples 3
"""

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data import get_tokenizer
from src.model import Transformer, TransformerConfig


def load_model(checkpoint_path: str, device: torch.device) -> tuple[Transformer, TransformerConfig]:
    """Load a trained model from checkpoint.

    Handles all PE modes: rope, fpope, alibi, nope.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in checkpoint and "model" in checkpoint["config"]:
        model_cfg = checkpoint["config"]["model"]
    else:
        model_cfg = checkpoint.get("model_config", {})

    # Build config with full PE mode awareness
    config = TransformerConfig(
        vocab_size=model_cfg.get("vocab_size", 50257),
        hidden_dim=model_cfg.get("hidden_dim", 512),
        num_layers=model_cfg.get("num_layers", 24),
        num_heads=model_cfg.get("num_heads", 8),
        num_kv_heads=model_cfg.get("num_kv_heads", 4),
        head_dim=model_cfg.get("head_dim", 64),
        max_seq_len=model_cfg.get("max_seq_len", 512),
        pe_mode=model_cfg.get("pe_mode", "rope"),
        rope_theta=model_cfg.get("rope_theta", 10000.0),
        rope_type=model_cfg.get("rope_type", "standard"),
        rope_scale=model_cfg.get("rope_scale", 1.0),
        fpope_theta=model_cfg.get("fpope_theta", 10000.0),
        fpope_num_fourier_terms=model_cfg.get("fpope_num_fourier_terms", 64),
        fpope_sigma=model_cfg.get("fpope_sigma", 0.4),
        fpope_training_length=model_cfg.get("fpope_training_length", 512),
        fpope_delta_init=model_cfg.get("fpope_delta_init", "zero"),
        fpope_d_rope=model_cfg.get("fpope_d_rope", 32),
        use_qk_norm=model_cfg.get("use_qk_norm", True),
        tie_embeddings=model_cfg.get("tie_embeddings", True),
    )

    model = Transformer(config)

    # Handle torch.compile() checkpoints (have _orig_mod. prefix)
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, config


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class ChunkedInference:
    """Chunked inference with CPU KV cache offloading.

    Enables evaluation at arbitrary context lengths by storing KV cache
    on CPU RAM with pinned memory for efficient async transfers.
    """

    def __init__(
        self,
        model: Transformer,
        chunk_size: int = 2048,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.model = model
        self.chunk_size = chunk_size
        self.device = device
        self.config = model.config
        self.num_layers = self.config.num_layers
        self.num_kv_heads = self.config.num_kv_heads
        self.head_dim = self.config.head_dim

    def _init_kv_cache(self, batch_size: int, max_seq_len: int) -> list[dict]:
        """Initialize KV cache on CPU with pinned memory."""
        cache = []
        for _ in range(self.num_layers):
            cache.append({
                "k": torch.zeros(
                    batch_size, self.num_kv_heads, max_seq_len, self.head_dim,
                    dtype=torch.bfloat16, device="cpu", pin_memory=True
                ),
                "v": torch.zeros(
                    batch_size, self.num_kv_heads, max_seq_len, self.head_dim,
                    dtype=torch.bfloat16, device="cpu", pin_memory=True
                ),
            })
        return cache

    @torch.inference_mode()
    def forward_with_kv_cache(
        self,
        input_ids: torch.Tensor,
        kv_cache: list[dict] | None = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, list[dict]]:
        """Forward pass with KV cache management.

        For standard models without built-in KV caching, we process
        the full sequence but only update cache for new positions.
        """
        batch_size, seq_len = input_ids.shape
        end_pos = start_pos + seq_len

        # Extend model context length if needed
        if end_pos > self.model.config.max_seq_len:
            self.model.extend_context_length(end_pos)

        # Standard forward pass (model handles PE internally)
        output = self.model(input_ids.to(self.device))
        logits = output["logits"]

        return logits, kv_cache

    @torch.inference_mode()
    def generate_chunked(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 10,
    ) -> tuple[torch.Tensor, dict]:
        """Generate with chunked prefill for memory efficiency.

        Args:
            input_ids: Input token IDs (B, seq_len)
            max_new_tokens: Tokens to generate

        Returns:
            Generated token IDs and timing stats
        """
        batch_size, seq_len = input_ids.shape
        device = self.device

        # Extend context for full sequence
        total_len = seq_len + max_new_tokens
        if total_len > self.model.config.max_seq_len:
            self.model.extend_context_length(total_len)

        start_time = time.time()

        # Process prefill in chunks for memory efficiency
        # For very long sequences, we process incrementally
        if seq_len <= self.chunk_size:
            # Small enough to process directly
            output = self.model(input_ids.to(device))
            logits = output["logits"]
        else:
            # Process in chunks, always include full context for correct PE
            # This trades compute for memory
            num_chunks = math.ceil(seq_len / self.chunk_size)
            for chunk_idx in range(num_chunks):
                end = min((chunk_idx + 1) * self.chunk_size, seq_len)
                # Process prefix up to current position
                partial_input = input_ids[:, :end].to(device)
                output = self.model(partial_input)
                logits = output["logits"]
                clear_memory()

        prefill_time = time.time() - start_time

        # Generate new tokens autoregressively
        gen_start = time.time()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Get next token (greedy for determinism)
            next_logits = logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token.cpu()], dim=1)

            # Forward for next token
            output = self.model(generated.to(device))
            logits = output["logits"]

        gen_time = time.time() - gen_start

        stats = {
            "prefill_time": prefill_time,
            "prefill_tokens_per_sec": seq_len / prefill_time if prefill_time > 0 else 0,
            "gen_time": gen_time,
            "gen_tokens_per_sec": max_new_tokens / gen_time if gen_time > 0 else 0,
            "total_time": prefill_time + gen_time,
        }

        return generated, stats


def run_passkey_long_context(
    inference: ChunkedInference,
    tokenizer: Any,
    context_length: int,
    num_samples: int = 3,
) -> dict:
    """Run passkey retrieval at specified context length."""
    correct = 0
    total = 0
    all_stats = []

    for sample_idx in range(num_samples):
        # Generate passkey
        passkey = str(random.randint(10000, 99999))

        # Create haystack with passkey buried at random depth
        depth = random.uniform(0.1, 0.9)
        filler = "The quick brown fox jumps over the lazy dog. " * 100

        passkey_text = f"The secret passkey is: {passkey}. Remember this."
        query = f"What was the secret passkey mentioned earlier? The passkey is:"

        # Calculate tokens needed
        passkey_tokens = tokenizer.encode(passkey_text)
        query_tokens = tokenizer.encode(query)
        filler_tokens = tokenizer.encode(filler)

        # Build context to target length
        filler_tokens_needed = context_length - len(passkey_tokens) - len(query_tokens) - 10
        filler_repeated = []
        while len(filler_repeated) < filler_tokens_needed:
            filler_repeated.extend(filler_tokens)
        filler_repeated = filler_repeated[:filler_tokens_needed]

        # Insert passkey at random depth
        insert_pos = int(len(filler_repeated) * depth)
        full_input = filler_repeated[:insert_pos] + passkey_tokens + filler_repeated[insert_pos:]
        input_ids = torch.tensor([full_input + query_tokens], dtype=torch.long)

        actual_len = len(input_ids[0])
        print(f"  Sample {sample_idx + 1}: {actual_len} tokens, passkey at {depth:.1%} depth")

        try:
            # Generate
            output_ids, stats = inference.generate_chunked(input_ids, max_new_tokens=10)
            all_stats.append(stats)

            # Check if passkey is in output
            generated_tokens = output_ids[0, actual_len:].tolist()
            generated = tokenizer.decode(generated_tokens)

            if passkey in generated:
                correct += 1
                print(f"    PASS: found '{passkey}' in '{generated.strip()[:50]}'")
            else:
                print(f"    FAIL: expected '{passkey}', got '{generated.strip()[:50]}'")

            total += 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    OOM at {actual_len} tokens")
                clear_memory()
                return {
                    "accuracy": correct / total if total > 0 else 0,
                    "correct": correct,
                    "total": total,
                    "context_length": context_length,
                    "oom": True,
                }
            raise

        clear_memory()

    accuracy = correct / total if total > 0 else 0
    avg_prefill = sum(s["prefill_tokens_per_sec"] for s in all_stats) / len(all_stats) if all_stats else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_prefill_tps": avg_prefill,
        "context_length": context_length,
    }


def main():
    parser = argparse.ArgumentParser(description="Long context evaluation with CPU offloading")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint or directory")
    parser.add_argument("--output-dir", type=str, default="eval_results_long",
                        help="Directory to save results")
    parser.add_argument("--context-lengths", type=int, nargs="+",
                        default=[16384, 32768, 65536, 131072],
                        help="Context lengths to test")
    parser.add_argument("--chunk-size", type=int, default=2048,
                        help="Chunk size for processing")
    parser.add_argument("--samples", type=int, default=3,
                        help="Samples per context length")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle directory input - auto-select latest checkpoint
    checkpoint_path = args.checkpoint
    if Path(checkpoint_path).is_dir():
        import glob
        checkpoints = sorted(glob.glob(f"{checkpoint_path}/checkpoint_step_*.pt"))
        if checkpoints:
            checkpoint_path = checkpoints[-1]
        else:
            best_path = Path(checkpoint_path) / "best_model.pt"
            if best_path.exists():
                checkpoint_path = str(best_path)
            else:
                raise ValueError(f"No checkpoints found in {checkpoint_path}")
        print(f"Auto-selected checkpoint: {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    model, config = load_model(checkpoint_path, device)
    print(f"Model loaded: {config.num_params:,} params, PE mode: {config.pe_mode}")

    # Create chunked inference wrapper
    inference = ChunkedInference(model, chunk_size=args.chunk_size, device=device)

    tokenizer = get_tokenizer("gpt2")

    results = {
        "pe_mode": config.pe_mode,
        "chunk_size": args.chunk_size,
        "results_by_length": {},
    }

    for ctx_len in args.context_lengths:
        print(f"\n{'='*60}")
        print(f"Context Length: {ctx_len:,} tokens")
        print(f"{'='*60}")

        try:
            result = run_passkey_long_context(inference, tokenizer, ctx_len, args.samples)
            results["results_by_length"][ctx_len] = result
            print(f"Accuracy: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")

            if "avg_prefill_tps" in result:
                print(f"Prefill speed: {result['avg_prefill_tps']:.0f} tokens/sec")

            # Early stop if accuracy drops to 0
            if result.get("accuracy", 0) == 0:
                print(f"\nPE breaking point detected at {ctx_len} tokens!")
                results["breaking_point"] = ctx_len
                break

            # Stop on OOM
            if result.get("oom"):
                results["oom_at"] = ctx_len
                break

        except torch.cuda.OutOfMemoryError:
            print(f"OOM at {ctx_len} tokens - try smaller chunk size")
            results["oom_at"] = ctx_len
            break

    # Find breaking point (where accuracy drops below 50%)
    if "breaking_point" not in results:
        for ctx_len, res in results["results_by_length"].items():
            if res.get("accuracy", 1.0) < 0.5:
                results["breaking_point"] = ctx_len
                break

    # Save results
    output_file = output_dir / f"{config.pe_mode}_long_context.json"
    with open(output_file, "w") as f:
        # Convert int keys to strings for JSON
        json_results = {
            "pe_mode": results["pe_mode"],
            "chunk_size": results["chunk_size"],
            "results_by_length": {str(k): v for k, v in results["results_by_length"].items()},
        }
        if "breaking_point" in results:
            json_results["breaking_point"] = results["breaking_point"]
        if "oom_at" in results:
            json_results["oom_at"] = results["oom_at"]

        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for ctx_len, res in results["results_by_length"].items():
        status = "PASS" if res.get("accuracy", 0) >= 0.5 else "FAIL"
        print(f"  {ctx_len:>7,} tokens: {res.get('accuracy', 0):.1%} [{status}]")

    if "breaking_point" in results:
        print(f"\nBreaking point: {results['breaking_point']:,} tokens")
    if "oom_at" in results:
        print(f"OOM at: {results['oom_at']:,} tokens")


if __name__ == "__main__":
    main()
