"""Distributed Data Parallel (DDP) setup utilities."""

import os
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed() -> tuple[int, int, int]:
    """Initialize distributed training environment.

    Returns:
        Tuple of (rank, local_rank, world_size)
    """
    if not dist.is_initialized():
        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                rank=rank,
                world_size=world_size,
            )
        else:
            rank = 0
            local_rank = 0
            world_size = 1

            if torch.cuda.is_available():
                torch.cuda.set_device(0)
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get the rank of the current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get the total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM) -> torch.Tensor:
    """All-reduce a tensor across all processes.

    Args:
        tensor: Tensor to reduce
        op: Reduction operation (default: SUM)

    Returns:
        Reduced tensor
    """
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def wrap_model_ddp(
    model: torch.nn.Module,
    device_id: int,
    find_unused_parameters: bool = False,
    bucket_cap_mb: int = 50,
) -> torch.nn.Module:
    """Wrap a model with DistributedDataParallel.

    Args:
        model: Model to wrap
        device_id: CUDA device ID
        find_unused_parameters: Whether to detect unused parameters
        bucket_cap_mb: Bucket size for gradient reduction

    Returns:
        DDP-wrapped model (or original if not distributed)
    """
    if not dist.is_initialized():
        return model

    return DDP(
        model,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=find_unused_parameters,
        bucket_cap_mb=bucket_cap_mb,
    )


def print_rank0(*args: Any, **kwargs: Any) -> None:
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)
