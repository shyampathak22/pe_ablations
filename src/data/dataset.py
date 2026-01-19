"""Pre-tokenized dataset with memory-mapped storage."""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

from src.data.tokenizer import get_tokenizer


class PreTokenizedDataset(Dataset):
    """Dataset for pre-tokenized data stored as memory-mapped numpy arrays.

    Supports random chunking into fixed-length sequences for efficient
    language model training.
    """

    def __init__(
        self,
        data_path: str | Path,
        seq_len: int,
        split: str = "train",
    ):
        """Initialize the dataset.

        Args:
            data_path: Path to the directory containing tokenized .bin files
            seq_len: Sequence length for training
            split: Dataset split ('train' or 'validation')
        """
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.split = split

        bin_file = self.data_path / f"{split}.bin"
        if not bin_file.exists():
            raise FileNotFoundError(
                f"Tokenized data not found at {bin_file}. "
                "Run pretokenize_dataset() first."
            )

        self.data = np.memmap(bin_file, dtype=np.uint16, mode="r")
        self.num_tokens = len(self.data)
        self.num_samples = (self.num_tokens - 1) // seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a training sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'input_ids' and 'labels' tensors
        """
        start = idx * self.seq_len
        end = start + self.seq_len + 1

        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))

        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }


def pretokenize_dataset(
    output_dir: str | Path,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    tokenizer_name: str = "gpt2",
    num_proc: int = 4,
) -> dict[str, int]:
    """Pre-tokenize a dataset and save as memory-mapped numpy arrays.

    Args:
        output_dir: Directory to save tokenized data
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration name
        tokenizer_name: Tokenizer to use
        num_proc: Number of processes for tokenization

    Returns:
        Dictionary with token counts for each split
    """
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(tokenizer_name)
    dataset = load_dataset(dataset_name, dataset_config)

    def tokenize_function(examples: dict) -> dict:
        return {"tokens": tokenizer(examples["text"], return_attention_mask=False)["input_ids"]}

    token_counts = {}

    for split in ["train", "validation"]:
        if split not in dataset:
            continue

        print(f"Tokenizing {split} split...")

        tokenized = dataset[split].map(
            tokenize_function,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset[split].column_names,
            desc=f"Tokenizing {split}",
        )

        all_tokens = []
        for example in tqdm(tokenized, desc=f"Collecting {split} tokens"):
            all_tokens.extend(example["tokens"])

        all_tokens = np.array(all_tokens, dtype=np.uint16)
        token_counts[split] = len(all_tokens)

        bin_file = output_dir / f"{split}.bin"
        memmap = np.memmap(bin_file, dtype=np.uint16, mode="w+", shape=all_tokens.shape)
        memmap[:] = all_tokens
        memmap.flush()

        print(f"{split}: {len(all_tokens):,} tokens saved to {bin_file}")

    return token_counts


def create_dataloader(
    data_path: str | Path,
    seq_len: int,
    batch_size: int,
    split: str = "train",
    num_workers: int = 4,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """Create a DataLoader for the pre-tokenized dataset.

    Args:
        data_path: Path to tokenized data directory
        seq_len: Sequence length
        batch_size: Batch size per GPU
        split: Dataset split
        num_workers: Number of data loading workers
        distributed: Whether to use distributed sampling
        world_size: Total number of processes
        rank: Current process rank

    Returns:
        Configured DataLoader
    """
    dataset = PreTokenizedDataset(data_path, seq_len, split)

    sampler = None
    shuffle = True

    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == "train"),
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
