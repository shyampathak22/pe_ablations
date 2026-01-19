"""Tokenizer wrapper for GPT-2 tokenizer."""

from transformers import AutoTokenizer, PreTrainedTokenizer


def get_tokenizer(name: str = "gpt2") -> PreTrainedTokenizer:
    """Get a tokenizer by name.

    Args:
        name: Tokenizer name (default: gpt2)

    Returns:
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
