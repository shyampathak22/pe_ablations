"""Passkey retrieval benchmark for context length extrapolation testing."""

import random
import string
from dataclasses import dataclass

import torch
from tqdm import tqdm


@dataclass
class PasskeyResult:
    """Result from a single passkey retrieval test."""

    context_length: int
    passkey_position: float
    passkey: str
    predicted: str
    correct: bool


class PasskeyRetrievalBenchmark:
    """Passkey retrieval benchmark.

    Tests the model's ability to retrieve a hidden passkey from within
    a large context of irrelevant text. This is a simple but effective
    test for context length extrapolation.
    """

    FILLER_TEXT = (
        "The grass is green. The sky is blue. The sun is yellow. "
        "Here we go. There and back again. "
    )

    PROMPT_TEMPLATE = (
        "There is an important info hidden inside a lot of irrelevant text. "
        "Find it and memorize it. I will quiz you about the important information there.\n\n"
        "{context}\n\n"
        "What is the pass key? The pass key is "
    )

    PASSKEY_TEMPLATE = "The pass key is {passkey}. Remember it. {passkey} is the pass key."

    def __init__(
        self,
        tokenizer,
        num_samples: int = 10,
        passkey_length: int = 5,
        positions: list[float] | None = None,
    ):
        """Initialize the benchmark.

        Args:
            tokenizer: Tokenizer for encoding/decoding
            num_samples: Number of samples per (context_length, position) pair
            passkey_length: Length of the passkey (digits)
            positions: List of relative positions (0.0 to 1.0) to test
        """
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.passkey_length = passkey_length
        self.positions = positions or [0.0, 0.25, 0.5, 0.75, 1.0]

    def _generate_passkey(self) -> str:
        """Generate a random numeric passkey."""
        return "".join(random.choices(string.digits, k=self.passkey_length))

    def _generate_filler(self, num_tokens: int) -> str:
        """Generate filler text with approximately the target number of tokens."""
        filler_tokens = len(self.tokenizer.encode(self.FILLER_TEXT))
        num_repeats = (num_tokens // filler_tokens) + 1
        filler = self.FILLER_TEXT * num_repeats

        tokens = self.tokenizer.encode(filler)[:num_tokens]
        return self.tokenizer.decode(tokens)

    def _create_sample(
        self,
        context_length: int,
        position: float,
    ) -> tuple[str, str, str]:
        """Create a single test sample.

        Args:
            context_length: Total context length in tokens
            position: Relative position of passkey (0.0 to 1.0)

        Returns:
            Tuple of (prompt, passkey, full_text_for_debugging)
        """
        passkey = self._generate_passkey()
        passkey_text = self.PASSKEY_TEMPLATE.format(passkey=passkey)

        template_tokens = len(
            self.tokenizer.encode(
                self.PROMPT_TEMPLATE.format(context="") + passkey_text
            )
        )
        filler_tokens = context_length - template_tokens

        if filler_tokens < 0:
            filler_tokens = 0

        passkey_pos = int(position * filler_tokens)

        before_filler = self._generate_filler(passkey_pos)
        after_filler = self._generate_filler(filler_tokens - passkey_pos)

        context = before_filler + passkey_text + after_filler
        prompt = self.PROMPT_TEMPLATE.format(context=context)

        return prompt, passkey, prompt

    @torch.no_grad()
    def evaluate(
        self,
        model,
        context_lengths: list[int],
        device: torch.device,
        max_new_tokens: int = 10,
    ) -> dict:
        """Run the benchmark.

        Args:
            model: Model to evaluate
            context_lengths: List of context lengths to test
            device: Device to run on
            max_new_tokens: Maximum tokens to generate for answer

        Returns:
            Dictionary with results and accuracy metrics
        """
        model.eval()
        results: list[PasskeyResult] = []

        for ctx_len in context_lengths:
            if ctx_len > model.config.max_seq_len:
                model.extend_context_length(ctx_len + max_new_tokens)

            for position in tqdm(
                self.positions,
                desc=f"Context {ctx_len}",
                leave=False,
            ):
                for _ in range(self.num_samples):
                    prompt, passkey, _ = self._create_sample(ctx_len, position)

                    input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

                    if input_ids.shape[1] > ctx_len:
                        input_ids = input_ids[:, -ctx_len:]

                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                        top_k=1,
                    )

                    new_tokens = output_ids[0, input_ids.shape[1] :]
                    predicted = self.tokenizer.decode(new_tokens).strip()

                    predicted_digits = "".join(c for c in predicted if c.isdigit())
                    predicted_digits = predicted_digits[: self.passkey_length]

                    correct = predicted_digits == passkey

                    results.append(
                        PasskeyResult(
                            context_length=ctx_len,
                            passkey_position=position,
                            passkey=passkey,
                            predicted=predicted_digits,
                            correct=correct,
                        )
                    )

        accuracy_by_length = {}
        accuracy_by_position = {}

        for ctx_len in context_lengths:
            ctx_results = [r for r in results if r.context_length == ctx_len]
            accuracy_by_length[ctx_len] = (
                sum(r.correct for r in ctx_results) / len(ctx_results)
                if ctx_results
                else 0.0
            )

        for position in self.positions:
            pos_results = [r for r in results if r.passkey_position == position]
            accuracy_by_position[position] = (
                sum(r.correct for r in pos_results) / len(pos_results)
                if pos_results
                else 0.0
            )

        overall_accuracy = sum(r.correct for r in results) / len(results) if results else 0.0

        return {
            "results": results,
            "accuracy_by_length": accuracy_by_length,
            "accuracy_by_position": accuracy_by_position,
            "overall_accuracy": overall_accuracy,
        }
