"""RULER benchmark wrapper for comprehensive context length evaluation."""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm


@dataclass
class RULERResult:
    """Result from a single RULER test."""

    task_name: str
    context_length: int
    query: str
    expected: str
    predicted: str
    correct: bool


class RULERBenchmark:
    """RULER benchmark for comprehensive context length testing.

    Implements a subset of RULER tasks focused on retrieval and
    simple reasoning at various context lengths.

    Reference: https://arxiv.org/abs/2404.06654
    """

    FILLER_SENTENCES = [
        "The weather today is quite pleasant with clear skies.",
        "Many people enjoy reading books in their spare time.",
        "Technology continues to evolve at a rapid pace.",
        "Healthy eating habits contribute to overall well-being.",
        "Exercise is important for maintaining physical fitness.",
        "Music has the power to evoke strong emotions.",
        "Learning new skills can be both challenging and rewarding.",
        "Travel opens up opportunities to experience different cultures.",
        "Good communication is essential in all relationships.",
        "Nature provides countless benefits to human health.",
    ]

    def __init__(
        self,
        tokenizer,
        tasks: list[str] | None = None,
    ):
        """Initialize the benchmark.

        Args:
            tokenizer: Tokenizer for encoding/decoding
            tasks: List of tasks to run. Default: all supported tasks.
        """
        self.tokenizer = tokenizer
        self.tasks = tasks or [
            "single_needle",
            "multi_needle",
            "kv_retrieval",
            "variable_tracking",
        ]

    def _generate_filler(self, num_tokens: int) -> str:
        """Generate filler text with approximately the target number of tokens."""
        filler = ""
        current_tokens = 0

        while current_tokens < num_tokens:
            sentence = random.choice(self.FILLER_SENTENCES)
            filler += " " + sentence
            current_tokens = len(self.tokenizer.encode(filler))

        tokens = self.tokenizer.encode(filler)[:num_tokens]
        return self.tokenizer.decode(tokens)

    def _create_single_needle_sample(
        self,
        context_length: int,
    ) -> tuple[str, str, str]:
        """Single needle retrieval task."""
        key = f"SPECIAL_KEY_{random.randint(1000, 9999)}"
        value = f"VALUE_{random.randint(10000, 99999)}"

        needle = f"The {key} is {value}."

        total_filler = context_length - len(self.tokenizer.encode(needle)) - 50
        pos = random.randint(0, max(1, total_filler))

        before = self._generate_filler(pos)
        after = self._generate_filler(total_filler - pos)

        context = f"{before} {needle} {after}"
        prompt = f"{context}\n\nQuestion: What is the {key}?\nAnswer:"

        return prompt, value, "single_needle"

    def _create_multi_needle_sample(
        self,
        context_length: int,
        num_needles: int = 3,
    ) -> tuple[str, str, str]:
        """Multi-needle retrieval task."""
        needles = []
        for i in range(num_needles):
            key = f"KEY_{i}_{random.randint(100, 999)}"
            value = f"VAL_{random.randint(1000, 9999)}"
            needles.append((key, value))

        target_idx = random.randint(0, num_needles - 1)
        target_key, target_value = needles[target_idx]

        total_filler = context_length - 100
        segment_size = total_filler // (num_needles + 1)

        context = ""
        for i, (key, value) in enumerate(needles):
            context += self._generate_filler(segment_size)
            context += f" The {key} equals {value}. "

        context += self._generate_filler(segment_size)

        prompt = f"{context}\n\nQuestion: What does {target_key} equal?\nAnswer:"

        return prompt, target_value, "multi_needle"

    def _create_kv_retrieval_sample(
        self,
        context_length: int,
        num_pairs: int = 10,
    ) -> tuple[str, str, str]:
        """Key-value retrieval task with multiple pairs."""
        pairs = {}
        for i in range(num_pairs):
            key = f"item_{i:03d}"
            value = str(random.randint(100, 999))
            pairs[key] = value

        kv_text = "Key-Value pairs:\n"
        for key, value in pairs.items():
            kv_text += f"- {key}: {value}\n"

        target_key = random.choice(list(pairs.keys()))
        target_value = pairs[target_key]

        remaining_tokens = context_length - len(self.tokenizer.encode(kv_text)) - 50
        filler = self._generate_filler(max(0, remaining_tokens))

        prompt = f"{kv_text}\n{filler}\n\nWhat is the value of {target_key}?\nAnswer:"

        return prompt, target_value, "kv_retrieval"

    def _create_variable_tracking_sample(
        self,
        context_length: int,
        num_updates: int = 5,
    ) -> tuple[str, str, str]:
        """Variable tracking task with multiple updates."""
        var_name = f"X_{random.randint(100, 999)}"
        updates = []
        current_value = random.randint(1, 100)

        for _ in range(num_updates):
            operation = random.choice(["add", "subtract", "multiply"])
            operand = random.randint(1, 10)

            if operation == "add":
                updates.append(f"{var_name} = {var_name} + {operand}")
                current_value += operand
            elif operation == "subtract":
                updates.append(f"{var_name} = {var_name} - {operand}")
                current_value -= operand
            else:
                updates.append(f"{var_name} = {var_name} * {operand}")
                current_value *= operand

        init_value = current_value
        for update in reversed(updates):
            if "+" in update:
                operand = int(update.split("+")[1])
                init_value -= operand
            elif "-" in update:
                operand = int(update.split("-")[1])
                init_value += operand
            else:
                operand = int(update.split("*")[1])
                init_value //= operand

        context = f"Let {var_name} = {init_value}\n"
        segment_size = (context_length - 100) // (num_updates + 1)

        for update in updates:
            context += self._generate_filler(segment_size)
            context += f"\n{update}\n"

        context += self._generate_filler(segment_size)

        prompt = f"{context}\n\nWhat is the final value of {var_name}?\nAnswer:"

        return prompt, str(current_value), "variable_tracking"

    def _create_sample(
        self,
        task: str,
        context_length: int,
    ) -> tuple[str, str, str]:
        """Create a sample for the specified task."""
        if task == "single_needle":
            return self._create_single_needle_sample(context_length)
        elif task == "multi_needle":
            return self._create_multi_needle_sample(context_length)
        elif task == "kv_retrieval":
            return self._create_kv_retrieval_sample(context_length)
        elif task == "variable_tracking":
            return self._create_variable_tracking_sample(context_length)
        else:
            raise ValueError(f"Unknown task: {task}")

    @torch.no_grad()
    def evaluate(
        self,
        model,
        context_lengths: list[int],
        device: torch.device,
        samples_per_task: int = 10,
        max_new_tokens: int = 20,
    ) -> dict:
        """Run the benchmark.

        Args:
            model: Model to evaluate
            context_lengths: List of context lengths to test
            device: Device to run on
            samples_per_task: Number of samples per (task, context_length) pair
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with results and accuracy by task/length
        """
        model.eval()
        results: list[RULERResult] = []

        for task in self.tasks:
            for ctx_len in tqdm(
                context_lengths,
                desc=f"Task: {task}",
                leave=False,
            ):
                if ctx_len > model.config.max_seq_len:
                    model.extend_context_length(ctx_len + max_new_tokens)

                for _ in range(samples_per_task):
                    prompt, expected, task_name = self._create_sample(task, ctx_len)

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

                    predicted_clean = "".join(
                        c for c in predicted.split()[0] if c.isalnum() or c == "_"
                    )
                    correct = expected.lower() in predicted.lower() or predicted_clean == expected

                    results.append(
                        RULERResult(
                            task_name=task_name,
                            context_length=ctx_len,
                            query=prompt[-200:],
                            expected=expected,
                            predicted=predicted,
                            correct=correct,
                        )
                    )

        accuracy_by_task = {}
        for task in self.tasks:
            task_results = [r for r in results if r.task_name == task]
            accuracy_by_task[task] = (
                sum(r.correct for r in task_results) / len(task_results)
                if task_results
                else 0.0
            )

        accuracy_by_length = {}
        for ctx_len in context_lengths:
            len_results = [r for r in results if r.context_length == ctx_len]
            accuracy_by_length[ctx_len] = (
                sum(r.correct for r in len_results) / len(len_results)
                if len_results
                else 0.0
            )

        accuracy_by_task_length = {}
        for task in self.tasks:
            accuracy_by_task_length[task] = {}
            for ctx_len in context_lengths:
                task_len_results = [
                    r
                    for r in results
                    if r.task_name == task and r.context_length == ctx_len
                ]
                accuracy_by_task_length[task][ctx_len] = (
                    sum(r.correct for r in task_len_results) / len(task_len_results)
                    if task_len_results
                    else 0.0
                )

        return {
            "results": results,
            "accuracy_by_task": accuracy_by_task,
            "accuracy_by_length": accuracy_by_length,
            "accuracy_by_task_length": accuracy_by_task_length,
            "overall_accuracy": sum(r.correct for r in results) / len(results) if results else 0.0,
        }
