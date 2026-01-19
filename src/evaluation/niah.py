"""Needle in a Haystack (NIAH) benchmark for context length testing."""

import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


@dataclass
class NIAHResult:
    """Result from a single NIAH test."""

    context_length: int
    needle_depth: float
    needle: str
    predicted: str
    score: float


class NIAHBenchmark:
    """Needle in a Haystack benchmark.

    Tests the model's ability to retrieve specific information ("needle")
    hidden at various depths within a larger context ("haystack").
    Results are typically visualized as a heatmap.
    """

    HAYSTACK_TEXT = (
        "The city of San Francisco is known for its iconic Golden Gate Bridge, "
        "diverse neighborhoods, and vibrant culture. Founded during the California "
        "Gold Rush, it has grown into a major center for technology and innovation. "
        "The city's famous cable cars still traverse its steep hills, offering "
        "breathtaking views of the bay. Fisherman's Wharf attracts tourists with "
        "fresh seafood and sea lion sightings. The Mission District showcases "
        "beautiful murals and authentic Mexican cuisine. Chinatown, one of the "
        "oldest in North America, offers a glimpse into rich Chinese-American heritage. "
    )

    NEEDLE_TEMPLATE = "The best thing to do in San Francisco is {activity}."
    QUESTION = "What is the best thing to do in San Francisco?"

    ACTIVITIES = [
        "eat a sandwich and sit in Dolores Park",
        "visit the Exploratorium museum",
        "take a ferry to Alcatraz Island",
        "walk across the Golden Gate Bridge",
        "explore the California Academy of Sciences",
    ]

    def __init__(
        self,
        tokenizer,
        num_samples: int = 1,
        depths: list[float] | None = None,
    ):
        """Initialize the benchmark.

        Args:
            tokenizer: Tokenizer for encoding/decoding
            num_samples: Number of samples per (context_length, depth) pair
            depths: List of relative depths (0.0 to 1.0) for needle placement
        """
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.depths = depths or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def _generate_haystack(self, num_tokens: int) -> str:
        """Generate haystack text with approximately the target number of tokens."""
        haystack_tokens = len(self.tokenizer.encode(self.HAYSTACK_TEXT))
        num_repeats = (num_tokens // haystack_tokens) + 1
        haystack = self.HAYSTACK_TEXT * num_repeats

        tokens = self.tokenizer.encode(haystack)[:num_tokens]
        return self.tokenizer.decode(tokens)

    def _create_sample(
        self,
        context_length: int,
        depth: float,
    ) -> tuple[str, str, str]:
        """Create a single test sample.

        Args:
            context_length: Total context length in tokens
            depth: Relative depth of needle (0.0 = start, 1.0 = end)

        Returns:
            Tuple of (prompt, expected_answer, needle_activity)
        """
        activity = random.choice(self.ACTIVITIES)
        needle = self.NEEDLE_TEMPLATE.format(activity=activity)

        needle_tokens = len(self.tokenizer.encode(needle))
        question_tokens = len(self.tokenizer.encode(self.QUESTION))
        haystack_tokens = context_length - needle_tokens - question_tokens - 10

        if haystack_tokens < 0:
            haystack_tokens = 100

        needle_pos = int(depth * haystack_tokens)

        before_haystack = self._generate_haystack(needle_pos)
        after_haystack = self._generate_haystack(haystack_tokens - needle_pos)

        prompt = f"{before_haystack} {needle} {after_haystack}\n\n{self.QUESTION}"

        return prompt, activity, activity

    def _score_response(self, predicted: str, expected: str) -> float:
        """Score the response (0.0 to 1.0)."""
        predicted_lower = predicted.lower()
        expected_lower = expected.lower()

        if expected_lower in predicted_lower:
            return 1.0

        expected_words = set(expected_lower.split())
        predicted_words = set(predicted_lower.split())
        overlap = len(expected_words & predicted_words)
        if overlap >= len(expected_words) * 0.5:
            return 0.5

        return 0.0

    @torch.no_grad()
    def evaluate(
        self,
        model,
        context_lengths: list[int],
        device: torch.device,
        max_new_tokens: int = 50,
    ) -> dict:
        """Run the benchmark.

        Args:
            model: Model to evaluate
            context_lengths: List of context lengths to test
            device: Device to run on
            max_new_tokens: Maximum tokens to generate for answer

        Returns:
            Dictionary with results and score matrices
        """
        model.eval()
        results: list[NIAHResult] = []

        for ctx_len in context_lengths:
            if ctx_len > model.config.max_seq_len:
                model.extend_context_length(ctx_len + max_new_tokens)

            for depth in tqdm(
                self.depths,
                desc=f"Context {ctx_len}",
                leave=False,
            ):
                for _ in range(self.num_samples):
                    prompt, expected, activity = self._create_sample(ctx_len, depth)

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

                    score = self._score_response(predicted, expected)

                    results.append(
                        NIAHResult(
                            context_length=ctx_len,
                            needle_depth=depth,
                            needle=activity,
                            predicted=predicted,
                            score=score,
                        )
                    )

        score_matrix = np.zeros((len(self.depths), len(context_lengths)))
        for i, depth in enumerate(self.depths):
            for j, ctx_len in enumerate(context_lengths):
                depth_ctx_results = [
                    r
                    for r in results
                    if r.needle_depth == depth and r.context_length == ctx_len
                ]
                if depth_ctx_results:
                    score_matrix[i, j] = np.mean([r.score for r in depth_ctx_results])

        return {
            "results": results,
            "score_matrix": score_matrix,
            "depths": self.depths,
            "context_lengths": context_lengths,
            "overall_score": np.mean([r.score for r in results]) if results else 0.0,
        }

    def plot_heatmap(
        self,
        eval_results: dict,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Plot the NIAH heatmap.

        Args:
            eval_results: Results from evaluate()
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        score_matrix = eval_results["score_matrix"]
        depths = eval_results["depths"]
        context_lengths = eval_results["context_lengths"]

        im = ax.imshow(
            score_matrix,
            cmap="RdYlGn",
            aspect="auto",
            vmin=0,
            vmax=1,
        )

        ax.set_xticks(range(len(context_lengths)))
        ax.set_xticklabels([f"{x // 1000}K" for x in context_lengths])
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([f"{d:.0%}" for d in depths])

        ax.set_xlabel("Context Length")
        ax.set_ylabel("Needle Depth")
        ax.set_title("Needle in a Haystack Performance")

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Retrieval Score")

        for i in range(len(depths)):
            for j in range(len(context_lengths)):
                text = ax.text(
                    j,
                    i,
                    f"{score_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black" if score_matrix[i, j] > 0.5 else "white",
                    fontsize=8,
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
