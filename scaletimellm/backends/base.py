"""Language-model backend interface."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Generation:
    """One sampled completion and its accounting metadata."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    sample_index: int = 0
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens


class LMBackend(ABC):
    """Common interface implemented by HF, vLLM and mock backends.

    ``generate`` returns one inner list per input prompt.  The inner list has
    exactly ``n`` generations.  Implementations must add every consumed token
    to :attr:`cost_tokens` so strategies and analyses can audit accounting.
    """

    name: str = "base"

    def __init__(self, n_params: int, seed: int = 0) -> None:
        if n_params <= 0:
            raise ValueError("n_params must be positive")
        self.n_params = int(n_params)
        self.seed = int(seed)
        self.cost_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.num_calls = 0
        self.load_time_s = 0.0
        self._started = time.perf_counter()

    def reset_cost(self) -> None:
        """Reset cumulative token counters."""
        self.cost_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.num_calls = 0

    def _record(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += int(prompt_tokens)
        self.completion_tokens += int(completion_tokens)
        self.cost_tokens += int(prompt_tokens) + int(completion_tokens)
        self.num_calls += 1

    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 128,
        seed: int | None = None,
    ) -> list[list[Generation]]:
        """Generate ``n`` continuations for each prompt."""

    def info(self) -> dict[str, Any]:
        """Serialisable backend metadata for result records."""
        return {
            "backend": self.name,
            "n_params": self.n_params,
            "seed": self.seed,
            "cost_tokens": self.cost_tokens,
        }
