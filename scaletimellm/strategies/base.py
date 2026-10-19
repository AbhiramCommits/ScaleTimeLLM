"""Shared strategy interface and result container."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from scaletimellm.backends.base import Generation, LMBackend
from scaletimellm.tasks.base import Problem, Task
from scaletimellm.utils.flops import estimate_flops


@dataclass
class StrategyResult:
    """Outcome and exact accounting for one strategy on one problem."""

    answer: str | None
    is_correct: bool
    n_samples: int
    prompt_tokens: int
    completion_tokens: int
    flops: float
    wall_time_s: float
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens consumed by the strategy."""
        return self.prompt_tokens + self.completion_tokens


class Strategy(ABC):
    """Base class for inference-time scaling strategies."""

    name = "base"

    def __init__(
        self,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        verifier: Any | None = None,
        **_: Any,
    ) -> None:
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.verifier = verifier

    @abstractmethod
    def solve(
        self,
        problem: Problem,
        backend: LMBackend,
        task: Task,
        budget: int,
        seed: int = 0,
    ) -> StrategyResult:
        """Solve one problem within the prescribed sampling budget."""

    @staticmethod
    def _generations_to_trace(
        generations: list[Generation],
        answers: list[str | None] | None = None,
        scores: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        trace = []
        for index, generation in enumerate(generations):
            entry: dict[str, Any] = {
                "sample": index,
                "extracted": answers[index] if answers is not None else None,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "text": generation.text[:4000],
            }
            if scores is not None:
                entry["verifier_score"] = float(scores[index])
            trace.append(entry)
        return trace

    @staticmethod
    def _sum_tokens(generations: list[Generation]) -> tuple[int, int]:
        return (
            sum(generation.prompt_tokens for generation in generations),
            sum(generation.completion_tokens for generation in generations),
        )

    @staticmethod
    def _finalize(
        *,
        answer: str | None,
        problem: Problem,
        task: Task,
        generations: list[Generation],
        wall_time_s: float,
        trace: list[dict[str, Any]],
        n_params: int,
    ) -> StrategyResult:
        prompt_tokens, completion_tokens = Strategy._sum_tokens(generations)
        return StrategyResult(
            answer=answer,
            is_correct=bool(task.grade(problem, answer)),
            n_samples=len(generations),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            flops=estimate_flops(n_params, prompt_tokens, completion_tokens),
            wall_time_s=wall_time_s,
            trace=trace,
        )


def timed() -> float:
    """Small indirection so tests can monkeypatch wall time if needed."""
    return time.perf_counter()
