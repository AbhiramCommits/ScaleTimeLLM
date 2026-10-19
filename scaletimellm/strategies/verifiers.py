"""Verifier adapters used by selection and weighted-voting strategies."""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from scaletimellm.tasks.base import Problem, Task


@runtime_checkable
class VerifierLike(Protocol):
    """Protocol for a scalar completion scorer."""

    def score(self, prompt: str, completion: str) -> float:
        """Return a higher-is-better verifier score (normally a logit)."""


class OracleVerifier:
    """An upper-bound verifier that calls the task grader.

    This is not available at inference time in practice; it is used only for
    headroom analysis.
    """

    name = "oracle"

    def __init__(self, task: Task, problem: Problem) -> None:
        self.task = task
        self.problem = problem
        self.calls = 0

    def score(self, prompt: str, completion: str) -> float:
        del prompt
        self.calls += 1
        answer = self.task.extract(completion)
        return 1.0 if self.task.grade(self.problem, answer) else 0.0

    def score_many(self, prompt: str, completions: list[str]) -> list[float]:
        """Batched oracle scoring."""
        return [self.score(prompt, completion) for completion in completions]


class LengthPriorVerifier:
    """A baseline that prefers completions near a target length."""

    name = "length"

    def __init__(self, target_tokens: int = 64) -> None:
        self.target_tokens = max(1, int(target_tokens))

    def score(self, prompt: str, completion: str) -> float:
        del prompt
        tokens = max(1, len(completion.split()))
        return -abs(tokens - self.target_tokens) / self.target_tokens

    def score_many(self, prompt: str, completions: list[str]) -> list[float]:
        """Batched length scoring."""
        return [self.score(prompt, completion) for completion in completions]


class ConstantVerifier:
    """Fallback scorer that assigns every completion the same value."""

    name = "constant"

    def score(self, prompt: str, completion: str) -> float:
        del prompt, completion
        return 0.0

    def score_many(self, prompt: str, completions: list[str]) -> list[float]:
        """Batched constant scoring."""
        return [0.0 for _ in completions]


def score_many(verifier: VerifierLike, prompt: str, completions: list[str]) -> list[float]:
    """Use a batch method if available, otherwise call ``score`` per item."""
    if hasattr(verifier, "score_many"):
        return [float(value) for value in verifier.score_many(prompt, completions)]
    return [float(verifier.score(prompt, completion)) for completion in completions]


def softmax_weights(scores: list[float]) -> list[float]:
    """Stable softmax over verifier scores."""
    if not scores:
        return []
    maximum = max(scores)
    exps = [math.exp(score - maximum) for score in scores]
    total = sum(exps)
    if total <= 0 or not math.isfinite(total):
        return [1.0 / len(scores)] * len(scores)
    return [value / total for value in exps]
