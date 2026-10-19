"""Build verifier objects for strategies, with a lazy learned-verifier import."""

from __future__ import annotations

from typing import Any

from scaletimellm.strategies.verifiers import LengthPriorVerifier, OracleVerifier
from scaletimellm.tasks.base import Problem, Task


def build_verifier(
    name: str,
    *,
    task: Task,
    problem: Problem,
    checkpoint: str = "artifacts/verifier",
    target_tokens: int = 64,
) -> Any:
    """Resolve ``oracle``, ``length``, ``outcome`` or ``pairwise`` verifiers."""
    key = (name or "outcome").lower().strip()
    if key == "oracle":
        return OracleVerifier(task, problem)
    if key in {"length", "length_prior"}:
        return LengthPriorVerifier(target_tokens=target_tokens)
    if key in {"outcome", "pairwise", "learned"}:
        try:
            from scaletimellm.verifier.score import Verifier
        except ImportError as error:
            raise RuntimeError(
                "learned verifier is not available; train it with scripts/train_verifier.sh "
                "or pass verifier=oracle"
            ) from error
        objective = "pairwise" if key == "pairwise" else "outcome"
        return Verifier.from_checkpoint(checkpoint, objective=objective)
    raise ValueError(f"unknown verifier: {name!r}")
