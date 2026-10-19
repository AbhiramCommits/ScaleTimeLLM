"""Task interface and problem container."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Problem:
    """A single evaluation problem.

    ``gold`` is always the reference answer as a string.  Extra fields used by a
    grader (unit tests, chart data, image handle) live in ``metadata``.  ``text``
    is the human-readable question; ``prompt_override`` lets a task provide a
    fully rendered prompt (for example an ASCII chart).
    """

    id: str
    text: str
    gold: str
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_override: str | None = None
    difficulty: str | None = None
    is_correct: bool | None = None
    difficulty_pass_rate: float | None = None


class Task(ABC):
    """Common interface implemented by every task/grader."""

    name: str = "base"
    modality: str = "text"

    @abstractmethod
    def load(self, n_problems: int, seed: int) -> list[Problem]:
        """Load a deterministic slice of ``n_problems`` problems."""

    @abstractmethod
    def prompt(self, problem: Problem) -> str:
        """Render the model prompt for a problem."""

    @abstractmethod
    def extract(self, completion: str) -> str | None:
        """Extract a comparable answer from a raw completion."""

    @abstractmethod
    def grade(self, problem: Problem, answer: str | None) -> bool:
        """Return whether ``answer`` is correct for ``problem``."""

    def source(self) -> str:
        """Human-readable task source used in records."""
        return self.name
