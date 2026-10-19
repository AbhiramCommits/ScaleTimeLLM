"""Offline synthetic arithmetic reasoning task.

The task is deliberately simple to grade but contains easy/medium/hard
arithmetic structures.  It is the primary task for mock-backend runs and the
fallback when a real dataset cannot be downloaded.
"""

from __future__ import annotations

import random

from scaletimellm.tasks.answers import extract_final_number, numbers_match
from scaletimellm.tasks.base import Problem, Task


class SyntheticTask(Task):
    """Seeded mixed-difficulty arithmetic word problems."""

    name = "synthetic"
    modality = "text"

    def load(self, n_problems: int, seed: int) -> list[Problem]:
        """Load a deterministic mixed-difficulty arithmetic slice."""
        rng = random.Random(seed * 100_003 + 17)
        problems: list[Problem] = []
        for index in range(n_problems):
            bucket = index % 3
            if bucket == 0:
                a, b = rng.randint(12, 89), rng.randint(2, 40)
                if rng.random() < 0.5:
                    expression, value = f"{a} + {b}", a + b
                    difficulty = "easy"
                else:
                    a, b = max(a, b), min(a, b)
                    expression, value = f"{a} - {b}", a - b
                    difficulty = "easy"
            elif bucket == 1:
                a, b = rng.randint(3, 19), rng.randint(3, 14)
                expression, value = f"{a} * {b}", a * b
                difficulty = "medium"
            else:
                a, b, c = rng.randint(7, 35), rng.randint(2, 20), rng.randint(2, 9)
                if rng.random() < 0.5:
                    expression, value = f"({a} + {b}) * {c}", (a + b) * c
                else:
                    a, b = max(a, b), min(a, b)
                    expression, value = f"({a} - {b}) * {c}", (a - b) * c
                difficulty = "hard"
            prompt = (
                "You are a careful reasoning assistant. Solve the following problem "
                'step by step and put the final numeric answer after "####".\n\n'
                f"Question: Compute {expression}.\n"
                f"Difficulty: {difficulty}\n\n"
                "Answer:"
            )
            problems.append(
                Problem(
                    id=f"synthetic-{seed}-{index:04d}",
                    text=f"Compute {expression}.",
                    gold=str(value),
                    metadata={
                        "expression": expression,
                        "source": "synthetic",
                        "task": self.name,
                    },
                    prompt_override=prompt,
                    difficulty=difficulty,
                )
            )
        return problems

    def prompt(self, problem: Problem) -> str:
        """Render the chain-of-thought arithmetic prompt."""
        if problem.prompt_override:
            return problem.prompt_override
        return (
            "Solve the problem step by step and put the final numeric answer after "
            f'"####".\n\nQuestion: {problem.text}\n\nAnswer:'
        )

    def extract(self, completion: str) -> str | None:
        """Extract the final numeric answer."""
        return extract_final_number(completion)

    def grade(self, problem: Problem, answer: str | None) -> bool:
        """Grade a numeric answer with tolerance."""
        return numbers_match(problem.gold, answer)
