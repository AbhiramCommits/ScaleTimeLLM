"""GSM8K grade-school math task with a synthetic offline fallback."""

from __future__ import annotations

import os
import re
from typing import Any

from scaletimellm.tasks.answers import extract_final_number, numbers_match
from scaletimellm.tasks.base import Problem, Task
from scaletimellm.tasks.synthetic import SyntheticTask


def downloads_allowed() -> bool:
    """Whether network dataset downloads are enabled by the environment."""
    if os.environ.get("SCALETIMELLM_OFFLINE", "0").lower() in {"1", "true", "yes"}:
        return False
    return os.environ.get("HF_HUB_OFFLINE", "0").lower() not in {"1", "true", "yes"}


class GSM8KTask(Task):
    """GSM8K chain-of-thought math.

    The loader attempts ``openai/gsm8k``; if that is unavailable (offline CI, no
    cache, or a network error) it returns a deterministic synthetic arithmetic
    slice and marks every problem with ``source=gsm8k_synthetic_fallback``.
    """

    name = "gsm8k"
    modality = "text"

    def load(self, n_problems: int, seed: int) -> list[Problem]:
        problems: list[Problem] = []
        if downloads_allowed():
            try:
                from datasets import load_dataset

                dataset = load_dataset("openai/gsm8k", "main", split="test")
                order = list(range(len(dataset)))
                import random

                random.Random(seed).shuffle(order)
                for index in order:
                    row: dict[str, Any] = dataset[int(index)]
                    gold = extract_final_number(str(row.get("answer", "")))
                    if gold is None:
                        continue
                    prompt = (
                        "You are a careful reasoning assistant. Solve the grade-school math "
                        'problem step by step and put the final numeric answer after "####".\n\n'
                        f"Question: {row['question']}\n\nAnswer:"
                    )
                    problems.append(
                        Problem(
                            id=f"gsm8k-{int(index):05d}",
                            text=str(row["question"]),
                            gold=gold,
                            metadata={"source": "gsm8k", "task": self.name},
                            prompt_override=prompt,
                        )
                    )
                    if len(problems) >= n_problems:
                        return problems
            except Exception:
                problems = []
        fallback = SyntheticTask().load(n_problems, seed=seed + 7_919)
        for index, problem in enumerate(fallback):
            problem.id = f"gsm8k-fallback-{seed}-{index:04d}"
            problem.metadata = {
                "source": "gsm8k_synthetic_fallback",
                "task": self.name,
                "reason": "offline or dataset unavailable",
            }
        return fallback

    def prompt(self, problem: Problem) -> str:
        if problem.prompt_override:
            return problem.prompt_override
        return (
            "Solve the grade-school math problem step by step and put the final numeric "
            f'answer after "####".\n\nQuestion: {problem.text}\n\nAnswer:'
        )

    def extract(self, completion: str) -> str | None:
        return extract_final_number(completion)

    def grade(self, problem: Problem, answer: str | None) -> bool:
        return numbers_match(problem.gold, answer, rel_tol=1e-4, abs_tol=1e-4)


def extract_gsm8k_gold(answer_field: str) -> str | None:
    """Extract the gold value after the final ``####`` in a GSM8K answer field."""
    matches = re.findall(r"####\s*(.+)", answer_field)
    if not matches:
        return extract_final_number(answer_field)
    return extract_final_number(matches[-1])
