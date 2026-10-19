"""HumanEval code generation with a sandboxed unit-test grader."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from scaletimellm.tasks.answers import extract_fenced_code
from scaletimellm.tasks.base import Problem, Task
from scaletimellm.tasks.gsm8k import downloads_allowed
from scaletimellm.tasks.sandbox import run_python_tests


@dataclass(frozen=True)
class CodeSpec:
    """A tiny synthetic coding problem used when HumanEval is unavailable."""

    spec_id: str
    signature: str
    description: str
    tests: str


_CODE_SPECS = [
    CodeSpec(
        "add",
        "add(a, b)",
        "return the sum of two numbers",
        "def check(candidate):\n"
        "    assert candidate(2, 3) == 5\n"
        "    assert candidate(-4, 1) == -3\n"
        "check(add)\n",
    ),
    CodeSpec(
        "sub",
        "sub(a, b)",
        "return the first number minus the second",
        "def check(candidate):\n"
        "    assert candidate(9, 4) == 5\n"
        "    assert candidate(2, 7) == -5\n"
        "check(sub)\n",
    ),
    CodeSpec(
        "mul",
        "mul(a, b)",
        "return the product of two integers",
        "def check(candidate):\n"
        "    assert candidate(6, 7) == 42\n"
        "    assert candidate(-2, 5) == -10\n"
        "check(mul)\n",
    ),
    CodeSpec(
        "is_even",
        "is_even(n)",
        "return True when n is even and False otherwise",
        "def check(candidate):\n"
        "    assert candidate(4) is True\n"
        "    assert candidate(7) is False\n"
        "check(is_even)\n",
    ),
    CodeSpec(
        "max3",
        "max3(a, b, c)",
        "return the largest of three numbers",
        "def check(candidate):\n"
        "    assert candidate(1, 9, 4) == 9\n"
        "    assert candidate(-2, -8, -3) == -2\n"
        "check(max3)\n",
    ),
    CodeSpec(
        "abs_diff",
        "abs_diff(a, b)",
        "return the absolute difference between two numbers",
        "def check(candidate):\n"
        "    assert candidate(3, 10) == 7\n"
        "    assert candidate(8, 2) == 6\n"
        "check(abs_diff)\n",
    ),
]


class HumanEvalTask(Task):
    """HumanEval completion task graded by executing unit tests.

    If the dataset is unavailable, deterministic synthetic coding problems with
    the same execution grader are used instead.  Every record carries the source
    in its metadata.
    """

    name = "humaneval"
    modality = "text"

    def __init__(self, timeout_s: float = 2.0) -> None:
        self.timeout_s = timeout_s

    def load(self, n_problems: int, seed: int) -> list[Problem]:
        if downloads_allowed():
            try:
                from datasets import load_dataset

                dataset = load_dataset("openai_humaneval", split="test")
                order = list(range(len(dataset)))
                random.Random(seed).shuffle(order)
                problems: list[Problem] = []
                for index in order[:n_problems]:
                    row: dict[str, Any] = dataset[int(index)]
                    entry_point = str(row["entry_point"])
                    prompt = (
                        "Complete the following Python function. Return only a fenced "
                        "Python code block containing the complete function.\n\n"
                        f"```python\n{row['prompt']}```\n"
                    )
                    problems.append(
                        Problem(
                            id=f"humaneval-{int(index):03d}",
                            text=str(row["prompt"]),
                            gold="passes unit tests",
                            metadata={
                                "source": "humaneval",
                                "task": self.name,
                                "entry_point": entry_point,
                                "prompt_code": str(row["prompt"]),
                                "test": str(row["test"]),
                            },
                            prompt_override=prompt,
                        )
                    )
                if problems:
                    return problems
            except Exception:
                pass
        return self._synthetic(n_problems, seed)

    def _synthetic(self, n_problems: int, seed: int) -> list[Problem]:
        rng = random.Random(seed * 31 + 5)
        problems = []
        for index in range(n_problems):
            spec = _CODE_SPECS[rng.randrange(len(_CODE_SPECS))]
            prompt = (
                "Write a complete Python function with signature "
                f"`{spec.signature}` that should {spec.description}. "
                "Return only a fenced Python code block.\n"
                f"[spec_id={spec.spec_id}]"
            )
            problems.append(
                Problem(
                    id=f"humaneval-synthetic-{seed}-{index:03d}",
                    text=f"{spec.signature}: {spec.description}",
                    gold="passes unit tests",
                    metadata={
                        "source": "humaneval_synthetic_fallback",
                        "task": self.name,
                        "entry_point": spec.spec_id,
                        "prompt_code": "",
                        "test": spec.tests,
                        "spec_id": spec.spec_id,
                    },
                    prompt_override=prompt,
                )
            )
        return problems

    def prompt(self, problem: Problem) -> str:
        if problem.prompt_override:
            return problem.prompt_override
        body = problem.metadata.get("prompt_code", problem.text)
        return (
            "Complete the following Python function. Return only a fenced Python code "
            f"block containing the complete function.\n\n```python\n{body}```\n"
        )

    def extract(self, completion: str) -> str | None:
        return extract_fenced_code(completion)

    def grade(self, problem: Problem, answer: str | None) -> bool:
        if not answer:
            return False
        prompt_code = str(problem.metadata.get("prompt_code", ""))
        entry_point = str(problem.metadata.get("entry_point", ""))
        code = answer
        if prompt_code and f"def {entry_point}" not in code:
            code = prompt_code + "\n" + code
        test_code = str(problem.metadata.get("test", ""))
        if not test_code:
            return False
        passed, _ = run_python_tests(
            code,
            test_code,
            entry_point=None,
            timeout=self.timeout_s,
        )
        return passed


def extract_python(completion: str) -> str | None:
    """Backward-compatible helper used by tests and notebooks."""
    return extract_fenced_code(completion)
