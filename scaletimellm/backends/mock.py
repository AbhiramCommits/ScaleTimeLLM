"""Deterministic synthetic language-model backend.

``MockLM`` is a simulator, not a pretrained model.  It is included so the full
pipeline can be exercised without network access or model weights.  For each
prompt it parses the synthetic task's arithmetic expression, code specification
or chart table, decides whether the sample is correct using a seeded Bernoulli
draw with a configurable probability, and emits a chain-of-thought-style
completion.  Incorrect samples receive a plausible, diverse distractor.  Token
costs are counted with a deterministic whitespace/punctuation tokenizer and can
be scaled with ``verbosity``.

Every result produced with this backend is labelled ``backend=mock`` in the raw
JSONL, tables and README.
"""

from __future__ import annotations

import ast
import hashlib
import math
import operator
import re
from typing import Any

from scaletimellm.backends.base import Generation, LMBackend

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expression: str) -> float | None:
    """Evaluate arithmetic containing only numbers and arithmetic operators."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](visit(node.operand))
        raise ValueError("unsupported expression")

    try:
        value = visit(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _count_tokens(text: str) -> int:
    """A deterministic tokenizer used only by the mock backend."""
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def _uniform(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _extract_expression(prompt: str) -> str | None:
    patterns = [
        r"Compute\s+([0-9+\-*/() .]+?)\s*\.",
        r"What is\s+([0-9+\-*/() .]+?)\s*\?",
        r"Question:\s*([0-9+\-*/() .]+?)\s*[?.]",
    ]
    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    # A bare arithmetic expression such as "2+2?" used in the quick-start check.
    match = re.fullmatch(r"\s*([0-9+\-*/() .]+?)\s*\??\s*", prompt)
    if match and re.search(r"\d", match.group(1)):
        return match.group(1).strip()
    return None


_CODE_SOLUTIONS: dict[str, str] = {
    "add": "def add(a, b):\n    return a + b",
    "sub": "def sub(a, b):\n    return a - b",
    "mul": "def mul(a, b):\n    return a * b",
    "is_even": "def is_even(n):\n    return n % 2 == 0",
    "max3": "def max3(a, b, c):\n    return max(a, b, c)",
    "abs_diff": "def abs_diff(a, b):\n    return abs(a - b)",
}


def _code_variant(spec_id: str, correct: bool) -> str:
    if correct and spec_id in _CODE_SOLUTIONS:
        return _CODE_SOLUTIONS[spec_id]
    bugs = ["return None", "return 0", "raise ValueError('not implemented')"]
    name = "add"
    args = "a, b"
    for candidate, body in _CODE_SOLUTIONS.items():
        if body.startswith(f"def {candidate}("):
            name = candidate
            args = body.split("(", 1)[1].split(")", 1)[0]
            break
    return f"def {name}({args}):\n    {bugs[0] if spec_id != 'add' else bugs[1]}"


def _parse_chart(prompt: str) -> tuple[dict[str, float], str]:
    data: dict[str, float] = {}
    for match in re.finditer(r"^\s*([A-Za-z][A-Za-z0-9 _-]*):\s*(-?\d+(?:\.\d+)?)\s*$", prompt, re.M):
        data[match.group(1).strip()] = float(match.group(2))
    question_match = re.search(r"Question:\s*(.+)", prompt)
    question = question_match.group(1).strip() if question_match else ""
    return data, question


class MockLM(LMBackend):
    """Seeded synthetic backend with controllable per-sample accuracy."""

    name = "mock"

    def __init__(
        self,
        n_params: int = 494_032_768,
        accuracy: float = 0.42,
        seed: int = 0,
        verbosity: float = 1.0,
        difficulty_adjustment: bool = True,
        **_: Any,
    ) -> None:
        super().__init__(n_params=n_params, seed=seed)
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("accuracy must be in [0, 1]")
        if verbosity <= 0:
            raise ValueError("verbosity must be positive")
        self.accuracy = float(accuracy)
        self.verbosity = float(verbosity)
        self.difficulty_adjustment = bool(difficulty_adjustment)

    def _difficulty_delta(self, prompt: str) -> float:
        if not self.difficulty_adjustment:
            return 0.0
        lower = prompt.lower()
        if "difficulty: easy" in lower:
            return 0.28
        if "difficulty: hard" in lower:
            return -0.28
        expression = _extract_expression(prompt)
        if expression:
            operations = len(re.findall(r"[+\-*/]", expression))
            if operations >= 2:
                return -0.12
            if "*" in expression or "/" in expression:
                return 0.0
            return 0.10
        return 0.0

    def _is_correct(self, prompt: str, sample_index: int, seed: int) -> bool:
        p = min(0.98, max(0.02, self.accuracy + self._difficulty_delta(prompt)))
        key = f"mock|{seed}|{sample_index}|{self.n_params}|{prompt}"
        return _uniform(key) < p

    def _distractor_number(self, gold: float, sample_index: int, seed: int) -> str:
        deltas = [-10, -3, -2, -1, 1, 2, 3, 10]
        delta = deltas[int(_uniform(f"delta|{seed}|{sample_index}|{gold}") * len(deltas))]
        value = gold + delta
        if abs(value - gold) < 1e-9:
            value = gold + 1
        return _format_number(value)

    def _math_completion(
        self, prompt: str, sample_index: int, seed: int, correct: bool, max_new_tokens: int
    ) -> str:
        expression = _extract_expression(prompt)
        if expression is None:
            return "I could not parse the question.\n#### 0"
        value = _safe_eval(expression)
        if value is None:
            return "The expression is invalid.\n#### 0"
        answer = _format_number(value) if correct else self._distractor_number(value, sample_index, seed)
        steps = [
            "I will evaluate the arithmetic carefully.",
            f"The expression is {expression}.",
            "Working left to right and respecting parentheses gives the result.",
        ]
        repeats = max(1, int(round(self.verbosity * 2)))
        steps = steps + ["I double-check the computation."] * repeats
        if not correct:
            steps[-1] = "I may have made an arithmetic slip."
        text = " ".join(steps) + f"\n#### {answer}"
        return self._trim_to_budget(text, max_new_tokens, f"#### {answer}")

    def _code_completion(
        self, prompt: str, correct: bool, max_new_tokens: int
    ) -> str:
        match = re.search(r"\[spec_id=(\w+)\]", prompt)
        spec_id = match.group(1) if match else "add"
        code = _code_variant(spec_id, correct)
        text = "Here is the implementation.\n```python\n" + code + "\n```"
        return self._trim_to_budget(text, max_new_tokens, code)

    def _chart_completion(
        self, prompt: str, sample_index: int, seed: int, correct: bool, max_new_tokens: int
    ) -> str:
        data, question = _parse_chart(prompt)
        if not data:
            return "Answer: unknown"
        question_l = question.lower()
        answer: str
        if "highest" in question_l or "largest" in question_l or "most" in question_l:
            answer = max(data, key=data.get)
        elif "lowest" in question_l or "smallest" in question_l or "least" in question_l:
            answer = min(data, key=data.get)
        elif "sum" in question_l or "total" in question_l:
            answer = _format_number(sum(data.values()))
        elif "value for" in question_l or "value of" in question_l:
            category = next((c for c in data if c.lower() in question_l), None)
            answer = _format_number(data[category]) if category else "unknown"
        elif "difference" in question_l:
            names = [c for c in data if c.lower() in question_l]
            if len(names) >= 2:
                answer = _format_number(abs(data[names[0]] - data[names[1]]))
            else:
                answer = "unknown"
        else:
            answer = max(data, key=data.get)
        if not correct:
            categories = [c for c in data if c != answer]
            if categories:
                answer = categories[int(_uniform(f"chart|{seed}|{sample_index}") * len(categories))]
            else:
                answer = self._distractor_number(float(sum(data.values())), sample_index, seed)
        text = "I read the chart labels and values.\nAnswer: " + str(answer)
        return self._trim_to_budget(text, max_new_tokens, f"Answer: {answer}")

    @staticmethod
    def _trim_to_budget(text: str, max_new_tokens: int, required: str) -> str:
        if _count_tokens(text) <= max_new_tokens:
            return text
        if _count_tokens(required) >= max_new_tokens:
            return required
        prefix_budget = max_new_tokens - _count_tokens(required)
        words = text[: -len(required)].split()
        kept: list[str] = []
        for word in words:
            if _count_tokens(" ".join(kept + [word])) > prefix_budget:
                break
            kept.append(word)
        return (" ".join(kept) + "\n" + required).strip()

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        temperature: float = 0.8,
        max_new_tokens: int = 128,
        seed: int | None = None,
    ) -> list[list[Generation]]:
        if n <= 0:
            raise ValueError("n must be positive")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        seed = self.seed if seed is None else int(seed)
        del temperature  # accuracy is controlled directly; temperature has no effect in the simulator
        output: list[list[Generation]] = []
        for prompt in prompts:
            prompt_tokens = _count_tokens(prompt)
            generations: list[Generation] = []
            for sample_index in range(n):
                correct = self._is_correct(prompt, sample_index, seed)
                if "[spec_id=" in prompt:
                    text = self._code_completion(prompt, correct, max_new_tokens)
                elif "Chart data:" in prompt:
                    text = self._chart_completion(prompt, sample_index, seed, correct, max_new_tokens)
                else:
                    text = self._math_completion(prompt, sample_index, seed, correct, max_new_tokens)
                completion_tokens = min(max_new_tokens, _count_tokens(text))
                self._record(prompt_tokens, completion_tokens)
                generations.append(
                    Generation(
                        text=text,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        sample_index=sample_index,
                        seed=seed,
                    )
                )
            output.append(generations)
        return output
