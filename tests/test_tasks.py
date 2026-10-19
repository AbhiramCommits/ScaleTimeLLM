"""Unit tests for answer extraction, grading and task loading."""

from __future__ import annotations

from scaletimellm.backends.mock import MockLM
from scaletimellm.tasks.answers import (
    extract_fenced_code,
    extract_final_number,
    numbers_match,
    relaxed_string_match,
)
from scaletimellm.tasks.chartqa import ChartQATask
from scaletimellm.tasks.difficulty import assign_difficulties
from scaletimellm.tasks.gsm8k import GSM8KTask, extract_gsm8k_gold
from scaletimellm.tasks.humaneval import HumanEvalTask
from scaletimellm.tasks.sandbox import run_python_tests
from scaletimellm.tasks.synthetic import SyntheticTask


def test_extract_marker_negative_and_float() -> None:
    assert extract_final_number("work...\n#### -42") == "-42"
    assert extract_final_number("The answer is 3.50") == "3.5"
    assert extract_final_number("result: -0.25") == "-0.25"


def test_extract_last_number_when_no_marker() -> None:
    assert extract_final_number("First 2 then 7 plus 3 gives 10") == "10"


def test_extract_no_answer_returns_none() -> None:
    assert extract_final_number("I do not know.") is None
    assert extract_final_number("") is None


def test_numeric_tolerance_and_wrong_format() -> None:
    assert numbers_match("100", "100.00001")
    assert not numbers_match("100", "101")
    assert not numbers_match("100", "one hundred")
    assert numbers_match("-3", " -3.0 ")


def test_gsm8k_gold_extraction() -> None:
    assert extract_gsm8k_gold("steps\n#### 1,234") == "1234"
    assert extract_gsm8k_gold("no marker but ends at 9") == "9"


def test_synthetic_task_deterministic_and_grades() -> None:
    task = SyntheticTask()
    first = task.load(5, seed=3)
    second = task.load(5, seed=3)
    assert [p.id for p in first] == [p.id for p in second]
    assert [p.gold for p in first] == [p.gold for p in second]
    assert task.grade(first[0], first[0].gold)
    assert not task.grade(first[0], str(int(first[0].gold) + 1))


def test_gsm8k_offline_fallback(monkeypatch) -> None:
    monkeypatch.setenv("SCALETIMELLM_OFFLINE", "1")
    problems = GSM8KTask().load(4, seed=1)
    assert len(problems) == 4
    assert problems[0].metadata["source"] == "gsm8k_synthetic_fallback"


def test_code_extraction_prefers_last_fence() -> None:
    completion = "noise\n```python\ndef f():\n    return 1\n```\n"
    assert extract_fenced_code(completion) == "def f():\n    return 1"


def test_code_failing_and_passing() -> None:
    good = "def add(a, b):\n    return a + b\n"
    bad = "def add(a, b):\n    return a - b\n"
    tests = "def check(f):\n    assert f(2, 3) == 5\ncheck(add)\n"
    assert run_python_tests(good, tests, timeout=1.0)[0]
    assert not run_python_tests(bad, tests, timeout=1.0)[0]


def test_code_timeout_and_import_restriction() -> None:
    tests = "assert add(1, 2) == 3\n"
    timeout_code = "def add(a, b):\n    while True:\n        pass\n"
    assert not run_python_tests(timeout_code, tests, timeout=0.4)[0]
    unsafe = "import os\ndef add(a, b):\n    return a + b\n"
    passed, output = run_python_tests(unsafe, tests, timeout=1.0)
    assert not passed and "unsafe" in output


def test_humaneval_synthetic_grades_correct_code(monkeypatch) -> None:
    monkeypatch.setenv("SCALETIMELLM_OFFLINE", "1")
    task = HumanEvalTask(timeout_s=1.0)
    problems = task.load(10, seed=0)
    solutions = {
        "add": "def add(a, b):\n    return a + b",
        "sub": "def sub(a, b):\n    return a - b",
        "mul": "def mul(a, b):\n    return a * b",
        "is_even": "def is_even(n):\n    return n % 2 == 0",
        "max3": "def max3(a, b, c):\n    return max(a, b, c)",
        "abs_diff": "def abs_diff(a, b):\n    return abs(a - b)",
    }
    problem = next(p for p in problems if p.metadata["spec_id"] in solutions)
    spec_id = problem.metadata["spec_id"]
    good = f"```python\n{solutions[spec_id]}\n```"
    bad = f"```python\ndef {spec_id}(*args):\n    return None\n```"
    assert task.grade(problem, task.extract(good))
    assert not task.grade(problem, task.extract(bad))


def test_chart_relaxed_accuracy(monkeypatch) -> None:
    monkeypatch.setenv("SCALETIMELLM_OFFLINE", "1")
    task = ChartQATask(force_synthetic=True)
    problems = task.load(8, seed=2)
    numeric = next(p for p in problems if p.gold.isdigit())
    value = float(numeric.gold)
    assert task.grade(numeric, f"{value * 1.04:.2f}")
    assert not task.grade(numeric, f"{value * 1.10:.2f}")
    categorical = next(p for p in problems if not p.gold.isdigit())
    assert task.grade(categorical, categorical.gold.upper())


def test_chart_no_answer() -> None:
    task = ChartQATask(force_synthetic=True)
    assert task.extract("") is None
    assert task.extract("I cannot read this chart") == "I cannot read this chart"


def test_relaxed_string_match() -> None:
    assert relaxed_string_match("Jan", " jan. ")
    assert relaxed_string_match("20", "20.9")
    assert not relaxed_string_match("20", "22")


def test_difficulty_buckets_cached(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCALETIMELLM_OFFLINE", "1")
    task = SyntheticTask()
    problems = task.load(6, seed=0)
    backend = MockLM(accuracy=0.5, seed=0)
    cache = tmp_path / "difficulty.json"
    assign_difficulties(problems, task, backend, k=4, seed=0, cache_path=cache)
    assert all(p.difficulty in {"easy", "medium", "hard"} for p in problems)
    assert cache.exists()
    before = backend.cost_tokens
    second = task.load(6, seed=0)
    assign_difficulties(second, task, backend, k=4, seed=0, cache_path=cache)
    assert backend.cost_tokens == before
    assert [p.difficulty for p in second] == [p.difficulty for p in problems]
