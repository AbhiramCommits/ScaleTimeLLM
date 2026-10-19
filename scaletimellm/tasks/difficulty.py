"""Difficulty buckets from a cached greedy-pass-rate proxy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scaletimellm.backends.base import LMBackend
from scaletimellm.tasks.base import Problem, Task


def _bucket(pass_rate: float) -> str:
    if pass_rate >= 0.75:
        return "easy"
    if pass_rate >= 0.25:
        return "medium"
    return "hard"


def _fingerprint(problems: list[Problem], task: Task, backend: LMBackend, k: int, seed: int) -> str:
    payload = {
        "task": task.name,
        "backend": backend.name,
        "n_params": backend.n_params,
        "k": k,
        "seed": seed,
        "ids": [problem.id for problem in problems],
        "golds": [problem.gold for problem in problems],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def assign_difficulties(
    problems: list[Problem],
    task: Task,
    backend: LMBackend,
    k: int = 4,
    seed: int = 0,
    cache_path: str | Path | None = None,
    max_new_tokens: int = 96,
    temperature: float = 0.8,
) -> list[Problem]:
    """Assign each problem an easy/medium/hard bucket and cache the result.

    The proxy is the fraction of ``k`` sampled pilot solutions that pass the
    task grader (not a learned difficulty model).  Buckets are fixed thresholds:
    ``>=0.75`` easy, ``>=0.25`` medium, else hard.  Pilot generation is reported
    in the cache metadata and is intentionally excluded from the inference
    budget of the strategies being compared.
    """
    cache: dict[str, Any] | None = None
    fingerprint = _fingerprint(problems, task, backend, k, seed)
    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists():
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if candidate.get("fingerprint") == fingerprint:
                cache = candidate
        except (json.JSONDecodeError, OSError):
            cache = None

    if cache is not None:
        buckets = cache["problems"]
        for problem in problems:
            entry = buckets.get(problem.id, {})
            problem.difficulty = entry.get("difficulty", "medium")
            problem.difficulty_pass_rate = entry.get("pass_rate")
            problem.is_correct = entry.get("passed")
        return problems

    before = backend.cost_tokens
    entries: dict[str, Any] = {}
    for problem in problems:
        generations = backend.generate(
            [task.prompt(problem)],
            n=k,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )[0]
        outcomes = []
        for generation in generations:
            answer = task.extract(generation.text)
            outcomes.append(bool(task.grade(problem, answer)))
        pass_rate = sum(outcomes) / max(1, len(outcomes))
        problem.difficulty = _bucket(pass_rate)
        problem.difficulty_pass_rate = pass_rate
        # ``is_correct`` here denotes the pilot pass rate being positive.
        problem.is_correct = pass_rate > 0.5
        entries[problem.id] = {
            "difficulty": problem.difficulty,
            "pass_rate": pass_rate,
            "passed": problem.is_correct,
            "outcomes": outcomes,
        }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "task": task.name,
                    "backend": backend.name,
                    "n_params": backend.n_params,
                    "k": k,
                    "seed": seed,
                    "pilot_cost_tokens": backend.cost_tokens - before,
                    "problems": entries,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return problems
