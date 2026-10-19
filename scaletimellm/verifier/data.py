"""Generate outcome-labelled verifier training data from a base model."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from scaletimellm.backends.base import LMBackend
from scaletimellm.config import ExperimentConfig
from scaletimellm.experiments.run import make_backend, make_task
from scaletimellm.tasks.base import Problem, Task
from scaletimellm.utils.seeding import set_all_seeds


def _prefixes(completion: str, label: int, task: Task, problem: Problem) -> list[dict[str, Any]]:
    """Create process-style prefixes with outcome labels for optional training."""
    chunks = []
    for delimiter in ("\n", ". "):
        if delimiter in completion:
            chunks = completion.split(delimiter)
            break
    if len(chunks) <= 1:
        return []
    prefixes = []
    for index in range(1, len(chunks)):
        prefix = delimiter.join(chunks[:index]).strip()  # type: ignore[possibly-undefined]
        if not prefix:
            continue
        prefix_label = int(task.grade(problem, task.extract(prefix)))
        prefixes.append({"text": prefix, "label": prefix_label, "outcome_label": label})
    return prefixes[-2:]


def generate_verifier_data(
    config: ExperimentConfig,
    *,
    n_train_problems: int = 120,
    n_eval_problems: int = 40,
    k: int = 4,
    seed: int = 0,
    output_dir: str | Path = "artifacts/verifier/data",
    task_name: str = "synthetic",
    include_process: bool = True,
    backend: LMBackend | None = None,
    task: Task | None = None,
) -> dict[str, Any]:
    """Sample K solutions per problem and label them with the task grader.

    Train/eval problems are disjoint slices of a seeded load.  Only the prompt and
    sampled completion are stored for the verifier; the gold answer is retained
    in metadata for auditing but is not part of the verifier input.
    """
    if config.offline:
        os.environ["SCALETIMELLM_OFFLINE"] = "1"
    set_all_seeds(seed)
    backend = backend or make_backend(config, config.model, seed=seed)
    task = task or make_task(config, task_name)
    problems = task.load(n_train_problems + n_eval_problems, seed=seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / "train.jsonl"
    eval_path = destination / "eval.jsonl"
    counts = {"train": 0, "eval": 0}
    labels = {"train": 0, "eval": 0}
    with train_path.open("w", encoding="utf-8") as train_handle, eval_path.open(
        "w", encoding="utf-8"
    ) as eval_handle:
        for problem_index, problem in enumerate(problems):
            split = "train" if problem_index < n_train_problems else "eval"
            generations = backend.generate(
                [task.prompt(problem)],
                n=k,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
                seed=seed,
            )[0]
            for generation in generations:
                answer = task.extract(generation.text)
                label = int(task.grade(problem, answer))
                row = {
                    "prompt": task.prompt(problem),
                    "completion": generation.text,
                    "label": label,
                    "answer": answer,
                    "problem_id": problem.id,
                    "split": split,
                    "gold": problem.gold,
                    "difficulty": problem.difficulty,
                    "task": task.name,
                    "source": problem.metadata.get("source", task.name),
                    "backend": backend.name,
                    "n_params": backend.n_params,
                    "prompt_tokens": generation.prompt_tokens,
                    "completion_tokens": generation.completion_tokens,
                }
                if include_process:
                    row["process_prefixes"] = _prefixes(
                        generation.text, label, task, problem
                    )
                handle = train_handle if split == "train" else eval_handle
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                counts[split] += 1
                labels[split] += label
    metadata = {
        "seed": seed,
        "k": k,
        "n_train_problems": n_train_problems,
        "n_eval_problems": n_eval_problems,
        "train_rows": counts["train"],
        "eval_rows": counts["eval"],
        "train_positive": labels["train"],
        "eval_positive": labels["eval"],
        "train_positive_rate": labels["train"] / max(1, counts["train"]),
        "eval_positive_rate": labels["eval"] / max(1, counts["eval"]),
        "task": task.name,
        "backend": backend.name,
        "model": config.model.name,
        "n_params": config.model.n_params,
        "synthetic": backend.name == "mock",
        "paths": {"train": str(train_path), "eval": str(eval_path)},
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL verifier dataset."""
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def preference_pairs(rows: list[dict[str, Any]], max_per_problem: int = 4) -> list[dict[str, str]]:
    """Build (chosen, rejected) pairs from outcome-labelled rows."""
    by_problem: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_problem.setdefault(str(row["problem_id"]), []).append(row)
    pairs = []
    for problem_rows in by_problem.values():
        positives = [row for row in problem_rows if int(row["label"]) == 1][:max_per_problem]
        negatives = [row for row in problem_rows if int(row["label"]) == 0][:max_per_problem]
        for positive in positives:
            for negative in negatives:
                pairs.append(
                    {
                        "chosen": f"{positive['prompt']}\n\n{positive['completion']}",
                        "rejected": f"{negative['prompt']}\n\n{negative['completion']}",
                    }
                )
    return pairs


def verifier_input_text(prompt: str, completion: str) -> str:
    """Canonical text layout used by every learned verifier."""
    return f"{prompt}\n\n[COMPLETION]\n{completion}"
