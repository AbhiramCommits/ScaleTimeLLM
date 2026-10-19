"""Cross-product experiment sweep with per-cell caching."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from scaletimellm.backends.base import LMBackend
from scaletimellm.config import ExperimentConfig
from scaletimellm.experiments.run import (
    make_backend,
    make_strategy,
    make_task,
    record_payload,
)
from scaletimellm.strategies.resolve import build_verifier
from scaletimellm.tasks.base import Problem, Task
from scaletimellm.tasks.difficulty import assign_difficulties
from scaletimellm.utils.monitoring import maybe_init_wandb
from scaletimellm.utils.seeding import set_all_seeds


def _cache_key(config: ExperimentConfig, model: str, task: str, strategy: str, budget: int, seed: int) -> str:
    payload = {
        "experiment": config.name,
        "model": model,
        "task": task,
        "strategy": strategy,
        "budget": budget,
        "seed": seed,
        "n_problems": config.n_problems,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "verifier": config.verifier,
        "backend": config.backend.name,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _prepare_problem_sets(
    config: ExperimentConfig,
    task: Task,
    task_name: str,
    seed: int,
    backends: dict[str, LMBackend],
) -> dict[str, list[Problem]]:
    """Load once and assign cached difficulty buckets for each model."""
    problems = task.load(config.n_problems, seed=seed)
    sets = {}
    cache_dir = Path(config.cache_dir) / "difficulty"
    for model in config.resolved_models():
        backend = backends[model.name]
        copied = copy.deepcopy(problems)
        cache_path = cache_dir / (
            f"{task_name}_{model.name.replace('/', '_')}_n{config.n_problems}_k{config.difficulty_pilot_k}_s{seed}.json"
        )
        assign_difficulties(
            copied,
            task,
            backend,
            k=config.difficulty_pilot_k,
            seed=seed,
            cache_path=cache_path,
            max_new_tokens=min(config.max_new_tokens, 96),
            temperature=config.temperature,
        )
        sets[model.name] = copied
    return sets


def run_sweep(
    config: ExperimentConfig,
    *,
    output_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Run the configured strategy x budget x model x task x seed cross product."""
    if config.offline:
        os.environ["SCALETIMELLM_OFFLINE"] = "1"
    started = time.perf_counter()
    wandb_run = maybe_init_wandb(
        config,
        run_name=config.name,
        extra={"n_problems": config.n_problems, "seeds": config.seeds},
    )
    output_dir = Path(output_dir or config.output_dir)
    raw_dir = output_dir / "raw" / "sweep"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Raw cell JSONL files double as the per-cell cache.  This keeps every
    # committed artifact in one place and makes resume a matter of re-reading
    # the files that are already there.
    cache_dir = Path(cache_dir) if cache_dir is not None else raw_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    models = config.resolved_models()
    tasks = config.resolved_tasks()
    strategies = config.resolved_strategies()
    seeds = list(config.seeds)
    backends = {
        model.name: make_backend(config, model, seed=0)
        for model in models
    }
    task_objects = {name: make_task(config, name) for name in tasks}
    problem_sets: dict[tuple[str, str, int], list[Problem]] = {}
    for task_name in tasks:
        for seed in seeds:
            prepared = _prepare_problem_sets(
                config, task_objects[task_name], task_name, seed, backends
            )
            for model_name, problems in prepared.items():
                problem_sets[(task_name, model_name, seed)] = problems

    learned_verifier = None
    if any(
        strategy.lower()
        in {
            "best_of_n",
            "weighted_vote",
            "step_search",
            "best-of-n",
            "weighted-vote",
            "step-search",
        }
        for strategy in strategies
    ) and config.verifier not in {"oracle", "length", "length_prior"}:
        dummy_problem = next(iter(problem_sets.values()))[0]
        learned_verifier = build_verifier(
            config.verifier,
            task=task_objects[tasks[0]],
            problem=dummy_problem,
            checkpoint=config.verifier_path,
            target_tokens=config.max_new_tokens // 2,
        )

    cells = []
    cached_cells = 0
    for model in models:
        for task_name in tasks:
            task = task_objects[task_name]
            for seed in seeds:
                set_all_seeds(seed)
                problems = problem_sets[(task_name, model.name, seed)]
                for strategy_name in strategies:
                    budgets = [1] if strategy_name == "greedy" else config.budgets
                    for budget in budgets:
                        key = _cache_key(
                            config, model.name, task_name, strategy_name, budget, seed
                        )
                        cache_path = cache_dir / f"{key}.jsonl"
                        records: list[dict[str, Any]] = []
                        if resume and cache_path.exists():
                            with cache_path.open("r", encoding="utf-8") as handle:
                                records = [json.loads(line) for line in handle if line.strip()]
                            if len(records) != len(problems):
                                records = []
                            else:
                                cached_cells += 1
                        if not records:
                            backend = backends[model.name]
                            backend.reset_cost()
                            for problem in problems:
                                strategy = make_strategy(
                                    config,
                                    strategy_name,
                                    task,
                                    problem,
                                    shared_verifier=learned_verifier,
                                )
                                result = strategy.solve(
                                    problem, backend, task, budget=budget, seed=seed
                                )
                                records.append(
                                    record_payload(
                                        config=config,
                                        model=model,
                                        backend=backend,
                                        task=task,
                                        problem=problem,
                                        strategy_name=strategy_name,
                                        budget=budget,
                                        seed=seed,
                                        result=result,
                                    )
                                )
                            tmp_path = cache_path.with_suffix(".tmp")
                            with tmp_path.open("w", encoding="utf-8") as handle:
                                for record in records:
                                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                            tmp_path.replace(cache_path)
                        cells.append(
                            {
                                "model": model.name,
                                "task": task_name,
                                "strategy": strategy_name,
                                "budget": budget,
                                "seed": seed,
                                "accuracy": sum(record["is_correct"] for record in records)
                                / max(1, len(records)),
                                "records": len(records),
                                "cache_path": str(cache_path),
                            }
                        )
                        print(
                            f"[sweep] {model.name} {task_name} {strategy_name} b{budget} s{seed}: "
                            f"acc={cells[-1]['accuracy']:.3f}",
                            file=sys.stderr,
                        )
    summary = {
        "experiment": config.name,
        "backend": config.backend.name,
        "models": [model.name for model in models],
        "tasks": tasks,
        "strategies": strategies,
        "budgets": config.budgets,
        "seeds": seeds,
        "n_problems": config.n_problems,
        "cells": len(cells),
        "records": sum(cell["records"] for cell in cells),
        "cached_cells": cached_cells,
        "wall_time_s": time.perf_counter() - started,
        "raw_dir": str(raw_dir),
        "raw_jsonl_files": len(list(raw_dir.glob("*.jsonl"))),
        "cell_summaries": cells,
    }
    (output_dir / "sweep_run.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "cells": summary["cells"],
                "records": summary["records"],
                "wall_time_s": summary["wall_time_s"],
                "cached_cells": summary["cached_cells"],
            }
        )
        wandb_run.finish()
    return summary
