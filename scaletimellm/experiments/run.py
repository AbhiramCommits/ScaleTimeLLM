"""Single-configuration experiment runner and backend construction."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scaletimellm.backends.base import LMBackend
from scaletimellm.backends.registry import get_backend
from scaletimellm.config import ExperimentConfig, ModelConfig
from scaletimellm.strategies.base import Strategy
from scaletimellm.strategies.registry import get_strategy
from scaletimellm.strategies.resolve import build_verifier
from scaletimellm.tasks.base import Problem, Task
from scaletimellm.tasks.registry import get_task
from scaletimellm.utils.flops import cost_usd
from scaletimellm.utils.seeding import set_all_seeds


def make_backend(config: ExperimentConfig, model: ModelConfig, seed: int = 0) -> LMBackend:
    """Construct the configured backend for one model."""
    if model.mock_accuracy is not None:
        accuracy = model.mock_accuracy
    else:
        accuracy = config.backend.mock_accuracy
    verbosity = model.mock_verbosity or config.backend.mock_verbosity
    if config.backend.name.lower() in {"mock", "synthetic", "simulator"}:
        return get_backend(
            "mock",
            n_params=model.n_params,
            accuracy=accuracy,
            verbosity=verbosity,
            seed=seed,
        )
    if config.backend.name.lower() == "vllm":
        return get_backend(
            "vllm",
            model_name=model.name,
            n_params=model.n_params,
            seed=seed,
            tensor_parallel_size=config.backend.tensor_parallel_size,
            gpu_memory_utilization=config.backend.gpu_memory_utilization,
            fallback_to_mock=config.backend.fallback_to_mock,
            accuracy=accuracy,
            verbosity=verbosity,
        )
    return get_backend(
        "hf",
        model_name=model.name,
        n_params=model.n_params,
        device=model.device,
        dtype=model.dtype,
        trust_remote_code=model.trust_remote_code,
        seed=seed,
        fallback_to_mock=config.backend.fallback_to_mock,
        accuracy=accuracy,
        verbosity=verbosity,
    )


def make_task(config: ExperimentConfig, task_name: str) -> Task:
    """Construct a task with config-appropriate fallbacks."""
    kwargs: dict[str, Any] = {"timeout_s": config.sandbox_timeout_s}
    if task_name.lower() in {"chartqa", "chart", "multimodal"} and config.offline:
        kwargs["force_synthetic"] = True
    return get_task(task_name, **kwargs)


def make_strategy(
    config: ExperimentConfig,
    strategy_name: str,
    task: Task,
    problem: Problem,
) -> Strategy:
    """Construct a strategy and any verifier it needs."""
    verifier = None
    key = strategy_name.lower()
    if key in {
        "best_of_n",
        "best-of-n",
        "bon",
        "weighted_vote",
        "weighted-vote",
        "step_search",
        "step-search",
    }:
        verifier = build_verifier(
            config.verifier,
            task=task,
            problem=problem,
            checkpoint=config.verifier_path,
            target_tokens=config.max_new_tokens // 2,
        )
    return get_strategy(
        strategy_name,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        verifier=verifier,
    )


def record_payload(
    *,
    config: ExperimentConfig,
    model: ModelConfig,
    backend: LMBackend,
    task: Task,
    problem: Problem,
    strategy_name: str,
    budget: int,
    seed: int,
    result: Any,
) -> dict[str, Any]:
    """Build one JSONL record with complete accounting and provenance."""
    price = cost_usd(
        result.prompt_tokens,
        result.completion_tokens,
        config.price_input_per_million,
        config.price_output_per_million,
    )
    return {
        "experiment": config.name,
        "backend": backend.name,
        "backend_config": config.backend.name,
        "is_synthetic": backend.name == "mock",
        "mock_accuracy": config.backend.mock_accuracy,
        "model": model.name,
        "n_params": backend.n_params,
        "task": task.name,
        "task_source": problem.metadata.get("source", task.name),
        "modality": task.modality,
        "strategy": strategy_name,
        "verifier": config.verifier if strategy_name != "greedy" else "none",
        "budget_requested": int(budget),
        "seed": int(seed),
        "problem_id": problem.id,
        "difficulty": problem.difficulty,
        "difficulty_pass_rate": problem.difficulty_pass_rate,
        "gold": problem.gold,
        "answer": result.answer,
        "is_correct": bool(result.is_correct),
        "n_samples": int(result.n_samples),
        "prompt_tokens": int(result.prompt_tokens),
        "completion_tokens": int(result.completion_tokens),
        "total_tokens": int(result.total_tokens),
        "flops": float(result.flops),
        "wall_time_s": float(result.wall_time_s),
        "cost_usd": float(price),
        "price_input_per_million": config.price_input_per_million,
        "price_output_per_million": config.price_output_per_million,
        "trace": result.trace,
    }


def run_experiment(
    config: ExperimentConfig,
    *,
    strategy_name: str | None = None,
    budget: int | None = None,
    task_name: str | None = None,
    seed: int | None = None,
    n_problems: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one strategy/budget/task/seed combination and write raw JSONL."""
    if config.offline:
        os.environ["SCALETIMELLM_OFFLINE"] = "1"
    strategy_name = strategy_name or config.strategy
    task_name = task_name or config.task
    seed = int(config.seeds[0] if seed is None else seed)
    budget = int(budget or config.budgets[0])
    n_problems = int(n_problems or config.n_problems)
    set_all_seeds(seed)
    model = config.model
    backend = make_backend(config, model, seed=seed)
    task = make_task(config, task_name)
    problems = task.load(n_problems, seed=seed)
    records = []
    backend.reset_cost()
    for problem in problems:
        strategy = make_strategy(config, strategy_name, task, problem)
        result = strategy.solve(problem, backend, task, budget=budget, seed=seed)
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
    destination = Path(output_dir or config.output_dir) / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    safe_model = model.name.replace("/", "_")
    path = destination / (
        f"{config.name}_{safe_model}_{task.name}_{strategy_name}_b{budget}_s{seed}.jsonl"
    )
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "path": str(path),
        "records": len(records),
        "accuracy": sum(record["is_correct"] for record in records) / max(1, len(records)),
        "backend": backend.name,
        "model": model.name,
        "task": task.name,
        "strategy": strategy_name,
        "budget": budget,
        "seed": seed,
        "backend_info": backend.info(),
        "synthetic": backend.name == "mock",
    }
    print(json.dumps(summary, indent=2))
    return summary


def strategy_metadata(strategy: Strategy) -> dict[str, Any]:
    """Serialize non-model strategy state for reproducibility."""
    return {key: value for key, value in asdict(strategy).items() if key != "verifier"}
