"""Serving latency and throughput benchmark."""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from scaletimellm.config import ExperimentConfig
from scaletimellm.experiments.run import make_backend, make_strategy, make_task
from scaletimellm.utils.seeding import set_all_seeds


def benchmark_serving(
    config: ExperimentConfig,
    *,
    strategy_name: str | None = None,
    budget: int | None = None,
    n_problems: int | None = None,
    seed: int | None = None,
    output_path: str | Path = "results/serving_bench.json",
) -> dict[str, Any]:
    """Benchmark the configured backend and strategy on a small problem set."""
    strategy_name = strategy_name or config.strategy
    budget = int(budget or config.budgets[0])
    n_problems = int(n_problems or config.n_problems)
    seed = int(config.seeds[0] if seed is None else seed)
    set_all_seeds(seed)
    backend = make_backend(config, config.model, seed=seed)
    task = make_task(config, config.task)
    problems = task.load(n_problems, seed=seed)
    latencies = []
    prompt_tokens = 0
    completion_tokens = 0
    samples = 0
    correct = 0
    process = psutil.Process()
    start_rss = process.memory_info().rss / (1024**2)
    wall_started = time.perf_counter()
    for problem in problems:
        strategy = make_strategy(config, strategy_name, task, problem)
        started = time.perf_counter()
        result = strategy.solve(problem, backend, task, budget=budget, seed=seed)
        latencies.append(time.perf_counter() - started)
        prompt_tokens += result.prompt_tokens
        completion_tokens += result.completion_tokens
        samples += result.n_samples
        correct += int(result.is_correct)
    wall_time = time.perf_counter() - wall_started
    peak_rss = process.memory_info().rss / (1024**2)
    latency_array = np.asarray(latencies, dtype=float)
    try:
        import torch

        torch_version = torch.__version__
        if torch.cuda.is_available():
            gpu_memory_mb = float(torch.cuda.max_memory_allocated() / (1024**2))
            device = "cuda"
        else:
            gpu_memory_mb = None
            device = backend.info().get("device", "cpu")
    except ImportError:
        torch_version = None
        gpu_memory_mb = None
        device = "unknown"
    payload = {
        "backend_requested": config.backend.name,
        "backend_used": backend.name,
        "model": config.model.name,
        "n_params": backend.n_params,
        "strategy": strategy_name,
        "budget": budget,
        "seed": seed,
        "n_problems": len(problems),
        "n_samples": samples,
        "accuracy": correct / max(1, len(problems)),
        "latency_p50_s": float(np.percentile(latency_array, 50)),
        "latency_p95_s": float(np.percentile(latency_array, 95)),
        "latency_mean_s": float(latency_array.mean()),
        "latency_min_s": float(latency_array.min()),
        "latency_max_s": float(latency_array.max()),
        "total_wall_time_s": wall_time,
        "throughput_completion_tokens_per_s": completion_tokens / wall_time,
        "throughput_total_tokens_per_s": (prompt_tokens + completion_tokens) / wall_time,
        "throughput_problems_per_s": len(problems) / wall_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "start_rss_mb": start_rss,
        "peak_rss_mb": peak_rss,
        "gpu_memory_mb": gpu_memory_mb,
        "device": device,
        "model_load_time_s": backend.load_time_s,
        "backend_info": backend.info(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=True),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
        },
        "software": {
            "python": sys.version.split()[0],
            "torch": torch_version,
        },
        "synthetic": backend.name == "mock",
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
