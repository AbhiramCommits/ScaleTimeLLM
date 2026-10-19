"""Aggregate raw JSONL records into tidy tables with bootstrap confidence intervals."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_RAW_DIR = Path("results/raw/sweep")
DEFAULT_RESULTS_DIR = Path("results")


def load_records(paths: Iterable[str | Path] | str | Path | None = None) -> pd.DataFrame:
    """Load raw JSONL records from one file, a directory, or a glob."""
    if paths is None:
        paths = DEFAULT_RAW_DIR
    if isinstance(paths, (str, Path)):
        root = Path(paths)
        if root.is_dir():
            files = sorted(root.rglob("*.jsonl"))
        else:
            files = [root]
    else:
        files = []
        for value in paths:
            candidate = Path(value)
            files.extend(sorted(candidate.rglob("*.jsonl")) if candidate.is_dir() else [candidate])
    rows: list[dict[str, Any]] = []
    for file in files:
        if not file.exists():
            continue
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"no JSONL records found under {paths}")
    frame = pd.DataFrame(rows)
    if "budget_requested" in frame:
        frame["budget"] = frame["budget_requested"].astype(int)
    if "model" not in frame and "model_name" in frame:
        frame["model"] = frame["model_name"]
    return frame


def bootstrap_ci(
    values: Iterable[float],
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Non-parametric bootstrap mean and percentile CI."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(array.mean())
    if array.size == 1 or np.allclose(array, array[0]):
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_boot, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return mean, float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def _problem_means(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    keys = group_columns + ["problem_id"]
    return (
        frame.groupby(keys, dropna=False)["is_correct"]
        .mean()
        .reset_index()
    )


def aggregate_cells(
    frame: pd.DataFrame,
    group_columns: list[str] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Aggregate to strategy cells with bootstrap CIs over problems."""
    group_columns = group_columns or [
        "task",
        "model",
        "strategy",
        "budget",
        "backend",
    ]
    group_columns = [column for column in group_columns if column in frame.columns]
    problem_means = _problem_means(frame, group_columns)
    rows = []
    for key, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_map = dict(zip(group_columns, key, strict=True))
        per_problem = problem_means
        for column, value in key_map.items():
            per_problem = per_problem[per_problem[column] == value]
        mean, low, high = bootstrap_ci(per_problem["is_correct"], n_boot=n_boot, seed=seed)
        correct = float(group["is_correct"].sum())
        total_tokens = float(group["total_tokens"].sum())
        total_cost = float(group["cost_usd"].sum())
        rows.append(
            {
                **key_map,
                "accuracy": mean,
                "ci_low": low,
                "ci_high": high,
                "n_records": int(len(group)),
                "n_problems": int(group["problem_id"].nunique()),
                "n_seeds": int(group["seed"].nunique()),
                "total_tokens": total_tokens,
                "mean_tokens_per_problem": float(
                    group.groupby("problem_id")["total_tokens"].mean().mean()
                ),
                "mean_flops": float(group["flops"].mean()),
                "mean_latency_s": float(group["wall_time_s"].mean()),
                "mean_n_samples": float(group["n_samples"].mean()),
                "total_correct": correct,
                "tokens_per_correct": total_tokens / correct if correct > 0 else float("inf"),
                "dollars_per_correct": total_cost / correct if correct > 0 else float("inf"),
                "mean_cost_usd": float(group["cost_usd"].mean()),
                "synthetic": bool(group["is_synthetic"].iloc[0])
                if "is_synthetic" in group
                else True,
                "task_source": str(group["task_source"].iloc[0])
                if "task_source" in group
                else str(key_map.get("task", "")),
                "modality": str(group["modality"].iloc[0])
                if "modality" in group
                else "text",
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(group_columns).reset_index(drop=True)


def run_aggregation(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    output_dir: str | Path = DEFAULT_RESULTS_DIR,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw records and write headline/difficulty aggregates."""
    frame = load_records(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    headline = aggregate_cells(frame, n_boot=n_boot, seed=seed)
    difficulty = aggregate_cells(
        frame,
        group_columns=[
            "task",
            "model",
            "strategy",
            "budget",
            "difficulty",
            "backend",
        ],
        n_boot=n_boot,
        seed=seed,
    )
    headline.to_csv(output_dir / "aggregate.csv", index=False)
    difficulty.to_csv(output_dir / "aggregate_by_difficulty.csv", index=False)
    return frame, headline, difficulty


def headline_lookup(cells: pd.DataFrame, **filters: Any) -> dict[str, Any] | None:
    """Return the first matching cell as a dictionary."""
    subset = cells
    for key, value in filters.items():
        if key in subset:
            subset = subset[subset[key] == value]
    if subset.empty:
        return None
    return subset.iloc[0].to_dict()


def format_pct(value: float, digits: int = 1) -> str:
    """Format a proportion as a percentage."""
    return f"{100.0 * value:.{digits}f}%"
