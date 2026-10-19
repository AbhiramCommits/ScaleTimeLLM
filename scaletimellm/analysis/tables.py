"""Markdown and CSV tables generated from analysed artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._\n"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if np.isnan(value):
                    values.append("n/a")
                else:
                    values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _write(frame: pd.DataFrame, directory: Path, name: str) -> None:
    frame.to_csv(directory / f"{name}.csv", index=False)
    (directory / f"{name}.md").write_text(_markdown(frame), encoding="utf-8")


def _ci(row: pd.Series) -> str:
    return f"{row['accuracy']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"


def aggregate_by_task_strategy_budget(cells: pd.DataFrame) -> pd.DataFrame:
    """Model-aggregated headline grid (weighted by records)."""
    rows = []
    for key, group in cells.groupby(["task", "strategy", "budget"]):
        task, strategy, budget = key
        weights = group["n_records"].to_numpy(dtype=float)
        accuracy = float(np.average(group["accuracy"], weights=weights))
        rows.append(
            {
                "task": task,
                "strategy": strategy,
                "budget": int(budget),
                "accuracy": accuracy,
                "ci_low": float(group["ci_low"].min()),
                "ci_high": float(group["ci_high"].max()),
                "mean_flops": float(np.average(group["mean_flops"], weights=weights)),
                "mean_tokens": float(
                    np.average(group["mean_tokens_per_problem"], weights=weights)
                ),
                "tokens_per_correct": float(group["total_tokens"].sum() / max(1, group["total_correct"].sum())),
                "dollars_per_correct": float(
                    (group["mean_cost_usd"] * group["n_records"]).sum()
                    / max(1, group["total_correct"].sum())
                ),
                "mean_latency_s": float(np.average(group["mean_latency_s"], weights=weights)),
                "n_models": int(group["model"].nunique()),
                "n_problems": int(group["n_problems"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "strategy", "budget"]).reset_index(drop=True)


def run_tables(
    cells_path: str | Path = "results/aggregate.csv",
    difficulty_path: str | Path = "results/aggregate_by_difficulty.csv",
    compute_path: str | Path = "results/compute_optimal.json",
    verifier_path: str | Path = "results/verifier_metrics.json",
    serving_path: str | Path = "results/serving_bench.json",
    multimodal_path: str | Path = "results/multimodal.json",
    output_dir: str | Path = "results/tables",
) -> list[str]:
    """Write all result tables as Markdown and CSV."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(cells_path)
    _ = pd.read_csv(difficulty_path)
    compute = json.loads(Path(compute_path).read_text(encoding="utf-8"))
    verifier = json.loads(Path(verifier_path).read_text(encoding="utf-8"))
    serving = json.loads(Path(serving_path).read_text(encoding="utf-8"))
    multimodal = json.loads(Path(multimodal_path).read_text(encoding="utf-8"))

    headline = cells.copy()
    headline["accuracy_ci"] = headline.apply(_ci, axis=1)
    headline = headline[
        [
            "task",
            "model",
            "strategy",
            "budget",
            "accuracy_ci",
            "mean_flops",
            "mean_tokens_per_problem",
            "tokens_per_correct",
            "dollars_per_correct",
            "mean_latency_s",
        ]
    ].rename(columns={"mean_tokens_per_problem": "mean_tokens", "mean_latency_s": "latency_s"})
    _write(headline, output, "headline_grid")

    task_grid = aggregate_by_task_strategy_budget(cells)
    task_grid["accuracy_ci"] = task_grid.apply(_ci, axis=1)
    task_grid = task_grid[
        [
            "task",
            "strategy",
            "budget",
            "accuracy_ci",
            "mean_flops",
            "tokens_per_correct",
            "dollars_per_correct",
        ]
    ]
    _write(task_grid, output, "task_strategy_budget")

    difficulty_rows = pd.DataFrame(compute["compute_optimal_by_difficulty"])
    if not difficulty_rows.empty:
        difficulty_rows = difficulty_rows[
            [
                "task",
                "difficulty",
                "model",
                "strategy",
                "budget",
                "accuracy",
                "ci_low",
                "ci_high",
                "mean_flops",
                "n_problems",
            ]
        ]
    _write(difficulty_rows, output, "compute_optimal")

    equal = pd.DataFrame(compute["equal_flops"])
    if not equal.empty:
        equal = equal[
            [
                "task",
                "small_model",
                "small_strategy",
                "small_budget",
                "small_accuracy",
                "small_flops",
                "big_model",
                "big_strategy",
                "big_accuracy",
                "big_flops",
                "equal_flops_ratio",
                "delta_accuracy",
                "delta_ci_low",
                "delta_ci_high",
                "verdict",
            ]
        ]
    _write(equal, output, "equal_flops")

    slopes = pd.DataFrame(compute["slopes"])
    _write(slopes, output, "slopes")

    verifier_rows = []
    for objective, metrics in verifier.get("objectives", {}).items():
        verifier_rows.append(
            {
                "objective": objective,
                "auroc": metrics.get("auroc"),
                "accuracy": metrics.get("accuracy"),
                "ece": metrics.get("ece"),
                "top1_selection_accuracy": metrics.get("top1_selection_accuracy"),
                "random_selection_accuracy": metrics.get("random_selection_accuracy"),
                "n": metrics.get("n"),
            }
        )
    _write(pd.DataFrame(verifier_rows), output, "verifier")

    serving_rows = pd.DataFrame(
        [
            {
                "backend": serving.get("backend_used"),
                "model": serving.get("model"),
                "strategy": serving.get("strategy"),
                "budget": serving.get("budget"),
                "latency_p50_s": serving.get("latency_p50_s"),
                "latency_p95_s": serving.get("latency_p95_s"),
                "completion_tokens_per_s": serving.get(
                    "throughput_completion_tokens_per_s"
                ),
                "problems_per_s": serving.get("throughput_problems_per_s"),
                "peak_rss_mb": serving.get("peak_rss_mb"),
            }
        ]
    )
    _write(serving_rows, output, "serving")

    multimodal_rows = pd.DataFrame(
        [
            {
                "source_task": multimodal.get("source_task"),
                "winning_text_strategy": multimodal.get("winning_text_strategy"),
                "winning_text_budget": multimodal.get("winning_text_budget"),
                "source_accuracy": multimodal.get("source_accuracy"),
                "chart_same_strategy_accuracy": multimodal.get(
                    "chart_same_strategy_accuracy"
                ),
                "chart_best_strategy": multimodal.get("chart_best_strategy"),
                "chart_best_accuracy": multimodal.get("chart_best_accuracy"),
                "chart_greedy_accuracy": multimodal.get("chart_greedy_accuracy"),
                "transfer_delta_vs_greedy": multimodal.get("transfer_delta_vs_greedy"),
                "transfer_delta_vs_chart_best": multimodal.get(
                    "transfer_delta_vs_chart_best"
                ),
            }
        ]
    )
    _write(multimodal_rows, output, "multimodal")

    best_rows = []
    for task in sorted(cells["task"].unique()):
        group = cells[(cells["task"] == task) & ~cells["strategy"].str.endswith(("_oracle", "_length"))]
        best = group.sort_values(["accuracy", "mean_flops"], ascending=[False, True]).iloc[0]
        best_rows.append(
            {
                "task": task,
                "model": best["model"],
                "strategy": best["strategy"],
                "budget": int(best["budget"]),
                "accuracy": best["accuracy"],
                "ci_low": best["ci_low"],
                "ci_high": best["ci_high"],
                "tokens_per_correct": best["tokens_per_correct"],
                "dollars_per_correct": best["dollars_per_correct"],
                "mean_flops": best["mean_flops"],
            }
        )
    _write(pd.DataFrame(best_rows), output, "best_per_task")
    return sorted(str(path) for path in output.glob("*"))
