"""Multimodal transfer analysis on the ChartQA / synthetic-chart task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from scaletimellm.analysis.compute_optimal import _json_safe


def _cell(
    cells: pd.DataFrame, task: str, strategy: str, budget: int, model: str | None = None
) -> pd.DataFrame:
    subset = cells[
        (cells["task"] == task)
        & (cells["strategy"] == strategy)
        & (cells["budget"] == budget)
    ]
    if model is not None:
        subset = subset[subset["model"] == model]
    return subset


def _weighted_accuracy(cells: pd.DataFrame) -> tuple[float, float, float]:
    if cells.empty:
        return float("nan"), float("nan"), float("nan")
    weights = cells["n_records"].to_numpy(dtype=float)
    values = cells["accuracy"].to_numpy(dtype=float)
    return float((weights * values).sum() / weights.sum()), float(cells["ci_low"].min()), float(
        cells["ci_high"].max()
    )


def build_multimodal_artifact(
    cells: pd.DataFrame,
    compute_payload: dict[str, Any],
    output_path: str | Path = "results/multimodal.json",
) -> dict[str, Any]:
    """Compare the text-task winner with chart-QA alternatives."""
    source_task = "synthetic"
    source_strategy = compute_payload["best_strategy_per_task"][source_task]
    source_budget = int(compute_payload["best_budget_per_task"][source_task])
    chart_best_strategy = compute_payload["best_strategy_per_task"]["chartqa"]
    chart_best_budget = int(compute_payload["best_budget_per_task"]["chartqa"])
    source_cells = _cell(cells, source_task, source_strategy, source_budget)
    chart_same = _cell(cells, "chartqa", source_strategy, source_budget)
    chart_best = _cell(cells, "chartqa", chart_best_strategy, chart_best_budget)
    chart_greedy = _cell(cells, "chartqa", "greedy", 1)
    source_acc, source_low, source_high = _weighted_accuracy(source_cells)
    same_acc, same_low, same_high = _weighted_accuracy(chart_same)
    best_acc, best_low, best_high = _weighted_accuracy(chart_best)
    greedy_acc, greedy_low, greedy_high = _weighted_accuracy(chart_greedy)
    payload = {
        "task": "chartqa",
        "task_source": str(chart_same["task_source"].iloc[0])
        if not chart_same.empty
        else "chartqa_synthetic_standin",
        "modality": "synthetic_chart_standin",
        "source_task": source_task,
        "winning_text_strategy": source_strategy,
        "winning_text_budget": source_budget,
        "source_accuracy": source_acc,
        "source_ci": [source_low, source_high],
        "chart_same_strategy_accuracy": same_acc,
        "chart_same_strategy_ci": [same_low, same_high],
        "chart_best_strategy": chart_best_strategy,
        "chart_best_budget": chart_best_budget,
        "chart_best_accuracy": best_acc,
        "chart_best_ci": [best_low, best_high],
        "chart_greedy_accuracy": greedy_acc,
        "chart_greedy_ci": [greedy_low, greedy_high],
        "transfer_delta_vs_greedy": same_acc - greedy_acc,
        "transfer_delta_vs_chart_best": same_acc - best_acc,
        "retention_source_to_chart": same_acc / source_acc if source_acc > 0 else None,
        "same_strategy_wins_on_chart": same_acc >= best_acc,
        "models": sorted(cells["model"].unique().tolist()),
        "backend": str(cells["backend"].iloc[0]),
        "synthetic": bool(cells["synthetic"].iloc[0]),
        "per_model": chart_same[
            ["model", "accuracy", "ci_low", "ci_high", "mean_flops"]
        ].to_dict(orient="records"),
        "source_per_model": source_cells[
            ["model", "accuracy", "ci_low", "ci_high", "mean_flops"]
        ].to_dict(orient="records"),
    }
    payload = _json_safe(payload)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
