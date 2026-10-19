"""Compute-optimal analysis: frontiers, slopes and equal-FLOPs comparisons."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scaletimellm.analysis.aggregate import bootstrap_ci


def _deployable(frame: pd.DataFrame) -> pd.DataFrame:
    """Exclude oracle and length-prior ablations from deployable comparisons."""
    return frame[
        ~frame["strategy"].str.endswith("_oracle")
        & ~frame["strategy"].str.endswith("_length")
    ].copy()


def pareto_frontier(cells: pd.DataFrame, metric: str = "accuracy") -> pd.DataFrame:
    """Return configurations not dominated on accuracy and mean FLOPs."""
    rows = []
    for _task, group in cells.groupby("task"):
        ordered = group.sort_values(["mean_flops", metric], ascending=[True, False])
        best = -math.inf
        for _, row in ordered.iterrows():
            if row[metric] > best + 1e-12:
                rows.append(row.to_dict())
                best = float(row[metric])
    return pd.DataFrame(rows)


def cost_pareto(cells: pd.DataFrame) -> pd.DataFrame:
    """Pareto frontier on accuracy versus dollars per correct answer."""
    rows = []
    for _task, group in cells.groupby("task"):
        finite = group[np.isfinite(group["dollars_per_correct"])]
        ordered = finite.sort_values(["dollars_per_correct", "accuracy"], ascending=[True, False])
        best = -math.inf
        for _, row in ordered.iterrows():
            if row["accuracy"] > best + 1e-12:
                rows.append(row.to_dict())
                best = float(row["accuracy"])
    return pd.DataFrame(rows)


def _choose_at_cap(group: pd.DataFrame, cap: float) -> dict[str, Any] | None:
    eligible = group[group["mean_flops"] <= cap]
    if eligible.empty:
        return None
    ordered = eligible.sort_values(["accuracy", "mean_flops"], ascending=[False, True])
    return ordered.iloc[0].to_dict()


def matched_budget_table(cells: pd.DataFrame, n_caps: int = 5) -> tuple[pd.DataFrame, dict[str, float]]:
    """Choose the best config under each of several matched FLOPs caps."""
    deployable = _deployable(cells)
    rows = []
    caps_by_task: dict[str, float] = {}
    for task, group in deployable.groupby("task"):
        low = float(group["mean_flops"].quantile(0.10))
        high = float(group["mean_flops"].quantile(0.90))
        caps = np.geomspace(max(low, 1.0), max(high, low * 1.01), n_caps)
        operating = float(np.median(caps))
        caps_by_task[task] = operating
        for cap in caps:
            chosen = _choose_at_cap(group, float(cap))
            if chosen is None:
                continue
            chosen = dict(chosen)
            chosen["flops_cap"] = float(cap)
            chosen["operating_cap"] = operating
            rows.append(chosen)
    return pd.DataFrame(rows), caps_by_task


def compute_optimal_by_difficulty(
    difficulty_cells: pd.DataFrame, caps_by_task: dict[str, float]
) -> pd.DataFrame:
    """Best deployable config per task/difficulty at each task's operating cap."""
    deployable = _deployable(difficulty_cells)
    rows = []
    for (task, _bucket), group in deployable.groupby(["task", "difficulty"]):
        cap = caps_by_task.get(task)
        if cap is None:
            continue
        chosen = _choose_at_cap(group, cap)
        if chosen is None:
            continue
        chosen = dict(chosen)
        chosen["flops_cap"] = cap
        rows.append(chosen)
    return pd.DataFrame(rows)


def accuracy_vs_log_flops_slopes(cells: pd.DataFrame) -> pd.DataFrame:
    """Fit accuracy = a + b*log10(FLOPs) for each task and strategy."""
    deployable = _deployable(cells)
    rows = []
    for (task, strategy), group in deployable.groupby(["task", "strategy"]):
        clean = group[np.isfinite(group["mean_flops"]) & (group["mean_flops"] > 0)]
        if len(clean) < 2:
            continue
        x = np.log10(clean["mean_flops"].to_numpy(dtype=float))
        y = clean["accuracy"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        predictions = slope * x + intercept
        ss_res = float(np.sum((y - predictions) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        rows.append(
            {
                "task": task,
                "strategy": strategy,
                "slope_accuracy_per_decade_flops": float(slope),
                "r2": float(r2),
                "n_points": int(len(clean)),
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "strategy"]).reset_index(drop=True)


def _records_for(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    subset = frame
    for key, value in spec.items():
        subset = subset[subset[key] == value]
    columns = ["task", "problem_id", "seed", "is_correct", "total_tokens", "flops", "cost_usd"]
    return subset[columns].copy()


def paired_delta_ci(
    frame: pd.DataFrame,
    spec_a: dict[str, Any],
    spec_b: dict[str, Any],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap the paired difference in accuracy (A - B) over problem keys."""
    left = _records_for(frame, spec_a).rename(columns={"is_correct": "correct_a"})
    right = _records_for(frame, spec_b).rename(columns={"is_correct": "correct_b"})
    merged = left.merge(
        right,
        on=["task", "problem_id", "seed"],
        suffixes=("_a", "_b"),
    )
    if merged.empty:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    # Average over seeds per problem before bootstrapping problems.
    per_problem = (
        merged.groupby(["task", "problem_id"])
        .agg(correct_a=("correct_a", "mean"), correct_b=("correct_b", "mean"))
        .reset_index()
    )
    differences = per_problem["correct_a"] - per_problem["correct_b"]
    delta, low, high = bootstrap_ci(differences, n_boot=n_boot, seed=seed)
    return {
        "delta": float(delta),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_problems": int(len(per_problem)),
    }


def equal_flops_comparison(
    frame: pd.DataFrame, cells: pd.DataFrame, n_boot: int = 2000
) -> pd.DataFrame:
    """Compare small-model sampling with big-model greedy at matched FLOPs."""
    deployable = _deployable(cells)
    params = frame.groupby("model")["n_params"].median()
    if len(params) < 2:
        return pd.DataFrame()
    small_model = str(params.idxmin())
    big_model = str(params.idxmax())
    rows = []
    for task in sorted(deployable["task"].unique()):
        big = deployable[
            (deployable["task"] == task)
            & (deployable["model"] == big_model)
            & (deployable["strategy"] == "greedy")
            & (deployable["budget"] == 1)
        ]
        if big.empty:
            continue
        big_row = big.iloc[0].to_dict()
        target = float(big_row["mean_flops"])
        all_candidates = deployable[
            (deployable["task"] == task) & (deployable["model"] == small_model)
        ]
        candidates = all_candidates[
            (all_candidates["mean_flops"] / target >= 0.75)
            & (all_candidates["mean_flops"] / target <= 1.25)
        ]
        if candidates.empty:
            candidates = all_candidates
        # Among configurations within +/-25% of the big model's compute, take
        # the most accurate; ties go to the configuration closest in log-FLOPs.
        candidates = candidates.assign(
            log_distance=(candidates["mean_flops"] / target).apply(lambda value: abs(math.log(value)))
        )
        candidate = candidates.sort_values(
            ["accuracy", "log_distance", "mean_flops"],
            ascending=[False, True, True],
        ).iloc[0]
        small_row = candidate.to_dict()
        delta = paired_delta_ci(
            frame,
            {
                "task": task,
                "model": small_model,
                "strategy": small_row["strategy"],
                "budget": int(small_row["budget"]),
            },
            {
                "task": task,
                "model": big_model,
                "strategy": "greedy",
                "budget": 1,
            },
            n_boot=n_boot,
        )
        if delta["n_problems"] == 0:
            continue
        if delta["ci_low"] > 0:
            verdict = "small_model_plus_sampling"
        elif delta["ci_high"] < 0:
            verdict = "bigger_model_greedy"
        else:
            verdict = "statistical_tie"
        rows.append(
            {
                "task": task,
                "small_model": small_model,
                "small_strategy": small_row["strategy"],
                "small_budget": int(small_row["budget"]),
                "small_accuracy": float(small_row["accuracy"]),
                "small_flops": float(small_row["mean_flops"]),
                "big_model": big_model,
                "big_strategy": "greedy",
                "big_budget": 1,
                "big_accuracy": float(big_row["accuracy"]),
                "big_flops": target,
                "delta_accuracy": delta["delta"],
                "delta_ci_low": delta["ci_low"],
                "delta_ci_high": delta["ci_high"],
                "equal_flops_ratio": float(small_row["mean_flops"] / target),
                "n_problems": delta["n_problems"],
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return _json_safe(value.item())
    return value


def run_compute_optimal(
    frame: pd.DataFrame,
    cells: pd.DataFrame,
    difficulty_cells: pd.DataFrame,
    output_path: str | Path = "results/compute_optimal.json",
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Compute all analyses and persist a JSON artifact."""
    frontier = pareto_frontier(cells)
    cost_frontier = cost_pareto(cells)
    matched, caps = matched_budget_table(cells)
    difficulty_optimal = compute_optimal_by_difficulty(difficulty_cells, caps)
    slopes = accuracy_vs_log_flops_slopes(cells)
    equal = equal_flops_comparison(frame, cells, n_boot=n_boot)
    payload = {
        "operating_caps_flops": caps,
        "pareto_frontier": frontier.to_dict(orient="records"),
        "cost_pareto_frontier": cost_frontier.to_dict(orient="records"),
        "matched_flops": matched.to_dict(orient="records"),
        "compute_optimal_by_difficulty": difficulty_optimal.to_dict(orient="records"),
        "slopes": slopes.to_dict(orient="records"),
        "equal_flops": equal.to_dict(orient="records"),
        "best_strategy_per_task": {},
        "best_budget_per_task": {},
    }
    for task, group in _deployable(cells).groupby("task"):
        best = group.sort_values(["accuracy", "mean_flops"], ascending=[False, True]).iloc[0]
        payload["best_strategy_per_task"][task] = str(best["strategy"])
        payload["best_budget_per_task"][task] = int(best["budget"])
    payload = _json_safe(payload)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
