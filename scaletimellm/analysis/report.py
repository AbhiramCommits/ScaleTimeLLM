"""Regenerate the full analysis and write summary.json plus REPORT.md."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from scaletimellm import __version__
from scaletimellm.analysis.aggregate import run_aggregation
from scaletimellm.analysis.compute_optimal import run_compute_optimal
from scaletimellm.analysis.multimodal import build_multimodal_artifact
from scaletimellm.analysis.plots import run_plots
from scaletimellm.analysis.tables import (
    _ci,
    _markdown,
    aggregate_by_task_strategy_budget,
    run_tables,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _best_per_task(cells: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for task in sorted(cells["task"].unique()):
        group = cells[
            (cells["task"] == task)
            & ~cells["strategy"].str.endswith(("_oracle", "_length"))
        ]
        best = group.sort_values(["accuracy", "mean_flops"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "task": task,
                "model": best["model"],
                "strategy": best["strategy"],
                "budget": int(best["budget"]),
                "accuracy": float(best["accuracy"]),
                "ci_low": float(best["ci_low"]),
                "ci_high": float(best["ci_high"]),
                "mean_flops": float(best["mean_flops"]),
                "tokens_per_correct": float(best["tokens_per_correct"]),
                "dollars_per_correct": float(best["dollars_per_correct"]),
            }
        )
    return rows


def write_report(
    results_dir: str | Path = "results",
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Run aggregation, compute-optimal analysis, tables, plots and write summary."""
    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)
    frame, cells, difficulty = run_aggregation(
        raw_dir=results / "raw" / "sweep", output_dir=results, n_boot=n_boot
    )
    compute = run_compute_optimal(
        frame,
        cells,
        difficulty,
        output_path=results / "compute_optimal.json",
        n_boot=n_boot,
    )
    multimodal = build_multimodal_artifact(
        cells, compute, output_path=results / "multimodal.json"
    )
    table_paths = run_tables(
        cells_path=results / "aggregate.csv",
        difficulty_path=results / "aggregate_by_difficulty.csv",
        compute_path=results / "compute_optimal.json",
        verifier_path=results / "verifier_metrics.json",
        serving_path=results / "serving_bench.json",
        multimodal_path=results / "multimodal.json",
        output_dir=results / "tables",
    )
    figure_paths = run_plots(
        cells_path=results / "aggregate.csv",
        difficulty_path=results / "aggregate_by_difficulty.csv",
        compute_path=results / "compute_optimal.json",
        output_dir=results / "figures",
    )
    verifier = json.loads((results / "verifier_metrics.json").read_text(encoding="utf-8"))
    serving = json.loads((results / "serving_bench.json").read_text(encoding="utf-8"))
    task_grid = aggregate_by_task_strategy_budget(cells)
    summary = {
        "package_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "provenance": {
            "sweep_backend": str(cells["backend"].iloc[0]),
            "sweep_synthetic": bool(cells["synthetic"].iloc[0]),
            "models": sorted(cells["model"].unique().tolist()),
            "tasks": sorted(cells["task"].unique().tolist()),
            "n_problems_per_seed": int(
                frame.groupby(["task", "seed"])["problem_id"].nunique().max()
            ),
            "problem_seed_evaluations_per_cell": int(cells["n_problems"].max()),
            "seeds": int(cells["n_seeds"].max()),
            "n_cells": int(len(cells)),
            "n_records": int(len(frame)),
            "bootstrap_resamples": n_boot,
            "price_input_per_million_usd": 0.15,
            "price_output_per_million_usd": 0.60,
            "hardware": serving.get("hardware"),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": serving.get("software", {}).get("torch"),
            "serving_backend": serving.get("backend_used"),
            "serving_model": serving.get("model"),
            "wandb_enabled": False,
        },
        "best_per_task": _best_per_task(cells),
        "task_strategy_budget": task_grid.to_dict(orient="records"),
        "compute_optimal_by_difficulty": compute["compute_optimal_by_difficulty"],
        "matched_flops": compute["matched_flops"],
        "equal_flops": compute["equal_flops"],
        "slopes": compute["slopes"],
        "verifier": verifier,
        "serving": serving,
        "multimodal": multimodal,
        "cost_per_correct": [
            {
                "task": row["task"],
                "model": row["model"],
                "strategy": row["strategy"],
                "budget": int(row["budget"]),
                "tokens_per_correct": float(row["tokens_per_correct"]),
                "dollars_per_correct": float(row["dollars_per_correct"]),
            }
            for row in _best_per_task(cells)
        ],
        "artifacts": {
            "raw": str(results / "raw" / "sweep"),
            "aggregate_csv": str(results / "aggregate.csv"),
            "summary_json": str(results / "summary.json"),
            "figures": figure_paths,
            "tables": table_paths,
        },
    }
    summary = _json_safe(summary)
    (results / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    report = _render_report(summary)
    (results / "REPORT.md").write_text(report, encoding="utf-8")
    return summary


def _render_report(summary: dict[str, Any]) -> str:
    provenance = summary["provenance"]
    best = pd.DataFrame(summary["best_per_task"])
    difficulty = pd.DataFrame(summary["compute_optimal_by_difficulty"])
    equal = pd.DataFrame(summary["equal_flops"])
    verifier = summary["verifier"]
    serving = summary["serving"]
    multimodal = summary["multimodal"]

    if not best.empty:
        best["accuracy_ci"] = best.apply(_ci, axis=1)
        best_view = best[
            ["task", "model", "strategy", "budget", "accuracy_ci", "tokens_per_correct", "dollars_per_correct"]
        ]
    else:
        best_view = best
    if not difficulty.empty:
        difficulty_view = difficulty[
            ["task", "difficulty", "model", "strategy", "budget", "accuracy", "mean_flops", "n_problems"]
        ]
    else:
        difficulty_view = difficulty
    if not equal.empty:
        equal_view = equal[
            [
                "task",
                "small_strategy",
                "small_budget",
                "small_accuracy",
                "big_accuracy",
                "equal_flops_ratio",
                "delta_accuracy",
                "delta_ci_low",
                "delta_ci_high",
                "verdict",
            ]
        ]
    else:
        equal_view = equal
    verifier_rows = pd.DataFrame(
        [
            {
                "objective": name,
                "auroc": metrics.get("auroc"),
                "accuracy": metrics.get("accuracy"),
                "ece": metrics.get("ece"),
                "top1": metrics.get("top1_selection_accuracy"),
                "random": metrics.get("random_selection_accuracy"),
            }
            for name, metrics in verifier.get("objectives", {}).items()
        ]
    )
    serving_rows = pd.DataFrame(
        [
            {
                "backend": serving.get("backend_used"),
                "model": serving.get("model"),
                "strategy": serving.get("strategy"),
                "budget": serving.get("budget"),
                "p50_s": serving.get("latency_p50_s"),
                "p95_s": serving.get("latency_p95_s"),
                "tokens_s": serving.get("throughput_completion_tokens_per_s"),
                "problems_s": serving.get("throughput_problems_per_s"),
            }
        ]
    )
    multimodal_rows = pd.DataFrame(
        [
            {"metric": key, "value": multimodal.get(key)}
            for key in (
                "winning_text_strategy",
                "winning_text_budget",
                "source_accuracy",
                "chart_same_strategy_accuracy",
                "chart_best_strategy",
                "chart_best_accuracy",
                "chart_greedy_accuracy",
                "transfer_delta_vs_greedy",
                "transfer_delta_vs_chart_best",
            )
        ]
    )
    lines = [
        "# ScaleTimeLLM — measured results report",
        "",
        "This file is generated by `python -m scaletimellm.cli report` from raw JSONL records. "
        "All sweep numbers come from the **mock/synthetic backend** unless explicitly labelled "
        "as HF serving measurements.",
        "",
        "## Provenance",
        "",
        f"- sweep backend: `{provenance['sweep_backend']}` (synthetic={provenance['sweep_synthetic']})",
        f"- models: {', '.join(f'`{m}`' for m in provenance['models'])}",
        f"- tasks: {', '.join(f'`{t}`' for t in provenance['tasks'])}",
        f"- seeds: {provenance['seeds']}; problems per seed: {provenance['n_problems_per_seed']}; "
        f"problem-seed evaluations per cell: {provenance['problem_seed_evaluations_per_cell']}; "
        f"cells: {provenance['n_cells']}; records: {provenance['n_records']}",
        f"- bootstrap resamples: {provenance['bootstrap_resamples']}",
        f"- price constants: ${provenance['price_input_per_million_usd']}/M input, ${provenance['price_output_per_million_usd']}/M output",
        f"- serving backend: `{provenance['serving_backend']}` with `{provenance['serving_model']}`",
        f"- hardware: {provenance['platform']}; Python {provenance['python']}; torch {provenance['torch']}",
        "- W&B: disabled",
        "",
        "## Headline best configuration per task",
        "",
        _markdown(best_view),
        "",
        "## Compute-optimal configuration per difficulty bucket",
        "",
        _markdown(difficulty_view),
        "",
        "## Equal-FLOPs comparison: small model + sampling vs. bigger model + greedy",
        "",
        _markdown(equal_view),
        "",
        "## Verifier metrics (held-out)",
        "",
        _markdown(verifier_rows),
        "",
        "## Serving benchmark",
        "",
        _markdown(serving_rows),
        "",
        "## Multimodal transfer",
        "",
        _markdown(multimodal_rows),
        "",
        "## Artifacts",
        "",
        f"- raw records: `{summary['artifacts']['raw']}`",
        f"- aggregate: `{summary['artifacts']['aggregate_csv']}`",
        f"- figures: {', '.join(f'`{p}`' for p in summary['artifacts']['figures'])}",
        f"- tables: {', '.join(f'`{p}`' for p in summary['artifacts']['tables'])}",
        "",
        "## Reproduction",
        "",
        "```bash",
        "pip install -e .",
        "bash scripts/train_verifier.sh",
        "python -m scaletimellm.cli sweep --config configs/full.yaml",
        "python -m scaletimellm.cli bench-serve --config configs/serving.yaml",
        "python -m scaletimellm.cli report",
        "```",
        "",
    ]
    return "\n".join(lines)
