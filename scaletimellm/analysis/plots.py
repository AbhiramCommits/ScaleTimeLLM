"""Figure generation from analysed artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STRATEGY_COLORS = {
    "greedy": "#1f77b4",
    "self_consistency": "#ff7f0e",
    "best_of_n": "#2ca02c",
    "weighted_vote": "#d62728",
    "step_search": "#9467bd",
    "best_of_n_oracle": "#8c564b",
    "best_of_n_length": "#7f7f7f",
}
MARKERS = {"mock-qwen2.5-0.5b": "o", "mock-qwen2.5-1.5b": "s"}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_accuracy_vs_flops(cells: pd.DataFrame, output: Path) -> None:
    tasks = sorted(cells["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 3.8), sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    for axis, task in zip(axes, tasks, strict=True):
        subset = cells[cells["task"] == task]
        for (strategy, model), group in subset.groupby(["strategy", "model"]):
            ordered = group.sort_values("mean_flops")
            axis.plot(
                ordered["mean_flops"],
                ordered["accuracy"],
                marker=MARKERS.get(model, "o"),
                markersize=3.5,
                linewidth=1.2,
                color=STRATEGY_COLORS.get(strategy, "black"),
                linestyle="-" if "0.5b" in model else "--",
                alpha=0.9,
                label=f"{strategy} ({'0.5B' if '0.5b' in model else '1.5B'})",
            )
        axis.set_xscale("log")
        axis.set_title(task)
        axis.set_xlabel("estimated FLOPs / problem (log)")
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.02, 1.03)
    axes[0].set_ylabel("accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles and len(axes) > 1:
        handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            fontsize=7,
            frameon=False,
            bbox_to_anchor=(0.5, -0.16),
        )
    fig.suptitle("Accuracy vs. estimated inference FLOPs (synthetic/mock backend)")
    _save(fig, output / "accuracy_vs_flops.png")


def _plot_budget_by_difficulty(
    cells: pd.DataFrame, compute_payload: dict, output: Path
) -> None:
    tasks = sorted(cells["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 3.6), sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    for axis, task in zip(axes, tasks, strict=True):
        strategy = compute_payload["best_strategy_per_task"].get(task, "self_consistency")
        small_model = str(cells["model"].iloc[0])
        subset = cells[
            (cells["task"] == task)
            & (cells["strategy"] == strategy)
            & (cells["model"] == small_model)
        ]
        for bucket, group in subset.groupby("difficulty"):
            ordered = group.sort_values("budget")
            axis.plot(
                ordered["budget"],
                ordered["accuracy"],
                marker="o",
                linewidth=1.4,
                label=bucket,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 3, 4, 8, 16])
        axis.set_xticklabels(["1", "2", "3", "4", "8", "16"])
        axis.set_title(f"{task}\nbest strategy: {strategy}")
        axis.set_xlabel("sampling budget N")
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.02, 1.03)
        axis.legend(fontsize=7, title="pilot difficulty")
    axes[0].set_ylabel("accuracy")
    fig.suptitle("Accuracy vs. sampling budget by difficulty bucket (0.5B mock model)")
    _save(fig, output / "accuracy_vs_budget_difficulty.png")


def _plot_pareto(cells: pd.DataFrame, output: Path) -> None:
    tasks = sorted(cells["task"].unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 3.7), sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    for axis, task in zip(axes, tasks, strict=True):
        subset = cells[(cells["task"] == task) & np.isfinite(cells["dollars_per_correct"])]
        for model, group in subset.groupby("model"):
            axis.scatter(
                group["dollars_per_correct"],
                group["accuracy"],
                s=14,
                alpha=0.45,
                marker=MARKERS.get(model, "o"),
                label=model,
            )
        # Cost Pareto frontier: accuracy non-decreasing as cost increases.
        ordered = subset.sort_values("dollars_per_correct")
        frontier_x, frontier_y, best = [], [], -1.0
        for _, row in ordered.iterrows():
            if row["accuracy"] > best + 1e-12:
                frontier_x.append(row["dollars_per_correct"])
                frontier_y.append(row["accuracy"])
                best = float(row["accuracy"])
        axis.plot(frontier_x, frontier_y, color="black", linewidth=1.5, label="cost frontier")
        axis.set_xscale("log")
        axis.set_title(task)
        axis.set_xlabel("$ per correct answer (log)")
        axis.grid(alpha=0.25)
        axis.set_ylim(-0.02, 1.03)
        axis.legend(fontsize=6)
    axes[0].set_ylabel("accuracy")
    fig.suptitle("Accuracy vs. cost per correct answer")
    _save(fig, output / "pareto_frontier.png")


def _plot_verifier_headroom(cells: pd.DataFrame, output: Path) -> None:
    tasks = sorted(cells["task"].unique())
    budget = 8 if 8 in set(cells["budget"]) else int(cells["budget"].max())
    strategies = ["best_of_n", "best_of_n_length", "best_of_n_oracle"]
    fig, axis = plt.subplots(figsize=(7.0, 3.8))
    width = 0.24
    x = np.arange(len(tasks))
    for offset, strategy in enumerate(strategies):
        values, lows, highs = [], [], []
        for task in tasks:
            subset = cells[
                (cells["task"] == task)
                & (cells["strategy"] == strategy)
                & (cells["budget"] == budget)
            ]
            if subset.empty:
                values.append(0.0)
                lows.append(0.0)
                highs.append(0.0)
                continue
            weights = subset["n_records"].to_numpy(dtype=float)
            accuracy = float(np.average(subset["accuracy"], weights=weights))
            values.append(accuracy)
            lows.append(accuracy - float(subset["ci_low"].min()))
            highs.append(float(subset["ci_high"].max()) - accuracy)
        axis.bar(
            x + (offset - 1) * width,
            values,
            width,
            yerr=np.vstack([lows, highs]),
            capsize=3,
            label=strategy,
            color=STRATEGY_COLORS[strategy],
            alpha=0.9,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(tasks)
    axis.set_ylabel(f"accuracy at N={budget}")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    axis.set_title("Verifier headroom: learned vs. length prior vs. oracle")
    _save(fig, output / "verifier_headroom.png")


def _plot_equal_flops(compute_payload: dict, output: Path) -> None:
    rows = compute_payload.get("equal_flops", [])
    if not rows:
        return
    tasks = [row["task"] for row in rows]
    deltas = [100 * row["delta_accuracy"] for row in rows]
    lows = [100 * (row["delta_accuracy"] - row["delta_ci_low"]) for row in rows]
    highs = [100 * (row["delta_ci_high"] - row["delta_accuracy"]) for row in rows]
    fig, axis = plt.subplots(figsize=(6.2, 3.4))
    axis.bar(tasks, deltas, color=["#2ca02c" if value >= 0 else "#d62728" for value in deltas])
    axis.errorbar(tasks, deltas, yerr=np.vstack([lows, highs]), fmt="none", color="black", capsize=4)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("accuracy delta (percentage points)")
    axis.set_title("Small model + sampling minus big model greedy at matched FLOPs")
    axis.grid(axis="y", alpha=0.25)
    for index, (delta, row) in enumerate(zip(deltas, rows, strict=True)):
        axis.text(
            index,
            delta + (3 if delta >= 0 else -8),
            f"{delta:+.1f} pp\nratio {row['equal_flops_ratio']:.2f}",
            ha="center",
            fontsize=7,
        )
    _save(fig, output / "equal_flops.png")


def run_plots(
    cells_path: str | Path = "results/aggregate.csv",
    difficulty_path: str | Path = "results/aggregate_by_difficulty.csv",
    compute_path: str | Path = "results/compute_optimal.json",
    output_dir: str | Path = "results/figures",
) -> list[str]:
    """Generate all result figures and return their paths."""
    cells = pd.read_csv(cells_path)
    difficulty = pd.read_csv(difficulty_path)
    compute_payload = json.loads(Path(compute_path).read_text(encoding="utf-8"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _plot_accuracy_vs_flops(cells, output)
    _plot_budget_by_difficulty(difficulty, compute_payload, output)
    _plot_pareto(cells, output)
    _plot_verifier_headroom(cells, output)
    _plot_equal_flops(compute_payload, output)
    return sorted(str(path) for path in output.glob("*.png"))
