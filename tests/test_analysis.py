"""Analysis tests: bootstrap, Pareto and compute-optimal selection."""

from __future__ import annotations

import pandas as pd
import pytest

from scaletimellm.analysis.aggregate import bootstrap_ci
from scaletimellm.analysis.compute_optimal import (
    _choose_at_cap,
    accuracy_vs_log_flops_slopes,
    pareto_frontier,
)
from scaletimellm.verifier.eval import metrics_from_labels


def test_bootstrap_ci_analytic_constant_case() -> None:
    mean, low, high = bootstrap_ci([1.0, 1.0, 1.0, 1.0], n_boot=1000)
    assert (mean, low, high) == (1.0, 1.0, 1.0)


def test_bootstrap_ci_analytic_two_point_case() -> None:
    mean, low, high = bootstrap_ci([0.0, 2.0], n_boot=5000, seed=0)
    assert mean == 1.0
    assert low == 0.0
    assert high == 2.0


def test_compute_optimal_chooses_accuracy_under_cap() -> None:
    group = pd.DataFrame(
        [
            {"strategy": "a", "budget": 1, "mean_flops": 10.0, "accuracy": 0.4},
            {"strategy": "b", "budget": 4, "mean_flops": 20.0, "accuracy": 0.7},
            {"strategy": "c", "budget": 16, "mean_flops": 40.0, "accuracy": 0.9},
        ]
    )
    chosen = _choose_at_cap(group, 25.0)
    assert chosen is not None
    assert chosen["strategy"] == "b"
    assert chosen["budget"] == 4


def test_pareto_frontier_drops_dominated_points() -> None:
    cells = pd.DataFrame(
        [
            {"task": "t", "strategy": "a", "budget": 1, "mean_flops": 10.0, "accuracy": 0.4},
            {"task": "t", "strategy": "b", "budget": 2, "mean_flops": 20.0, "accuracy": 0.3},
            {"task": "t", "strategy": "c", "budget": 4, "mean_flops": 20.0, "accuracy": 0.6},
            {"task": "t", "strategy": "d", "budget": 8, "mean_flops": 40.0, "accuracy": 0.5},
        ]
    )
    frontier = pareto_frontier(cells)
    assert set(frontier["strategy"]) == {"a", "c"}


def test_slope_recovers_known_linear_relation() -> None:
    cells = pd.DataFrame(
        [
            {"task": "t", "strategy": "s", "model": "m", "budget": 1, "mean_flops": 10.0, "accuracy": 0.2},
            {"task": "t", "strategy": "s", "model": "m", "budget": 2, "mean_flops": 100.0, "accuracy": 0.4},
            {"task": "t", "strategy": "s", "model": "m", "budget": 4, "mean_flops": 1000.0, "accuracy": 0.6},
        ]
    )
    slopes = accuracy_vs_log_flops_slopes(cells)
    assert slopes.iloc[0]["slope_accuracy_per_decade_flops"] == pytest.approx(0.2)
    assert slopes.iloc[0]["r2"] == pytest.approx(1.0)


def test_auroc_matches_sklearn_on_known_case() -> None:
    sklearn = pytest.importorskip("sklearn.metrics")
    labels = [0, 0, 1, 1, 1, 0]
    scores = [-2.0, -1.0, 0.1, 0.2, 2.0, -0.5]
    expected = sklearn.roc_auc_score(labels, scores)
    assert metrics_from_labels(labels, scores)["auroc"] == pytest.approx(expected)


def test_ece_is_zero_for_perfectly_calibrated_extremes() -> None:
    metrics = metrics_from_labels([0, 1], [-100.0, 100.0])
    assert metrics["ece"] < 1e-6
