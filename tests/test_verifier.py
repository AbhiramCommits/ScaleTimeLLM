"""Verifier training, checkpoint and metric tests."""

from __future__ import annotations

import json
import math

from scaletimellm.config import BackendConfig, ExperimentConfig, ModelConfig
from scaletimellm.verifier.data import preference_pairs
from scaletimellm.verifier.eval import metrics_from_labels, top1_selection
from scaletimellm.verifier.score import (
    LengthPriorVerifier,
    OracleVerifier,
    Verifier,
)
from scaletimellm.verifier.train import train_verifiers


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="verifier-test",
        model=ModelConfig(
            name="mock-tiny",
            n_params=8_000_000,
            mock_accuracy=0.5,
        ),
        backend=BackendConfig(
            name="mock", mock_accuracy=0.5, mock_verbosity=0.4
        ),
        task="synthetic",
        budgets=[1],
        seeds=[0],
        max_new_tokens=48,
        n_problems=10,
        offline=True,
        smoke=True,
    )


def test_verifier_trains_and_checkpoint_roundtrips(tmp_path) -> None:
    result = train_verifiers(
        _config(),
        n_train_problems=12,
        n_eval_problems=5,
        k=3,
        epochs=5,
        seed=0,
        artifact_dir=tmp_path / "verifier",
        results_path=tmp_path / "metrics.json",
        train_trl=False,
    )
    assert result["objectives"]["outcome"]["auroc"] > 0.5
    assert 0.0 <= result["objectives"]["outcome"]["ece"] <= 1.0
    checkpoint = tmp_path / "verifier" / "outcome"
    verifier = Verifier.from_checkpoint(checkpoint, objective="outcome")
    score = verifier.score("Question: 2+2?", "#### 4")
    assert math.isfinite(score)
    assert verifier.info()["kind"] == "custom"
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert "objectives" in metrics and metrics["data"]["synthetic"] is True
    assert (tmp_path / "verifier" / "pairwise" / "model.pt").exists()


def test_preference_pairs_are_chosen_positive() -> None:
    rows = [
        {"problem_id": "p", "label": 1, "prompt": "q", "completion": "good"},
        {"problem_id": "p", "label": 0, "prompt": "q", "completion": "bad"},
    ]
    pairs = preference_pairs(rows)
    assert pairs == [{"chosen": "q\n\ngood", "rejected": "q\n\nbad"}]


def test_verifier_metrics_known_case() -> None:
    labels = [0, 0, 1, 1]
    scores = [-2.0, -1.0, 1.0, 2.0]
    metrics = metrics_from_labels(labels, scores)
    assert metrics["auroc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert 0.0 <= metrics["ece"] <= 1.0


def test_top1_selection_and_random_baseline() -> None:
    rows = [
        {"problem_id": "a", "label": 1},
        {"problem_id": "a", "label": 0},
        {"problem_id": "b", "label": 1},
        {"problem_id": "b", "label": 0},
    ]
    top1, random_expected, n = top1_selection(rows, [0.9, 0.1, 0.2, 0.8])
    assert n == 2
    assert top1 == 0.5
    assert random_expected == 0.5


def test_length_prior_is_deterministic() -> None:
    verifier = LengthPriorVerifier(target_tokens=4)
    assert verifier.score("q", "a b c d") == 0.0
    assert verifier.score("q", "a b") < verifier.score("q", "a b c d")


def test_oracle_verifier_uses_task_grader() -> None:
    from scaletimellm.tasks.synthetic import SyntheticTask

    task = SyntheticTask()
    problem = task.load(1, seed=0)[0]
    verifier = OracleVerifier(task, problem)
    assert verifier.score("q", f"#### {problem.gold}") == 1.0
    assert verifier.score("q", "#### -999999") == 0.0
