"""Configuration, seeding and FLOP-accounting tests."""

from __future__ import annotations

import random

import numpy as np
import pytest

from scaletimellm.backends.mock import MockLM
from scaletimellm.config import ExperimentConfig, load_config
from scaletimellm.utils.flops import cost_usd, estimate_flops
from scaletimellm.utils.seeding import set_all_seeds


def test_load_smoke_config() -> None:
    config = load_config("configs/smoke.yaml")
    assert isinstance(config, ExperimentConfig)
    assert config.backend.name == "mock"
    assert config.seeds == [0]
    assert config.budgets == [1, 2, 4]
    assert config.smoke is True


def test_config_rejects_unknown_key(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("task: synthetic\nnot_a_field: 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_config(path)


def test_config_rejects_invalid_budget(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("task: synthetic\nbudgets: [0, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="budgets"):
        load_config(path)


def test_config_resolution_and_roundtrip() -> None:
    config = ExperimentConfig(
        task="synthetic",
        strategy="greedy",
        tasks=["synthetic", "chartqa"],
        strategies=["greedy", "self_consistency"],
    )
    assert config.resolved_tasks() == ["synthetic", "chartqa"]
    assert config.resolved_strategies() == ["greedy", "self_consistency"]
    assert config.resolved_models()[0].name == config.model.name
    assert config.to_dict()["task"] == "synthetic"


def test_seeding_is_deterministic() -> None:
    set_all_seeds(123)
    first = (random.random(), float(np.random.rand()), MockLM(seed=0).generate(["2+2?"], 1)[0][0].text)
    set_all_seeds(123)
    second = (random.random(), float(np.random.rand()), MockLM(seed=0).generate(["2+2?"], 1)[0][0].text)
    assert first == second
    try:
        import torch

        set_all_seeds(7)
        a = torch.rand(3)
        set_all_seeds(7)
        b = torch.rand(3)
        assert torch.equal(a, b)
    except ImportError:
        pass


def test_flops_formula_and_errors() -> None:
    assert estimate_flops(1_000, 10, 5) == 30_000.0
    with pytest.raises(ValueError):
        estimate_flops(0, 1, 1)
    with pytest.raises(ValueError):
        estimate_flops(100, -1, 1)


def test_cost_per_million_tokens() -> None:
    assert cost_usd(1_000_000, 0, 0.15, 0.60) == pytest.approx(0.15)
    assert cost_usd(0, 1_000_000, 0.15, 0.60) == pytest.approx(0.60)
    assert cost_usd(100, 100, 0.15, 0.60) == pytest.approx(0.000075)
