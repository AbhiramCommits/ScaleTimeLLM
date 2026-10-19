"""Strategy registry."""

from __future__ import annotations

from typing import Any

from scaletimellm.strategies.base import Strategy, StrategyResult
from scaletimellm.strategies.best_of_n import BestOfNStrategy
from scaletimellm.strategies.greedy import GreedyStrategy
from scaletimellm.strategies.self_consistency import SelfConsistencyStrategy
from scaletimellm.strategies.step_search import StepSearchStrategy
from scaletimellm.strategies.weighted_vote import WeightedVoteStrategy


def get_strategy(name: str, **kwargs: Any) -> Strategy:
    """Instantiate a strategy by registry name."""
    key = name.lower().strip()
    if key == "greedy":
        return GreedyStrategy(**kwargs)
    if key in {"self_consistency", "self-consistency", "sc"}:
        return SelfConsistencyStrategy(**kwargs)
    if key in {"best_of_n", "best-of-n", "bon"}:
        return BestOfNStrategy(**kwargs)
    if key in {"weighted_vote", "weighted-vote"}:
        return WeightedVoteStrategy(**kwargs)
    if key in {"step_search", "step-search"}:
        return StepSearchStrategy(**kwargs)
    raise ValueError(f"unknown strategy: {name!r}")


__all__ = ["Strategy", "StrategyResult", "get_strategy"]
