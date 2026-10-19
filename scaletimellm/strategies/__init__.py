"""Inference-time scaling strategies."""

from scaletimellm.strategies.base import Strategy, StrategyResult
from scaletimellm.strategies.registry import get_strategy

__all__ = ["Strategy", "StrategyResult", "get_strategy"]
