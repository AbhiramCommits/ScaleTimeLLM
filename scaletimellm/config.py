"""Typed configuration objects and a YAML loader.

Every experiment is reproducible from a YAML file.  The public entry point is
:func:`load_config`; :class:`ExperimentConfig` is also constructible directly in
Python and in tests.  Unknown keys are rejected so a typo cannot silently change
an experiment.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_INPUT_PRICE_PER_MILLION = 0.15
DEFAULT_OUTPUT_PRICE_PER_MILLION = 0.60
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_N_PARAMS = 494_032_768


@dataclass
class ModelConfig:
    """A model identity and the parameter count used for FLOP accounting.

    ``n_params`` is explicit rather than inferred so that mock runs can model
    hypothetical model sizes and so that the FLOPs calculation is reproducible
    even if a checkpoint is unavailable.  When the HF backend loads a model it
    replaces this value with the measured parameter count.
    """

    name: str = DEFAULT_MODEL
    n_params: int = DEFAULT_N_PARAMS
    dtype: str = "float32"
    device: str = "auto"
    trust_remote_code: bool = False
    mock_accuracy: float | None = None
    mock_verbosity: float = 1.0

    def __post_init__(self) -> None:
        if self.n_params <= 0:
            raise ValueError("model.n_params must be positive")
        if self.mock_accuracy is not None and not 0.0 <= self.mock_accuracy <= 1.0:
            raise ValueError("model.mock_accuracy must be in [0, 1]")
        if self.mock_verbosity <= 0:
            raise ValueError("model.mock_verbosity must be positive")


@dataclass
class BackendConfig:
    """Backend selection and mock-backend controls."""

    name: str = "mock"
    mock_accuracy: float = 0.42
    mock_verbosity: float = 1.0
    fallback_to_mock: bool = True
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.80

    def __post_init__(self) -> None:
        if not 0.0 <= self.mock_accuracy <= 1.0:
            raise ValueError("backend.mock_accuracy must be in [0, 1]")
        if self.mock_verbosity <= 0:
            raise ValueError("backend.mock_verbosity must be positive")


@dataclass
class ExperimentConfig:
    """Complete specification of one run or a cross-product sweep.

    The fields named in the paper-facing interface are ``model``, ``backend``,
    ``task``, ``strategy``, ``budgets``, ``seeds``, ``max_new_tokens``,
    ``temperature`` and ``n_problems``.  The plural ``models``, ``tasks`` and
    ``strategies`` fields optionally turn the configuration into a sweep; when
    they are absent the singular fields are used.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    task: str = "synthetic"
    strategy: str = "greedy"
    budgets: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    max_new_tokens: int = 128
    temperature: float = 0.8
    n_problems: int = 16

    name: str = "base"
    tasks: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    models: list[ModelConfig] = field(default_factory=list)
    verifier: str = "outcome"
    verifier_path: str = "artifacts/verifier"
    difficulty_pilot_k: int = 4
    data_dir: str = "data"
    output_dir: str = "results"
    cache_dir: str = "results/cache"
    price_input_per_million: float = DEFAULT_INPUT_PRICE_PER_MILLION
    price_output_per_million: float = DEFAULT_OUTPUT_PRICE_PER_MILLION
    wandb: bool = False
    wandb_project: str = "scaletimellm"
    smoke: bool = False
    sandbox_timeout_s: float = 2.0
    offline: bool = False

    def __post_init__(self) -> None:
        if not self.budgets or any(int(b) <= 0 for b in self.budgets):
            raise ValueError("budgets must be a non-empty list of positive integers")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if self.n_problems <= 0:
            raise ValueError("n_problems must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if self.price_input_per_million < 0 or self.price_output_per_million < 0:
            raise ValueError("prices must be non-negative")
        if self.difficulty_pilot_k <= 0:
            raise ValueError("difficulty_pilot_k must be positive")

    def resolved_models(self) -> list[ModelConfig]:
        """Return the model list for a sweep (or the single model)."""
        return list(self.models) if self.models else [self.model]

    def resolved_tasks(self) -> list[str]:
        """Return the task list for a sweep (or the single task)."""
        return list(self.tasks) if self.tasks else [self.task]

    def resolved_strategies(self) -> list[str]:
        """Return the strategy list for a sweep (or the single strategy)."""
        return list(self.strategies) if self.strategies else [self.strategy]

    def to_dict(self) -> dict[str, Any]:
        """JSON/YAML-serialisable representation."""
        return dataclasses.asdict(self)


# Mapping of accepted YAML keys to nested dataclass constructors.
_MODEL_FIELDS = {f.name for f in dataclasses.fields(ModelConfig)}
_BACKEND_FIELDS = {f.name for f in dataclasses.fields(BackendConfig)}
_CONFIG_FIELDS = {f.name for f in dataclasses.fields(ExperimentConfig)}


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {', '.join(unknown)}")


def _parse_model(value: Any, where: str) -> ModelConfig:
    if isinstance(value, ModelConfig):
        return value
    if isinstance(value, str):
        return ModelConfig(name=value)
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping or string")
    _reject_unknown(value, _MODEL_FIELDS, where)
    return ModelConfig(**value)


def _parse_backend(value: Any) -> BackendConfig:
    if isinstance(value, BackendConfig):
        return value
    if isinstance(value, str):
        return BackendConfig(name=value)
    if not isinstance(value, dict):
        raise ValueError("backend must be a mapping or string")
    _reject_unknown(value, _BACKEND_FIELDS, "backend")
    return BackendConfig(**value)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from a parsed YAML mapping."""
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    _reject_unknown(raw, _CONFIG_FIELDS, "configuration")
    values = dict(raw)
    if "model" in values:
        values["model"] = _parse_model(values["model"], "model")
    if "models" in values:
        models = values["models"]
        if not isinstance(models, list):
            raise ValueError("models must be a list")
        values["models"] = [_parse_model(m, "models[]") for m in models]
    if "backend" in values:
        values["backend"] = _parse_backend(values["backend"])
    for key in ("tasks", "strategies", "budgets", "seeds"):
        if key in values and not isinstance(values[key], list):
            raise ValueError(f"{key} must be a list")
    return ExperimentConfig(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = config_from_dict(raw)
    config.name = config.name or path.stem
    return config
