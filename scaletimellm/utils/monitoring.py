"""Optional Weights & Biases logging, disabled unless explicitly enabled."""

from __future__ import annotations

import os
import warnings
from typing import Any

from scaletimellm.config import ExperimentConfig


def wandb_enabled(config: ExperimentConfig | None = None) -> bool:
    """Return whether W&B logging is enabled.

    W&B is OFF by default.  It can be enabled either in YAML (``wandb: true``)
    or with ``SCALETIMELLM_WANDB=1``.  The environment variable is convenient for
    CI/forks that should never contact the service.
    """
    env_value = os.environ.get("SCALETIMELLM_WANDB", "0").lower()
    env_enabled = env_value in {"1", "true", "yes", "on"}
    return env_enabled or bool(config.wandb)


def maybe_init_wandb(
    config: ExperimentConfig,
    *,
    run_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Any | None:
    """Initialise a W&B run only when explicitly enabled.

    Returns ``None`` when logging is disabled or ``wandb`` is not installed.
    Importing is deliberately lazy so the package has no runtime dependency on
    W&B and offline reproduction never attempts a network connection.
    """
    if not wandb_enabled(config):
        os.environ.setdefault("WANDB_MODE", "disabled")
        return None
    try:
        import wandb
    except ImportError:
        warnings.warn(
            "W&B logging requested but wandb is not installed; continuing without logging.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    payload = config.to_dict()
    if extra:
        payload.update(extra)
    return wandb.init(project=config.wandb_project, name=run_name, config=payload)
