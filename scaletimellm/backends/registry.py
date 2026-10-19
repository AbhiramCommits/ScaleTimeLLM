"""Backend factory with an explicit, loud mock fallback."""

from __future__ import annotations

import warnings
from typing import Any

from scaletimellm.backends.base import Generation, LMBackend
from scaletimellm.backends.mock import MockLM


def get_backend(name: str, **kwargs: Any) -> LMBackend:
    """Instantiate a backend by name.

    Parameters
    ----------
    name:
        ``mock``, ``hf``/``transformers``, ``vllm``, or ``auto``.  ``auto``
        tries HF first and falls back to the mock when allowed.
    **kwargs:
        Forwarded to the backend constructor.  ``fallback_to_mock`` (default
        ``True``) controls whether import/load failures fall back.
    """
    key = name.lower().strip()
    fallback = bool(kwargs.pop("fallback_to_mock", True))
    if key in {"mock", "synthetic", "simulator"}:
        return MockLM(**kwargs)
    if key in {"hf", "huggingface", "transformers"}:
        try:
            from scaletimellm.backends.hf import HFBackend

            return HFBackend(**kwargs)
        except Exception as error:
            if not fallback:
                raise
            warnings.warn(
                f"HF backend unavailable ({type(error).__name__}: {error}). "
                "Falling back to MockLM; results are SYNTHETIC, not model results.",
                RuntimeWarning,
                stacklevel=2,
            )
            return MockLM(**_mock_kwargs(kwargs))
    if key == "vllm":
        try:
            from scaletimellm.backends.vllm_backend import VLLMBackend

            return VLLMBackend(**kwargs)
        except Exception as error:
            if not fallback:
                raise
            warnings.warn(
                f"vLLM backend unavailable ({type(error).__name__}: {error}). "
                "Falling back to MockLM; results are SYNTHETIC, not model results.",
                RuntimeWarning,
                stacklevel=2,
            )
            return MockLM(**_mock_kwargs(kwargs))
    if key == "auto":
        auto_kwargs = dict(kwargs)
        auto_kwargs["fallback_to_mock"] = False
        try:
            return get_backend("hf", **auto_kwargs)
        except Exception as error:
            if not fallback:
                raise
            warnings.warn(
                f"Automatic model load failed ({type(error).__name__}: {error}). "
                "Falling back to MockLM; results are SYNTHETIC, not model results.",
                RuntimeWarning,
                stacklevel=2,
            )
            return MockLM(**_mock_kwargs(kwargs))
    raise ValueError(f"unknown backend: {name!r}")


def _mock_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate backend kwargs into the subset understood by MockLM."""
    allowed = {
        "n_params",
        "accuracy",
        "seed",
        "verbosity",
        "difficulty_adjustment",
        "mock_accuracy",
        "mock_verbosity",
    }
    out = {key: value for key, value in kwargs.items() if key in allowed}
    if "mock_accuracy" in out:
        out["accuracy"] = out.pop("mock_accuracy")
    if "mock_verbosity" in out:
        out["verbosity"] = out.pop("mock_verbosity")
    if "model_name" in kwargs:
        out.setdefault("n_params", kwargs.get("n_params", 494_032_768))
    return out


__all__ = ["Generation", "LMBackend", "get_backend"]
