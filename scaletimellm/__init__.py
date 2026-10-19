"""ScaleTimeLLM: inference-time scaling experiments for small open LLMs."""

__version__ = "0.1.0"

from scaletimellm.config import ExperimentConfig, load_config

__all__ = ["ExperimentConfig", "load_config", "__version__"]
