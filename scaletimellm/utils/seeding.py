"""Seeding utilities shared by every experiment."""

from __future__ import annotations

import os
import random

import numpy as np


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy and (when installed) PyTorch deterministically.

    Note that ``PYTHONHASHSEED`` only takes effect for child processes started
    after this call.  The function sets it for completeness and also configures
    deterministic PyTorch algorithms where the platform supports them.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
