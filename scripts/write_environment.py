#!/usr/bin/env python
"""Record the exact runtime environment used for the committed artifacts."""

from __future__ import annotations

import json
import platform
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    packages = [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "torch",
        "transformers",
        "datasets",
        "trl",
        "scikit-learn",
        "PyYAML",
        "pytest",
        "ruff",
        "vllm",
        "wandb",
    ]
    payload = {
        "python": sys.version,
        "python_short": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "packages": {name: version(name) for name in packages},
        "backends": {
            "mock": True,
            "hf": version("transformers") is not None and version("torch") is not None,
            "vllm": version("vllm") is not None,
        },
        "model_weights_used_for_full_sweep": False,
        "hf_serving_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "hf_serving_device": "mps",
        "wandb_default": "disabled",
    }
    path = ROOT / "results" / "environment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
