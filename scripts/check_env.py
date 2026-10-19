#!/usr/bin/env python
"""Print which optional backends are importable and whether a model is reachable.

Run ``python scripts/check_env.py`` for the default report.  Add ``--load`` to
actually load the configured HF checkpoint (this downloads weights and is slow).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scaletimellm.backends.registry import get_backend  # noqa: E402
from scaletimellm.config import load_config  # noqa: E402
from scaletimellm.utils.flops import estimate_flops  # noqa: E402


def importable(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def model_reachable(model_name: str) -> tuple[bool, str]:
    try:
        from huggingface_hub import model_info

        info = model_info(model_name)
        return True, f"metadata reachable (sha={info.sha[:8] if info.sha else 'unknown'})"
    except Exception as error:  # network or auth failures are expected in CI
        return False, f"{type(error).__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--load", action="store_true", help="load the HF checkpoint")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    model_name = config.model.name
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "backends": {
            "mock": True,
            "hf": importable("transformers") and importable("torch"),
            "vllm": importable("vllm"),
        },
        "optional_logging": {"wandb": importable("wandb")},
        "configured_backend": config.backend.name,
        "configured_model": model_name,
        "model_params_for_flops": config.model.n_params,
    }
    reachable, detail = model_reachable(model_name)
    report["model_reachable"] = reachable
    report["model_detail"] = detail

    if args.load:
        try:
            backend = get_backend(
                "hf",
                model_name=model_name,
                fallback_to_mock=False,
                device=config.model.device,
                dtype=config.model.dtype,
            )
            generations = backend.generate(["2+2?"], n=1, temperature=0.0, max_new_tokens=8)
            report["model_load"] = {
                "ok": True,
                "measured_n_params": backend.measured_n_params,
                "sample": generations[0][0].text,
            }
        except Exception as error:
            report["model_load"] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    else:
        report["model_load"] = {"ok": None, "detail": "skipped; pass --load to download weights"}

    mock = get_backend("mock")
    sample = mock.generate(["2+2?"], n=2, temperature=0.8, max_new_tokens=16)
    report["mock_check"] = [
        {"text": generation.text, "tokens": generation.total_tokens} for generation in sample[0]
    ]
    report["flops_example"] = estimate_flops(
        config.model.n_params,
        sample[0][0].prompt_tokens,
        sample[0][0].completion_tokens,
    )
    print(json.dumps(report, indent=2))
    print("\nBackend summary:")
    for name, ok in report["backends"].items():
        print(f"  {name:5s}: {'available' if ok else 'not installed'}")
    print(f"  model: {model_name} ({'reachable' if reachable else 'not reachable'})")


if __name__ == "__main__":
    main()
