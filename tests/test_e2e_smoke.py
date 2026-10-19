"""End-to-end smoke pipeline test."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def test_e2e_smoke_pipeline(tmp_path) -> None:
    output = tmp_path / "smoke"
    started = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scaletimellm.cli",
            "smoke",
            "--output-dir",
            str(output),
            "--bootstrap",
            "100",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    elapsed = time.perf_counter() - started
    assert completed.returncode == 0, completed.stderr
    assert elapsed < 60
    summary_path = output / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["smoke"] is True
    assert summary["backend"] == "mock"
    assert summary["synthetic"] is True
    assert summary["cells"] >= 4
    assert summary["records"] >= 12
    assert (output / "aggregate.csv").exists()
    assert Path(summary["raw_dir"]).exists()
