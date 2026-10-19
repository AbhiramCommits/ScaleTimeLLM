#!/usr/bin/env python
"""Write a small real test/coverage summary under results/ for the README."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "results" / "coverage.json"
OUT = ROOT / "results" / "test_report.json"


def main() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"(\d+)\s+tests?\s+collected", completed.stdout + completed.stderr)
    if match is None:
        raise SystemExit("could not parse pytest collection count")
    test_count = int(match.group(1))
    coverage = json.loads(COVERAGE.read_text(encoding="utf-8"))
    payload = {
        "tests_collected": test_count,
        "coverage_percent": round(float(coverage["totals"]["percent_covered"]), 2),
        "coverage_statements": int(coverage["totals"]["num_statements"]),
        "coverage_missing": int(coverage["totals"]["missing_lines"]),
        "command": "pytest -q --cov=scaletimellm --cov-report=term-missing --cov-report=json:results/coverage.json",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
