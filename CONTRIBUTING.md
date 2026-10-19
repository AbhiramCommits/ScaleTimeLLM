# Contributing to ScaleTimeLLM

Thanks for helping improve the repository. Please keep changes small, reproducible and measured.

## Development setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/ruff check .
.venv/bin/pytest -q
```

Experiments must run from YAML configs. New results should be written under `results/` as raw JSONL
plus regenerated aggregate artifacts. Never hand-edit a number in `README.md`, `paper/main.md` or
`results/REPORT.md`; update the raw records and rerun the report.

## Pull requests

- Add or update tests for behavioral changes.
- Keep the mock backend deterministic and label synthetic artifacts.
- Run `make lint && make test` before opening a PR.
- Do not commit model weights, Hugging Face caches or secrets.
- W&B is optional and disabled by default; do not make CI depend on it.

## Commit style

Use short imperative commit messages (`feat: ...`, `fix: ...`, `docs: ...`, `test: ...`).
