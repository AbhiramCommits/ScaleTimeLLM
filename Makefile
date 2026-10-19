.PHONY: test smoke sweep report lint

PYTHON ?= python
SMOKE_OUT ?= results/smoke

test:
	@mkdir -p results
	$(PYTHON) -m pytest -q --cov=scaletimellm --cov-report=term-missing --cov-report=json:results/coverage.json
	$(PYTHON) scripts/write_test_report.py

smoke:
	$(PYTHON) -m scaletimellm.cli smoke --output-dir $(SMOKE_OUT)

sweep:
	$(PYTHON) -m scaletimellm.cli sweep --config configs/full.yaml

report:
	$(PYTHON) -m scaletimellm.cli report

lint:
	$(PYTHON) -m ruff check .
