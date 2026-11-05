PY ?= python3
export PYTHONPATH := src
export OPENBLAS_NUM_THREADS := 1

.PHONY: setup data run test lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only)
	$(PY) -m pip install -r requirements.txt

data:  ## smoke-size run of the whole pipeline on a small SYNTHETIC portfolio -> data/smoke_results (git-ignored)
	$(PY) scripts/run_pipeline.py --small --data synthetic

run:  ## try HMDA, then German Credit, else SYNTHETIC -> scorecard vs GBM, fairness audit, LDA, mitigation, reason codes, governance docs -> results/
	$(PY) scripts/run_pipeline.py

test:  ## pytest: WOE/PSI, fairness metrics vs Fairlearn, mitigation, SHAP reason codes, counterfactuals, parsers, small e2e
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/raw data/smoke_results .pytest_cache .ruff_cache
