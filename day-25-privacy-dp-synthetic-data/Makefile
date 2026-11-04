PY ?= python3
export PYTHONPATH := src
export OPENBLAS_NUM_THREADS := 1

.PHONY: setup data run test lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only, no PyTorch)
	$(PY) -m pip install -r requirements.txt

data:  ## smoke-size run of the whole pipeline on a tiny SYNTHETIC population -> data/smoke_results (git-ignored)
	$(PY) scripts/run_pipeline.py --small --data synthetic

run:  ## try UCI Adult, else SYNTHETIC population -> DP tables, query server, DP model, synthesisers, risk, memo -> results/
	$(PY) scripts/run_pipeline.py

test:  ## pytest: accountant, mechanisms, zCDP, query server, DP model, synthesisers, SDC, risk metrics, small e2e
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/raw data/smoke_results .pytest_cache .ruff_cache
