PY ?= python3
export PYTHONPATH := src
export OPENBLAS_NUM_THREADS := 1

.PHONY: setup data run test lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only)
	$(PY) -m pip install -r requirements.txt

data:  ## smoke-size run of the whole pipeline (SYNTHETIC simulator, 2 seeds x 120 nights) -> data/smoke_results (git-ignored)
	$(PY) scripts/run_pipeline.py --small --workers 1

run:  ## logs -> elasticity (OLS / randomised / 2SLS) -> 8 policies x 20 seeds x 730 nights -> competitor reaction + OPE -> results/
	$(PY) scripts/run_pipeline.py

test:  ## pytest: simulator, DP/bid prices, estimators, guardrails (never violated), bandit learning, OPE, small e2e
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/generated data/smoke_results .pytest_cache .ruff_cache
