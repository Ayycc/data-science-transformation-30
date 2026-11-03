PY ?= python3
export PYTHONPATH := src
export OPENBLAS_NUM_THREADS := 1

.PHONY: setup data run test lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU-only torch wheel index; no vector DB server or Docker)
	$(PY) -m pip install -r requirements.txt

data:  ## smoke-size run (tiny SYNTHETIC world) of the full pipeline -> data/smoke_results (git-ignored)
	$(PY) scripts/run_pipeline.py --small --data synthetic

run:  ## try MovieLens-1M, else SYNTHETIC world -> baselines, two-tower + FAISS, re-ranker, offline metrics, OPE -> results/ (~10 min)
	$(PY) scripts/run_pipeline.py

test:  ## pytest: generator, metrics, baselines, two-tower, FAISS, re-ranker/MMR, OPE estimators, MovieLens parser, small e2e
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/raw data/smoke_results .pytest_cache .ruff_cache
