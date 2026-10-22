PY ?= python3
export PYTHONPATH := src

.PHONY: setup data run test lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only)
	$(PY) -m pip install -r requirements.txt

data:  ## write full SYNTHETIC data to data/raw/ (git-ignored), refresh data/sample/, try to cache public LaLonde files
	$(PY) scripts/generate_data.py

run:  ## DAG + identification, PS methods, DML/causal forest, refutations, staggered DiD, synthetic control, LaLonde -> results/ (~6 min CPU)
	$(PY) scripts/run_pipeline.py

test:  ## pytest on small SYNTHETIC fixtures + the committed LaLonde sample (no network)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/raw .pytest_cache .ruff_cache
