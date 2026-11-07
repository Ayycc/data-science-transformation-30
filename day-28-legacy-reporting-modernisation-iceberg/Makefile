PY ?= python3
export PYTHONPATH := src

.PHONY: setup data run test lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only; no SAS, Spark, catalog server or Docker)
	$(PY) -m pip install -r requirements.txt

data:  ## smoke-size run of the whole pipeline (2,000 SYNTHETIC policies) -> data/smoke_results (git-ignored)
	$(PY) scripts/run_pipeline.py --small

run:  ## SYNTHETIC extracts -> 8 quarterly closes (legacy vs Ibis, reconcile, Iceberg, time travel) -> benchmark -> results/
	$(PY) scripts/run_pipeline.py

test:  ## pytest: legacy quirks, compat == legacy, DuckDB == Polars, reconciliation predicates, Iceberg time travel + schema evolution
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/lakehouse data/smoke data/smoke_results data/tmp_csv .pytest_cache .ruff_cache
