PY ?= python3
export PYTHONPATH := src
export DAGSTER_HOME := $(CURDIR)/data/dagster_home

.PHONY: setup data run test dev lint clean

setup:  ## install pinned dependencies
	$(PY) -m pip install -r requirements.txt

data:  ## SYNTHETIC month for 300 stations into data/ (git-ignored) + refresh data/sample/
	$(PY) scripts/generate_data.py

run:  ## regenerate data, 30-partition backfill, late-data sensor, range backfill, report, metrics (~2 min)
	$(PY) scripts/run_pipeline.py

test:  ## pytest on a tiny SYNTHETIC preset (20 stations x 30 days)
	$(PY) -m pytest -q

dev:  ## Dagster UI on http://localhost:3000 over the warehouse built by make run
	mkdir -p $(DAGSTER_HOME)
	$(PY) -m dagster dev -m air_quality.definitions

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/landing data/late_arrivals data/warehouse data/dagster_home data/truth data/reference data/range_demo data/.transient .pytest_cache .ruff_cache
