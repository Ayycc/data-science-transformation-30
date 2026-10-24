PY ?= python3
export PYTHONPATH := src

.PHONY: setup data run run-quick test lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only, no broker or Docker needed)
	$(PY) -m pip install -r requirements.txt

data:  ## probe public NASA C-MAPSS, refresh SYNTHETIC data/sample/, write full SYNTHETIC stream to data/raw/ (git-ignored)
	$(PY) scripts/generate_data.py

run:  ## simulate 500 machines x 14 days -> asyncio stream -> online models -> DuckDB -> results/ (~5 min CPU)
	$(PY) scripts/run_pipeline.py

run-quick:  ## same pipeline on the committed SYNTHETIC sample (seconds), writes data/quick_results/
	$(PY) scripts/run_pipeline.py --quick

test:  ## pytest on small SYNTHETIC fixtures + the committed sample (no network, no broker)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/raw data/quick_results data/*.duckdb data/*.duckdb.wal .pytest_cache .ruff_cache
