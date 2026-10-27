PY ?= python3
export PYTHONPATH := src
export FEAST_USAGE := False

.PHONY: setup data run test lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only)
	$(PY) -m pip install -r requirements.txt

data:  ## NYC TLC if reachable, else seeded SYNTHETIC city -> data/raw/ (git-ignored) + data/sample/
	$(PY) scripts/get_data.py

run:  ## data -> DuckDB features -> Feast (DuckDB offline, SQLite online) -> naive vs PIT models, online replay, skew test, latency -> results/ (~2 min CPU)
	$(PY) scripts/run_pipeline.py

test:  ## pytest: PIT correctness, feature windows, skew, latency, data (no network, no server)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/raw data/offline data/store .pytest_cache .ruff_cache
