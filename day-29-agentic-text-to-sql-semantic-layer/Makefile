PY ?= python3
export PYTHONPATH := src

.PHONY: setup data run test lint ask clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only; in-process DuckDB, no server or Docker)
	$(PY) -m pip install -r requirements.txt

data:  ## build the SYNTHETIC warehouse (data/warehouse.duckdb, git-ignored) and the committed CSV sample
	$(PY) -c "from pathlib import Path; from semsql import generator as g; t = g.generate(); print(g.write_duckdb(t, Path('data/warehouse.duckdb'))); print(g.write_sample(t, Path('data/sample')), 'sample bytes')"

run:  ## warehouse -> 60-question benchmark (semantic vs raw schema; Claude too if ANTHROPIC_API_KEY is set) -> red team -> results/
	$(PY) scripts/run_all.py

test:  ## pytest: guardrails (no write/DDL can execute), compiler, planner, benchmark, mocked Claude tool loop
	$(PY) -m pytest -q

ask:  ## ask one question, e.g. make ask Q="Revenue by region in FY2025"
	$(PY) scripts/ask.py "$(Q)"

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/warehouse.duckdb data/warehouse.duckdb.wal .pytest_cache .ruff_cache
