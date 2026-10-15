PY ?= python3
export PYTHONPATH := src

.PHONY: setup data sample run sample-run test lint clean

setup:  ## install pinned dependencies
	$(PY) -m pip install -r requirements.txt

data:  ## probe EIA-930, then generate the full SYNTHETIC dataset (5 regions x 4 years hourly) into data/generated
	$(PY) scripts/generate_data.py --scale full

sample:  ## regenerate the committed sample in data/sample (2 regions, 7 months)
	$(PY) scripts/generate_data.py --scale sample

run:  ## the backtest: one command reproduces every number and figure in README.md / results/
	$(PY) scripts/run_backtest.py --scale full

sample-run:  ## fast end-to-end run on the committed sample (writes to data/sample_results, not results/)
	$(PY) scripts/run_backtest.py --scale sample --results data/sample_results

test:  ## pytest: metrics, features (no leakage), conformal, generator determinism, smoke backtest on data/sample
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/generated data/sample_results .pytest_cache
