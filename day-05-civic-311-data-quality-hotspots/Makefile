PY ?= python3
export PYTHONPATH := src

.PHONY: setup data sample run sample-run test lint clean

setup:  ## install pinned dependencies
	$(PY) -m pip install -r requirements.txt

data:  ## probe the public 311 APIs, then generate the full SYNTHETIC extract (~1M requests) into data/generated
	$(PY) scripts/generate_data.py --scale full

sample:  ## regenerate the committed sample in data/sample (~8k requests)
	$(PY) scripts/generate_data.py --scale sample

run:  ## one command reproduces every number and figure in README.md / results/
	$(PY) scripts/run_pipeline.py --scale full

sample-run:  ## fast end-to-end run on the committed sample (writes to data/sample_results, not results/)
	$(PY) scripts/run_pipeline.py --scale sample --results data/sample_results

test:  ## pytest: contract rules, GE suite, harmonisation, Gi*, anomaly detector, end-to-end run on data/sample
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/generated data/sample_results data/sample_work .pytest_cache
