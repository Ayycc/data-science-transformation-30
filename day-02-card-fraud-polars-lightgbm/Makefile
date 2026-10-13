PY ?= python3
export PYTHONPATH := src

.PHONY: setup data sample run sample-run test lint clean

setup:  ## install pinned dependencies
	$(PY) -m pip install -r requirements.txt

data:  ## generate the full SYNTHETIC stream (~5.1M transactions) into data/generated
	$(PY) scripts/generate_data.py --scale full

sample:  ## regenerate the committed 0.5 MB sample in data/sample
	$(PY) scripts/generate_data.py --scale sample

run:  ## reproduce every number and figure in README.md / results/ (~10 min on 4 vCPU)
	$(PY) scripts/run_pipeline.py --scale full

sample-run:  ## fast end-to-end run on the committed sample (writes to data/sample_results, not results/)
	$(PY) scripts/run_pipeline.py --scale sample --results data/sample_results

test:  ## pytest on the committed sample (< 1 min)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/generated data/models data/models_* data/sample_results .pytest_cache
