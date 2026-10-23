PY ?= python3
export PYTHONPATH := src

.PHONY: setup data run test lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only)
	$(PY) -m pip install -r requirements.txt

data:  ## probe the public IBM AML dataset, write full SYNTHETIC graph to data/raw/ (git-ignored), refresh data/sample/
	$(PY) scripts/generate_data.py

run:  ## temporal graph, communities, motif detectors, tabular vs graph models, explainable alerts -> results/ (~2-3 min CPU)
	$(PY) scripts/run_pipeline.py

test:  ## pytest on small SYNTHETIC fixtures + the committed sample (no network)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/raw .pytest_cache .ruff_cache
