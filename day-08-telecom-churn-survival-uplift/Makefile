PY ?= python3
export PYTHONPATH := src

.PHONY: setup data run test lint clean

setup:  ## install pinned dependencies
	$(PY) -m pip install -r requirements.txt

data:  ## regenerate the SYNTHETIC panel (data/generated, git-ignored) and the committed 5k-customer sample
	$(PY) scripts/generate_data.py

run:  ## reproduce every number and figure in README.md / results/ (~2 min on 4 vCPU)
	$(PY) scripts/run_pipeline.py

test:  ## fast pytest suite (small fixtures, < 3 min)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/generated .pytest_cache .ruff_cache
