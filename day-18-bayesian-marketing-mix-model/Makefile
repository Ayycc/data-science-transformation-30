PY ?= python3
export PYTHONPATH := src

.PHONY: setup data run test lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only, needs gcc/g++ for PyTensor)
	$(PY) -m pip install -r requirements.txt

data:  ## seeded SYNTHETIC panel -> data/raw/ (git-ignored) + data/sample/ (committed)
	$(PY) scripts/get_data.py

run:  ## data -> prior predictive -> NUTS (4 x 1000) -> diagnostics, recovery, ROAS -> budget optimiser -> results/ (~6-8 min CPU)
	$(PY) scripts/run_pipeline.py

test:  ## pytest: generator, transforms vs PyMC-Marketing, optimiser, tiny-chain model smoke test (~1 min)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/raw .pytest_cache .ruff_cache
