PY ?= python3
export PYTHONPATH := src

.PHONY: setup data run run-synthetic test lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only)
	$(PY) -m pip install -r requirements.txt

data:  ## download public Capital Bikeshare + NOAA + DEM data to data/raw/ (git-ignored), refresh data/sample/; SYNTHETIC fallback
	$(PY) scripts/get_data.py

run:  ## data -> H3 features -> LightGBM forecasts -> OR-Tools routes -> 14-day dock simulation -> results/ (~6 min CPU)
	$(PY) scripts/run_pipeline.py

run-synthetic:  ## same pipeline on SYNTHETIC trips over the committed station layout (offline path) -> results/synthetic/ (~22 min)
	$(PY) scripts/run_pipeline.py --synthetic --out results/synthetic

test:  ## pytest on small fixtures + the committed sample (no network)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/raw .pytest_cache .ruff_cache
