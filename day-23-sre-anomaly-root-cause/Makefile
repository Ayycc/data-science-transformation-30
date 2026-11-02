PY ?= python3
export PYTHONPATH := src

.PHONY: setup data run test lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only; no Prometheus/Grafana/Docker needed)
	$(PY) -m pip install -r requirements.txt

data:  ## smoke-size run of the SYNTHETIC generator + full pipeline -> data/smoke_results (git-ignored)
	$(PY) scripts/run_pipeline.py --small

run:  ## generate SYNTHETIC telemetry -> detect (univariate, IForest, ECOD) -> group -> RCA -> evaluate (+ NAB check if reachable) -> results/ (~1 min)
	$(PY) scripts/run_pipeline.py

test:  ## pytest: generator, z-scores/detectors, grouping, change points, PC/Granger RCA, deploy prior, NAB parser, small e2e
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/nab data/smoke_results .pytest_cache .ruff_cache
