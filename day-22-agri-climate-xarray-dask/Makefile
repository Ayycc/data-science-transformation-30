PY ?= python3
export PYTHONPATH := src

.PHONY: setup data run test lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only, no Dask distributed cluster)
	$(PY) -m pip install -r requirements.txt

data:  ## ERA5 via Open-Meteo if reachable, else seeded SYNTHETIC cube -> data/cube.zarr (git-ignored); smoke-size run
	$(PY) scripts/run_pipeline.py --small --no-out-of-core

run:  ## cube (Zarr) -> Dask indices -> zonal stats -> yield CV (random vs spatial) -> +2C/-20% scenario -> perf + out-of-core demo -> results/ (~6 min)
	$(PY) scripts/run_pipeline.py

test:  ## pytest: generator, indices (GDD/SPI/seasonal sums), zonal stats vs rioxarray clip, model/CV/risk, public parser, small e2e incl. capped child process
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/cube.zarr data/big_cube.zarr data/ooc .pytest_cache .ruff_cache
