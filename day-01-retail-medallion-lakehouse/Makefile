PY ?= python3
export PYTHONPATH := src
export DO_NOT_TRACK := 1

.PHONY: setup data run sample-run test lint docs clean

setup:  ## install pinned dependencies
	$(PY) -m pip install -r requirements.txt

data:  ## generate the full SYNTHETIC landing zone (~2M order lines) into data/generated
	$(PY) scripts/generate_data.py --scale full

run:  ## reproduce every number and figure in README.md / results/
	$(PY) scripts/run_pipeline.py --scale full

sample-run:  ## fast end-to-end run on the committed 0.8 MB sample (writes to data/, not results/)
	$(PY) scripts/run_pipeline.py --scale sample --db data/warehouse/sample.duckdb --results data/sample_results

test:  ## pytest: generator, bronze loader, and dbt build + acceptance checks on the sample
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

docs:  ## serve dbt docs for the last build
	cd dbt && DUCKDB_PATH=../data/warehouse/retail.duckdb dbt docs serve --profiles-dir .

clean:
	rm -rf data/generated data/warehouse data/sample_results dbt/target dbt/logs .pytest_cache
