PY ?= python3
export PYTHONPATH := src
export DBT_SEND_ANONYMOUS_USAGE_STATS := false

.PHONY: setup data contracts run test contract-gate lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only, no Docker or lineage server)
	$(PY) -m pip install -r requirements.txt

data:  ## seeded SYNTHETIC domain feeds -> data/raw/ (git-ignored) + data/sample/ (committed)
	$(PY) scripts/generate_data.py

contracts:  ## regenerate dbt model-contract YAML from contracts/*.odcs.yaml
	$(PY) -c "from mesh.warehouse import render_contract_yaml; print(render_contract_yaml())"

run:  ## data -> DuckDB -> dbt build (3 domains + consumer) -> SLOs -> OpenLineage + column lineage -> CI gate replay -> results/ (~1 min)
	$(PY) scripts/run_pipeline.py

test:  ## pytest: generator, contracts, CI gate (deliberate breaking change), consumer replay, dbt e2e on the sample (~1 min)
	$(PY) -m pytest -q

contract-gate:  ## CI gate vs a git ref: make contract-gate BASE=origin/main
	$(PY) scripts/check_contracts.py --base $(or $(BASE),HEAD) --report contract-gate.json

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/raw data/warehouse dbt/*/target dbt/*/logs logs contract-gate.json .pytest_cache .ruff_cache
