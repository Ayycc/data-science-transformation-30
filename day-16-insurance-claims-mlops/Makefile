PY ?= python3
export PYTHONPATH := src
export MLFLOW_DISABLE_TELEMETRY := true
IMAGE ?= claims-pure-premium:local

.PHONY: setup data run test lint gate serve docker-build clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only)
	$(PY) -m pip install -r requirements.txt

data:  ## OpenML freMTPL2 if reachable, else seeded SYNTHETIC book -> data/raw/ (git-ignored) + data/sample/
	$(PY) scripts/get_data.py

run:  ## full lifecycle: MLflow (local SQLite) training + gated registry, API batch scoring, Evidently monitoring, retraining -> results/ (~1.5 min CPU)
	$(PY) scripts/run_pipeline.py

test:  ## pytest: API contract (TestClient), gate/registry, drift detection, data (no network, no server)
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

gate:  ## CI model-validation gate: GBM challenger must beat the GLM champion in a fresh local registry (exit 1 otherwise)
	$(PY) scripts/validate_gate.py

serve:  ## serve the exported champion (after make run) on :8000; Ctrl+C to stop
	CLAIMS_MODEL_URI=models/champion $(PY) -m uvicorn claims_mlops.api:app --host 127.0.0.1 --port 8000

docker-build:  ## optional artefact: build the scoring image (needs a Docker daemon and models/champion from make run)
	docker build -t $(IMAGE) .

clean:
	rm -rf data/raw data/reports mlruns models .pytest_cache .ruff_cache
