PY ?= python3
export PYTHONPATH := src

.PHONY: setup data run test lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only)
	$(PY) -m pip install -r requirements.txt

data:  ## seeded SYNTHETIC contracts + invoices -> data/raw/ (PDF, text, gold JSON; git-ignored) + data/sample/ (committed)
	$(PY) scripts/generate_data.py

run:  ## data -> pdfplumber -> regex/rules (+ Claude if ANTHROPIC_API_KEY is set) -> validation gate -> evaluation -> results/ (~1 min offline)
	$(PY) scripts/run_pipeline.py

test:  ## pytest: schemas, validation gate, extractors, evaluator, calibration/routing, prompt registry, LLM repair loop (fake client), e2e on sample
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/raw .pytest_cache .ruff_cache
