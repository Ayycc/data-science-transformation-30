PY ?= python3
export PYTHONPATH := src

.PHONY: setup data run test lint clean

setup:  ## install pinned dependencies (includes the spaCy en_core_web_md wheel)
	$(PY) -m pip install -r requirements.txt

data:  ## download + cache the Bitext corpus into data/raw/ (git-ignored); falls back to SYNTHETIC at run time
	$(PY) scripts/fetch_data.py

run:  ## redaction, baselines, active learning, abstention, topics -> results/ (~10 min CPU)
	$(PY) scripts/run_pipeline.py

test:  ## pytest on small SYNTHETIC fixtures
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/raw .pytest_cache .ruff_cache
