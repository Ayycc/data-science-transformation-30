PY ?= python3
export PYTHONPATH := src

.PHONY: setup data run test lint clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only; includes the en_core_web_md vector wheel from GitHub)
	$(PY) -m pip install -r requirements.txt

data:  ## eCFR 16 CFR 312 if reachable, else seeded SYNTHETIC regulation + gold Q&A -> data/raw/ (git-ignored) + data/sample/ (committed)
	$(PY) scripts/generate_data.py

run:  ## data -> chunk (structure vs fixed) -> BM25 / dense / hybrid RRF / +PRF / +re-rank -> guardrail -> cited answers (Claude if ANTHROPIC_API_KEY, else extractive) -> eval -> results/ (~30 s offline)
	$(PY) scripts/run_pipeline.py

test:  ## pytest: generator, chunking, retrieval, RRF, re-ranker cues, generators (fake Claude client), metrics, faithfulness checker, eCFR parser, e2e on sample
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/raw .pytest_cache .ruff_cache
