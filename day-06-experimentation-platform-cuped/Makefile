PY ?= python3
export PYTHONPATH := src

.PHONY: setup data run mc-small test coverage app lint clean

setup:  ## install pinned dependencies
	$(PY) -m pip install -r requirements.txt

data:  ## regenerate the committed SYNTHETIC sample in data/sample (8k users of E01)
	$(PY) scripts/generate_data.py

run:  ## one command reproduces every number and figure in README.md / results/ (~3 min on 4 vCPU)
	$(PY) scripts/run_pipeline.py --scale full
	$(PY) -m pytest -q --cov=expengine --cov-report=term --cov-report=json:results/coverage.json

mc-small:  ## fast smoke run of the whole pipeline at 5% scale (writes to data/small_results, not results/)
	$(PY) scripts/run_pipeline.py --scale small --results data/small_results --data data/small

test:  ## pytest with the >= 85% line-coverage gate on the library
	$(PY) -m pytest -q --cov=expengine --cov-report=term-missing --cov-fail-under=85

app:  ## optional Streamlit app over the committed sample (or your own CSV)
	$(PY) -m streamlit run app/streamlit_app.py

lint:
	$(PY) -m ruff check src tests scripts app
	$(PY) -m ruff format --check src tests scripts app

clean:
	rm -rf data/generated data/small data/small_results .pytest_cache .coverage
