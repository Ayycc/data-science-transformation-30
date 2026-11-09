PY ?= python3
export PYTHONPATH := src

.PHONY: setup data run twin portfolio docs app test lint ci-snapshot clean

setup:  ## install pinned dependencies (Python 3.11+, CPU only, no PyTorch)
	$(PY) -m pip install -r requirements.txt

data:  ## regenerate the SYNTHETIC history (data/generated/, git-ignored) and the committed sample
	$(PY) -c "from pathlib import Path; from twin30 import world; r = world.simulate_world(); print(world.write_history(r, Path('data/generated'))); print(world.write_sample(r, Path('data/sample')), 'sample bytes')"

run: twin portfolio  ## everything: twin pipeline -> results/, then portfolio report + docs site

twin:  ## world -> PyMC demand -> calibration -> validation -> scenarios -> optimiser -> results/
	$(PY) scripts/run_all.py

portfolio:  ## scan all 30 day-* folders -> PORTFOLIO_REPORT.md, results/portfolio/, docs/, mkdocs.yml, site/
	$(PY) scripts/build_portfolio.py --build-site

docs:  ## serve the MkDocs site locally
	$(PY) -m mkdocs serve

app:  ## Streamlit what-if explorer (needs results/model from make run)
	$(PY) -m streamlit run app/streamlit_app.py

test:  ## pytest: world, twin, MILP, risk metrics, PyMC smoke fit, portfolio scan, offline docs build, app smoke
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts app
	$(PY) -m ruff format --check src tests scripts app

ci-snapshot:  ## refresh data/sample/ci_snapshot.json from GitHub Actions (needs gh + network; not part of run)
	$(PY) scripts/ci_snapshot.py

clean:
	rm -rf data/generated data/quick site .pytest_cache .ruff_cache
