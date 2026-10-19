PY ?= python3
export PYTHONPATH := src

.PHONY: setup java data run test lint clean

setup:  ## install pinned dependencies (Delta jars are fetched from Maven Central on first Spark start)
	$(PY) -m pip install -r requirements.txt

java:  ## Spark 4 needs Java 17+ (ubuntu-latest GitHub runners ship it)
	@java -version 2>&1 | head -1 || (echo "Java 17+ not found: install a JDK (e.g. apt install openjdk-17-jre-headless) or set JAVA_HOME" && exit 1)

data: java  ## SYNTHETIC landing zone, ~14M deliveries (data/generated, git-ignored) + committed sample
	$(PY) scripts/generate_data.py

run: data  ## regenerate data if needed, run all batches, benchmarks, results/ and figures (~15 min on 4 vCPU)
	$(PY) scripts/run_pipeline.py

test: java  ## chispa/pytest suite on tiny SYNTHETIC fixtures
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

clean:
	rm -rf data/generated spark-warehouse metastore_db derby.log .pytest_cache .ruff_cache
