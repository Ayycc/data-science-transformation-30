import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pricing27.estimation import demand_models, estimate_elasticities, generate_logs  # noqa: E402
from pricing27.simulator import MarketConfig  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return MarketConfig()


@pytest.fixture(scope="session")
def logs(cfg):
    return generate_logs(cfg, seed=20261106)


@pytest.fixture(scope="session")
def est(logs):
    return estimate_elasticities(logs)


@pytest.fixture(scope="session")
def models(est, logs):
    m = demand_models(est)
    m["logs"] = logs
    return m
