import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from privacy25.data import PopulationConfig, generate_population, split  # noqa: E402


@pytest.fixture(scope="session")
def pop():
    return generate_population(PopulationConfig(n=4000, seed=7))


@pytest.fixture(scope="session")
def parts(pop):
    return split(pop, seed=1)
