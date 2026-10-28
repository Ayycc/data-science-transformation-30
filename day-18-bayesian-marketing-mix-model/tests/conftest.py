import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from bayes_mmm.data import simulate  # noqa: E402


@pytest.fixture(scope="session")
def panel():
    return simulate()


@pytest.fixture(scope="session")
def small_panel():
    return simulate(n_weeks=52, n_regions=3, seed=7)
