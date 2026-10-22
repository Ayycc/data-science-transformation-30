import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from policy_causal.simulate import simulate_individuals, simulate_panel  # noqa: E402


@pytest.fixture(scope="session")
def people():
    return simulate_individuals(4000, 7)


@pytest.fixture(scope="session")
def panel():
    return simulate_panel(36, 24, {8: 9, 12: 9, 16: 9}, 7)
