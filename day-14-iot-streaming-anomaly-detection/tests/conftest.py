from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_stream.config import SimConfig  # noqa: E402
from iot_stream.data import load_sample  # noqa: E402
from iot_stream.simulate import simulate  # noqa: E402


@pytest.fixture(scope="session")
def tiny_stream():
    """SYNTHETIC: 10 machines x 2 days."""
    return simulate(SimConfig(n_machines=10, days=2.0, seed=5, weibull_scale_days=3.0, regime_change_day=1.0))


@pytest.fixture(scope="session")
def sample_stream():
    return load_sample()
