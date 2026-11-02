import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sreanomaly.generate import SimConfig, generate  # noqa: E402

SMALL = SimConfig(n_days=5, n_incidents=12, n_concurrent=2, n_surges=1)


@pytest.fixture(scope="session")
def small_tel():
    return generate(SMALL)
