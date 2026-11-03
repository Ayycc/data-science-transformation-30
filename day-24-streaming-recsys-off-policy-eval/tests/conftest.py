import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recsys24.data import WorldConfig, generate_world  # noqa: E402

SMALL_WORLD = WorldConfig(n_users=300, n_items=200, mean_events=25, max_events=60)


@pytest.fixture(scope="session")
def small_ds():
    return generate_world(SMALL_WORLD)
