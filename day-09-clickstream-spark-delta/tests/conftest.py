from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clickstream.spark import get_spark  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    s = get_spark(app="clickstream-tests", shuffle_partitions=4, driver_memory="2g", ui=False)
    yield s
    s.stop()
