from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retention.pipeline import load_config, make_data  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config(ROOT / "config" / "config.yaml")


@pytest.fixture(scope="session")
def small(cfg):
    """20k-customer SYNTHETIC panel + campaign truth (same generator, smaller n)."""
    return make_data(cfg, n_customers=20_000)
