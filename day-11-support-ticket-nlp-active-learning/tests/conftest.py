"""Shared fixtures: small SYNTHETIC ticket sets (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ticket_intel.augment import augment_frame  # noqa: E402
from ticket_intel.pii import Redactor  # noqa: E402
from ticket_intel.synth import generate_base  # noqa: E402


@pytest.fixture(scope="session")
def tickets():
    return augment_frame(generate_base(40, seed=7), seed=8, pii_frac=0.5, multilingual_frac=0.1)


@pytest.fixture(scope="session")
def redactor():
    return Redactor()
