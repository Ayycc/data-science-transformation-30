import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legacy28.generator import as_of_close, generate  # noqa: E402
from legacy28.io import extracts_to_arrow  # noqa: E402
from legacy28.legacy import run_legacy  # noqa: E402


@pytest.fixture(scope="session")
def extracts():
    return generate(1_500, seed=7)


@pytest.fixture(scope="session")
def final(extracts):
    return as_of_close(extracts, 8)


@pytest.fixture(scope="session")
def sources(final):
    return extracts_to_arrow(final)


@pytest.fixture(scope="session")
def legacy_out(final):
    return run_legacy(final, 8)
