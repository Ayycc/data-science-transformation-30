from __future__ import annotations

import pandas as pd
import pytest

from claims_mlops import config as C
from claims_mlops import data as D
from claims_mlops import models as M


@pytest.fixture(scope="session")
def small_book() -> pd.DataFrame:
    """Small SYNTHETIC book (3,000 policies x 24 months) for fast tests."""
    return D.generate_synthetic(C.SimConfig(policies_per_month=3_000))


@pytest.fixture(scope="session")
def sample() -> pd.DataFrame:
    """The committed sample (SYNTHETIC) used to check the offline path."""
    return pd.read_csv(C.DATA_SAMPLE / "policies_sample.csv")


@pytest.fixture(scope="session")
def glm(small_book) -> M.BalancedModel:
    return M.fit(M.build_glm(), D.split_months(small_book, range(1, 13)))
