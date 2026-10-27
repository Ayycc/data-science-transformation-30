import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ridehail_fs.config import SimConfig  # noqa: E402
from ridehail_fs.data import simulate_city  # noqa: E402
from ridehail_fs.features import build_feature_tables, write_feature_tables  # noqa: E402


@pytest.fixture(scope="session")
def small_cfg() -> SimConfig:
    return replace(SimConfig(), n_days=3, n_drivers=120, requests_per_day=2500, train_days=2, warmup_days=1)


@pytest.fixture(scope="session")
def small_tables(small_cfg):
    return simulate_city(small_cfg)


@pytest.fixture(scope="session")
def small_feats(small_tables):
    return build_feature_tables(small_tables)


@pytest.fixture(scope="session")
def small_store(small_feats, tmp_path_factory):
    from ridehail_fs.feature_repo import make_store

    root = tmp_path_factory.mktemp("fs")
    write_feature_tables(small_feats, root / "offline")
    store, _ = make_store(root / "store", root / "offline")
    return store
