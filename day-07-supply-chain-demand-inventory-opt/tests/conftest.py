import copy

import pytest

from invopt.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def small_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["data"]["n_skus"] = 12
    c["data"]["m5_probe_timeout_s"] = 1
    c["sensitivity"]["lead_time_cv_multipliers"] = [1.0, 2.0]
    c["milp"]["time_limit_s"] = 5
    c["milp"]["crosscheck_time_limit_s"] = 10
    return c
