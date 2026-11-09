import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twin30 import calibrate, world  # noqa: E402
from twin30.demand import Posterior  # noqa: E402


@pytest.fixture(scope="session")
def small_run():
    """A short SYNTHETIC world (16 weeks) used across tests."""
    return world.simulate_world(seed=3, n_days=112, snapshot_days=(56, 112))


@pytest.fixture(scope="session")
def small_tables(small_run):
    return world.history_tables(small_run)


@pytest.fixture(scope="session")
def cal(small_tables):
    return calibrate.calibrate(small_tables, 0, 112)


@pytest.fixture(scope="session")
def oracle_posterior(small_run):
    """A 'posterior' that puts all mass on the true parameters (no MCMC needed)."""
    t = small_run.truth
    from twin30.demand import design

    # match log_season(doy) = sum a cos(w k (doy - p)) with Fourier cos/sin coefficients
    w = 2 * np.pi / 365.25
    four = np.zeros((3, 4))
    for c, (a1, p1, a2, p2) in enumerate(t.season):
        four[c] = [a1 * np.cos(w * p1), a1 * np.sin(w * p1), a2 * np.cos(2 * w * p2), a2 * np.sin(2 * w * p2)]
    a = np.log(np.array(t.base))[None, :] + np.log(np.array(t.store_size))[:, None]
    wd = np.log(np.array(t.weekday))
    _ = design
    n = 4
    return Posterior(
        a=np.repeat(a[None], n, axis=0),
        weekday=np.repeat(wd[None], n, axis=0),
        fourier=np.repeat(four[None], n, axis=0),
        trend=np.full((n, 3), t.trend_per_year),
        promo=np.repeat(np.log(t.promo_lift)[None], n, axis=0),
        post=np.repeat(np.log(t.post_promo_dip)[None], n, axis=0),
        alpha=np.full((n, 3), 100.0),
    )


@pytest.fixture(scope="session")
def snapshot(small_run):
    return json.loads(json.dumps(small_run.snapshots[56]))
