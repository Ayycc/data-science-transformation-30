from __future__ import annotations

import numpy as np
import pandas as pd

from claims_mlops import config as C
from claims_mlops import data as D


def test_generator_is_seeded_and_shaped(small_book):
    again = D.generate_synthetic(C.SimConfig(policies_per_month=3_000))
    pd.testing.assert_frame_equal(small_book, again)
    assert len(small_book) == 24 * 3_000
    assert set(C.FEATURES) <= set(small_book.columns)
    assert small_book["PolicyID"].is_unique


def test_tweedie_target_has_point_mass_at_zero_and_heavy_tail(small_book):
    pp = small_book[C.TARGET]
    assert 0.80 < (pp == 0).mean() < 0.97
    assert (small_book.loc[small_book.ClaimNb == 0, "ClaimAmount"] == 0).all()
    assert (small_book.loc[small_book.ClaimNb > 0, "ClaimAmount"] > 0).all()
    pos = pp[pp > 0]
    assert pos.quantile(0.99) > 5 * pos.median()


def test_injected_covariate_shift(small_book):
    pre = small_book[small_book.month < 19]
    post = small_book[small_book.month >= 19]
    assert (post.DrivAge < 27).mean() > 2.5 * (pre.DrivAge < 27).mean()
    assert np.log(post.Density).mean() > np.log(pre.Density).mean() + 0.5
    # untouched covariates stay put
    assert abs(post.VehPower.mean() - pre.VehPower.mean()) < 0.15


def test_fallback_to_synthetic_when_public_unreachable(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("403 Forbidden")

    monkeypatch.setattr(D, "_fetch_arff", boom)
    acq = D.acquire(cfg=C.SimConfig(n_months=2, policies_per_month=100))
    assert acq.provenance["path"] == "synthetic" and acq.provenance["synthetic"]
    assert "403" in acq.provenance["error"]
    assert len(acq.policies) == 200


def test_public_path_builds_monthly_batches():
    """freMTPL2-shaped frames (fake, SYNTHETIC values) go through the same monthly + shift logic."""
    rng = np.random.default_rng(0)
    n = 6_000
    freq = pd.DataFrame(
        {
            "IDpol": np.arange(n),
            "ClaimNb": rng.poisson(0.1, n),
            "Exposure": rng.uniform(0.1, 1, n),
            "Area": rng.choice([f"'{a}'" for a in C.AREAS], n),
            "VehPower": rng.integers(4, 15, n),
            "VehAge": rng.integers(0, 20, n),
            "DrivAge": rng.integers(18, 90, n),
            "BonusMalus": rng.integers(50, 150, n),
            "VehBrand": rng.choice(C.BRANDS, n),
            "VehGas": rng.choice(C.GAS, n),
            "Density": rng.integers(1, 20_000, n),
            "Region": rng.choice(C.REGIONS, n),
        }
    )
    ids = freq.loc[freq.ClaimNb > 0, "IDpol"]
    sev = pd.DataFrame({"IDpol": ids, "ClaimAmount": rng.gamma(2, 800, len(ids))})
    out = D.monthly_from_public(freq, sev, C.SimConfig(n_months=6, policies_per_month=500, shift_start_month=4))
    assert out.month.value_counts().eq(500).all()
    assert set(out.Area) <= set(C.AREAS)
    assert (out.loc[out.ClaimNb == 0, "ClaimAmount"] == 0).all()
    assert (out[out.month >= 4].DrivAge < 27).mean() > (out[out.month < 4].DrivAge < 27).mean()


def test_committed_sample_is_small_and_valid(sample):
    path = C.DATA_SAMPLE / "policies_sample.csv"
    assert path.stat().st_size <= 1_000_000
    assert set(C.FEATURES + [C.TARGET, C.WEIGHT, "month"]) <= set(sample.columns)
    assert sample.month.nunique() == 24
    assert sample[C.WEIGHT].between(0, 1).all()
