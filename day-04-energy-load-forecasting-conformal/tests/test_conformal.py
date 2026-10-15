import numpy as np
import pandas as pd
import pytest

from gridcast.conformal import PRED_COLS, CachedPrediction, conformalise


def _raw(n_days=80, sd=0.02, seed=0, shift_from=None):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2024-01-01", periods=n_days * 24, freq="h")
    noise = rng.normal(0, sd, len(ds))
    if shift_from is not None:
        noise = noise * np.where(ds >= pd.Timestamp(shift_from), 3.0, 1.0)  # errors triple
    raw = pd.DataFrame({"region": "A", "ds": ds, "horizon": ds.hour + 1, "level": 1000.0})
    raw["y_rel"] = 1.0 + noise
    raw["load_mw"] = raw["y_rel"] * raw["level"]
    raw["p_point"] = 1.0
    raw["p_med"] = 1.0
    raw["p_lo"] = 1.0 - 1.645 * sd
    raw["p_hi"] = 1.0 + 1.645 * sd
    return raw


def test_cached_prediction_returns_column():
    X = np.arange(12).reshape(3, 4)
    m = CachedPrediction(2).fit(None)
    assert hasattr(m, "fitted_")
    np.testing.assert_array_equal(m.predict(X), [2, 6, 10])
    assert len(PRED_COLS) == 4


@pytest.mark.parametrize("method", ["SplitCP", "CQR"])
def test_exchangeable_errors_give_nominal_coverage(method):
    raw = _raw()
    fs = pd.Timestamp("2024-02-26")
    p = conformalise(raw, fs, method, 28, "rolling")
    assert p["ds"].min() == fs and len(p) == (80 - 56) * 24
    cov = np.mean((p["y"] >= p["q05"]) & (p["y"] <= p["q95"]))
    assert 0.86 <= cov <= 0.94
    assert (p["q05"] <= p["q50"]).all() and (p["q50"] <= p["q95"]).all()


def test_static_vs_rolling_vs_aci_under_shift():
    """After the error scale triples, static intervals under-cover; rolling recovers; ACI recovers further."""
    raw = _raw(n_days=120, shift_from="2024-03-01")
    fs = pd.Timestamp("2024-02-26")
    cov = {}
    for name, kw in {"static": dict(mode="static"), "rolling": dict(mode="rolling"),
                     "aci": dict(mode="rolling", aci_gamma=0.05)}.items():  # fmt: skip
        p = conformalise(raw, fs, "SplitCP", 28, **kw)
        cov[name] = np.mean((p["y"] >= p["q05"]) & (p["y"] <= p["q95"]))
    assert cov["static"] < 0.6
    assert cov["rolling"] > cov["static"] + 0.2
    assert cov["aci"] >= cov["rolling"]


def test_aci_alpha_moves_when_missing():
    raw = _raw(n_days=90, shift_from="2024-03-01")
    p = conformalise(raw, pd.Timestamp("2024-03-01"), "SplitCP", 28, "rolling", aci_gamma=0.05)
    daily = p.groupby(p["ds"].dt.normalize())["alpha_used"].first()
    assert daily.iloc[5] < 0.10  # misses after the shift push alpha down, i.e. wider intervals


def test_bad_arguments():
    raw = _raw(n_days=60)
    with pytest.raises(ValueError):
        conformalise(raw, pd.Timestamp("2024-02-10"), "nope", 28)
    with pytest.raises(ValueError):
        conformalise(raw, pd.Timestamp("2024-02-10"), "SplitCP", 28, "static", aci_gamma=0.05)
