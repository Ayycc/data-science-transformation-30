import numpy as np
import pandas as pd
import pytest

from gridcast.metrics import coverage, coverage_table, mase, mase_scale, pinball, smape, summarise, winkler


def test_mase_scale_is_seasonal_naive_mae():
    y = np.array([1.0, 2.0, 3.0, 2.0, 4.0, 5.0])
    assert mase_scale(y, m=3) == pytest.approx(np.mean([1.0, 2.0, 2.0]))
    assert mase(np.array([2.0, 4.0]), np.array([1.0, 4.0]), scale=0.5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        mase_scale(y, m=6)


def test_smape_bounds():
    assert smape(np.array([100.0]), np.array([100.0])) == 0.0
    assert smape(np.array([100.0]), np.array([0.0])) == pytest.approx(200.0)
    assert smape(np.array([100.0, 100.0]), np.array([110.0, 90.0])) == pytest.approx((20 / 210 + 20 / 190) / 2 * 100)


def test_pinball_asymmetry():
    y = np.array([10.0])
    assert pinball(y, np.array([8.0]), 0.9) == pytest.approx(1.8)  # under-forecast is penalised by q
    assert pinball(y, np.array([12.0]), 0.9) == pytest.approx(0.2)
    assert pinball(y, np.array([12.0]), 0.5) == pytest.approx(1.0)


def test_coverage_and_winkler():
    y, lo, hi = np.array([1.0, 5.0, 10.0]), np.array([0.0, 0.0, 0.0]), np.array([4.0, 6.0, 8.0])
    assert coverage(y, lo, hi) == pytest.approx(2 / 3)
    # widths 4,6,8 plus a miss of 2 above the upper bound scaled by 2/alpha = 20
    assert winkler(y, lo, hi, alpha=0.1) == pytest.approx((4 + 6 + 8 + 40) / 3)


def test_summarise_perfect_forecast():
    n = 48
    pred = pd.DataFrame(
        {
            "model": "m",
            "region": "A",
            "fold": 0,
            "ds": pd.date_range("2024-01-01", periods=n, freq="h"),
            "horizon": np.tile(np.arange(1, 25), 2),
            "y": np.full(n, 100.0),
        }  # fmt: skip
    )
    pred["q50"] = pred["y"]
    pred["q05"], pred["q95"] = pred["y"] - 5, pred["y"] + 5
    s = summarise(pred, pd.DataFrame({"region": ["A"], "fold": [0], "scale": [10.0]})).iloc[0]
    assert s["mase"] == 0 and s["smape_pct"] == 0 and s["coverage90"] == 1.0
    assert s["mean_width_pct_of_load"] == pytest.approx(10.0)
    assert s["scaled_pinball"] == pytest.approx((0.05 * 5 + 0 + 0.05 * 5) / 3 / 10)
    t = coverage_table(pred, ["region", "horizon"])
    assert len(t) == 24 and (t["coverage"] == 1.0).all() and (t["n"] == 2).all()
