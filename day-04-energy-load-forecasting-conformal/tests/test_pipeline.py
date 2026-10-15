import json

import pandas as pd

from gridcast.backtest import SELECTED, fold_starts
from gridcast.config import FULL, SAMPLE
from gridcast.report import acceptance, write_results


def test_fold_layout():
    s = fold_starts(FULL)
    assert len(s) == 12 and s[0] == pd.Timestamp("2024-01-31")
    assert s[-1] + pd.Timedelta(days=28) == pd.Timestamp(FULL.end)
    assert pd.Timestamp(FULL.val_start) + pd.Timedelta(days=(FULL.n_val_folds - 1) * FULL.val_spacing_days + 28) < s[0]


def test_sample_backtest_end_to_end(sample_backtest, sample_df, tmp_path):
    res = sample_backtest
    pred = res["pred"]
    models = set(pred["model"])
    for m in ["SeasonalNaive-24h", "SeasonalNaive-168h", "AutoETS", "AutoARIMA", "MSTL", SELECTED]:
        assert m in models
    # every model forecasts the same points: 2 folds x 7 days x 24 h x 2 regions
    counts = pred.groupby("model").size()
    assert (counts == SAMPLE.n_folds * SAMPLE.fold_days * 24 * len(SAMPLE.regions)).all()
    assert pred["horizon"].between(1, 24).all()
    assert pred.groupby(["model", "region", "ds"]).size().max() == 1
    assert (pred["q05"] <= pred["q95"]).all()
    s = res["summary"].set_index("model")
    assert s["mase"].gt(0).all() and s["coverage90"].between(0, 1).all()
    acc = acceptance(res)
    assert 0 <= acc["coverage_overall"] <= 1 and acc["best_model"] in models

    m = write_results(res, sample_df, {"path_used": "synthetic_generator", "public_probe": {"skipped": True}},
                      tmp_path, {"total_s": 0.0})  # fmt: skip
    on_disk = json.loads((tmp_path / "metrics.json").read_text())
    assert on_disk["acceptance"] == m["acceptance"]
    for f in ["report.md", "mase_by_model.png", "coverage_region_horizon.png", "forecast_examples.html"]:
        assert (tmp_path / f).exists()


def test_backtest_is_deterministic(sample_backtest, sample_df):
    from gridcast.backtest import run_backtest

    again = run_backtest(sample_df, SAMPLE, log=lambda m: None, stress=False)
    pd.testing.assert_frame_equal(sample_backtest["summary"], again["summary"])
