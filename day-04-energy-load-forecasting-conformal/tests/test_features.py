import numpy as np
import pandas as pd

from gridcast.features import FEATURES, build_features


def test_no_load_leakage_beyond_origin(sample_df):
    """Features at target t must not change if load after the forecast origin (t - horizon) changes."""
    df = sample_df[sample_df["region"] == "NORTH"].reset_index(drop=True)
    base = build_features(df)
    target_day = pd.Timestamp("2024-05-15")
    origin = target_day - pd.Timedelta(hours=1)  # 23:00 the day before
    future = df["ds"] > origin
    perturbed = df.copy()
    perturbed.loc[future, "load_mw"] *= 1.5
    perturbed.loc[future, "temp_c"] += 10  # realised temperature after the origin is unknown too
    pert = build_features(perturbed)
    rows = (base["ds"] >= target_day) & (base["ds"] < target_day + pd.Timedelta(days=1))
    assert rows.sum() == 24
    pd.testing.assert_frame_equal(base.loc[rows, FEATURES].reset_index(drop=True),
                                  pert.loc[rows, FEATURES].reset_index(drop=True))  # fmt: skip
    # ...but the target itself does change, so the test is not vacuous.
    assert not np.allclose(base.loc[rows, "y_rel"], pert.loc[rows, "y_rel"])


def test_horizon_and_level(sample_df):
    f = build_features(sample_df)
    assert f["horizon"].between(1, 24).all()
    assert (f.loc[f["ds"].dt.hour == 0, "horizon"] == 1).all()
    ok = f.dropna(subset=["level"])
    assert (ok["level"] > 0).all()
    # The first rows lack 7 days + 24 h of history and are left as NaN for callers to drop.
    first = f[f["region"] == "NORTH"].sort_values("ds").head(191)
    assert first["level"].isna().all()
