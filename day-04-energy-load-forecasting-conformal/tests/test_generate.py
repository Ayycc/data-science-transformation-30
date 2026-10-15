import hashlib

import numpy as np
import pandas as pd

import gridcast.generate as gen
from gridcast.config import FULL, REGIME_START, SAMPLE
from gridcast.data import data_path, read_load, write_csv_gz
from gridcast.generate import generate, holiday_dates


def test_sample_is_reproducible_byte_for_byte(tmp_path):
    out = tmp_path / "load.csv.gz"
    write_csv_gz(generate(SAMPLE), out)
    committed = data_path(SAMPLE)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == hashlib.sha256(committed.read_bytes()).hexdigest()
    assert committed.stat().st_size < 1_000_000


def test_sample_shape(sample_df):
    assert set(sample_df["region"]) == set(SAMPLE.regions)
    n_hours = int((pd.Timestamp(SAMPLE.end) - pd.Timestamp(SAMPLE.start)) / pd.Timedelta(hours=1))
    assert len(sample_df) == n_hours * len(SAMPLE.regions)
    assert sample_df["load_mw"].gt(0).all()
    assert sample_df.groupby("region")["ds"].apply(lambda s: s.diff().dropna().eq(pd.Timedelta(hours=1)).all()).all()


def test_structure_holidays_weekly_and_regime_shift(sample_df, monkeypatch):
    hol = holiday_dates("2024-01-01", "2024-12-31")
    assert pd.Timestamp("2024-07-04") in hol and pd.Timestamp("2024-12-25") in hol
    # Counterfactual world without the data-centre regime shift: identical before April, ~12% of base after.
    monkeypatch.setattr(gen, "REGIME_UPLIFT", 0.0)
    no_shift = generate(SAMPLE)
    diff = (sample_df["load_mw"] - no_shift["load_mw"]).where(sample_df["region"] == "CENTRAL")
    before = diff[sample_df["ds"] < REGIME_START]
    after = diff[sample_df["ds"] >= "2024-04-15"]
    assert before.abs().max() == 0
    assert 0.10 * 4000 < after.mean() < 0.14 * 4000
    n = sample_df[sample_df["region"] == "NORTH"].set_index("ds")["load_mw"]
    weekday = n[n.index.dayofweek < 5].mean()
    weekend = n[n.index.dayofweek >= 5].mean()
    assert weekend < weekday


def test_temperature_shift_only_changes_window():
    a = generate(SAMPLE)
    b = generate(SAMPLE, temp_shift=("2024-05-01", "2024-05-08", 4.0))
    win = (a["ds"] >= "2024-05-01") & (a["ds"] < "2024-05-08")
    np.testing.assert_allclose((b["temp_c"] - a["temp_c"])[win], 4.0, atol=0.011)
    pd.testing.assert_frame_equal(a[a["ds"] < "2024-05-01"], b[b["ds"] < "2024-05-01"])
    assert pd.Timestamp(REGIME_START) < pd.Timestamp(FULL.end)


def test_read_roundtrip(sample_df):
    assert sample_df["ds"].dtype.kind == "M"
    assert read_load(data_path(SAMPLE)).equals(sample_df)
