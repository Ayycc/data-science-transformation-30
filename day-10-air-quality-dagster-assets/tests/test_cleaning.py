import numpy as np
import pandas as pd

from air_quality.cleaning import clean_readings, deduplicate, detect_drift
from air_quality.schemas import cleaned_schema, raw_schema, validate


def _stations(n=3):
    return pd.DataFrame(
        {
            "station_id": [f"ST{i:03d}" for i in range(n)],
            "region": "R00",
            "cal_offset_pm25": [5.0] + [0.0] * (n - 1),
            "cal_offset_no2": [0.0] * n,
        }
    )


def _raw(n_hours=24, n=3, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-09-01", periods=n_hours, freq="h")
    rows = []
    for i in range(n):
        rows.append(
            pd.DataFrame(
                {
                    "station_id": f"ST{i:03d}",
                    "ts": ts,
                    "pm25": np.round(10 + rng.normal(0, 2, n_hours), 1),
                    "no2": np.round(20 + rng.normal(0, 3, n_hours), 1),
                    "o3": np.round(30 + rng.normal(0, 3, n_hours), 1),
                    "landed_at": ts + pd.Timedelta(minutes=70),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_deduplicate_keeps_latest_delivery():
    raw = _raw()
    dup = raw.iloc[[0]].assign(pm25=99.0, landed_at=raw["landed_at"].iloc[0] + pd.Timedelta(minutes=30))
    out, n = deduplicate(pd.concat([raw, dup]))
    assert n == 1 and len(out) == len(raw)
    assert out.loc[(out.station_id == "ST000") & (out.ts == raw.ts.iloc[0]), "pm25"].item() == 99.0


def test_clean_flags_faults_and_applies_calibration():
    raw = _raw()
    raw.loc[raw.index[3], "no2"] = 25000.0  # unit bug -> out of range
    raw.loc[raw.index[5], "pm25"] = -1.0  # noise near zero -> clipped
    s1 = raw.index[raw.station_id == "ST001"]
    raw.loc[s1[4:10], "o3"] = 33.3  # 6 identical hours -> flatline
    raw.loc[s1[15], "pm25"] = 300.0  # spike
    clean, stats = clean_readings(raw, _stations())
    assert stats["no2_out_of_range"] == 1 and stats["out_of_range_values"] == 1
    assert stats["o3_flatline"] == 6
    assert stats["pm25_spike"] == 1
    c1 = clean[clean.station_id == "ST001"].reset_index(drop=True)
    assert c1.loc[4:9, "o3"].isna().all() and c1.loc[15, "pm25"] != c1.loc[15, "pm25"]
    # calibration: ST000 has a +5 offset in the registry
    merged = raw[raw.station_id == "ST000"].reset_index(drop=True)
    c0 = clean[clean.station_id == "ST000"].reset_index(drop=True)
    ok = c0["pm25"].notna() & (merged["pm25"] > 5)
    assert np.allclose(c0.loc[ok, "pm25"], merged.loc[ok, "pm25"] - 5.0)
    assert (clean[["pm25", "no2", "o3"]].min() >= 0).all()
    assert validate(cleaned_schema, clean)[0]


def test_raw_schema_rejects_bad_batches():
    raw = _raw()
    assert validate(raw_schema, raw)[0]
    bad = raw.copy()
    bad.loc[0, "station_id"] = "station-1"
    bad.loc[1, "landed_at"] = bad.loc[1, "ts"] - pd.Timedelta(hours=1)
    ok, n_fail, examples = validate(raw_schema, bad)
    assert not ok and n_fail >= 2 and examples
    assert not validate(raw_schema, raw.assign(extra=1))[0]  # strict: unexpected column


def test_drift_detection_finds_linear_bias_only():
    rng = np.random.default_rng(1)
    hours = pd.date_range("2026-09-01", periods=24 * 10, freq="h")
    regional = 15 + 8 * np.sin(np.arange(len(hours)) / 5) + np.where(np.arange(len(hours)) > 150, 40, 0)
    frames = []
    for i in range(12):
        val = regional * (1 + 0.05 * (i % 3)) + 2 * (i % 4) + rng.normal(0, 1.5, len(hours))
        if i == 0:
            val = val + 1.2 * np.arange(len(hours)) / 24  # drifting sensor
        frames.append(pd.DataFrame({"station_id": f"ST{i:03d}", "region": "R00", "ts": hours, "pm25": val}))
    st = pd.DataFrame({"station_id": [f"ST{i:03d}" for i in range(12)]})
    res = detect_drift(pd.concat(frames), st, pd.Timestamp("2026-09-10").date()).set_index("station_id")
    assert res.loc["ST000", "drift_suspect"]
    assert abs(res.loc["ST000", "drift_rate"] - 1.2) < 0.2
    assert res["drift_suspect"].sum() == 1
