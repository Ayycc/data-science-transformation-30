import os

import numpy as np
import pandas as pd

from ridehail_fs.data import simulate_city, tlc_to_tables, write_sample
from ridehail_fs.model import metrics


def test_simulation_is_seeded(small_cfg, small_tables):
    again = simulate_city(small_cfg)
    pd.testing.assert_frame_equal(small_tables["trips"], again["trips"])


def test_trip_schema_and_ranges(small_tables):
    t = small_tables["trips"]
    assert {"trip_id", "request_ts", "dropoff_ts", "zone_id", "dropoff_zone_id", "driver_id", "distance_km", "duration_min"} <= set(t.columns)
    assert (t.dropoff_ts > t.request_ts).all()
    assert t.duration_min.between(1, 240).all()
    assert t.request_ts.is_monotonic_increasing


def test_sample_is_small(small_tables, tmp_path):
    write_sample(small_tables, tmp_path)
    total = sum(os.path.getsize(tmp_path / f) for f in os.listdir(tmp_path))
    assert 0 < total < 1_000_000


def test_tlc_mapping_on_tlc_shaped_frame():
    ts = pd.Timestamp("2024-01-05 08:00")
    tlc = pd.DataFrame(
        {
            "tpep_pickup_datetime": [ts, ts + pd.Timedelta(minutes=3), ts],
            "tpep_dropoff_datetime": [ts + pd.Timedelta(minutes=12), ts + pd.Timedelta(minutes=30), ts - pd.Timedelta(minutes=1)],
            "PULocationID": [161, 237, 1],
            "DOLocationID": [236, 48, 1],
            "trip_distance": [2.0, 5.0, 1.0],
        }
    )
    out = tlc_to_tables(tlc, seed=1)
    assert len(out["trips"]) == 2  # negative duration dropped
    np.testing.assert_allclose(out["trips"].distance_km, [2 * 1.609344, 5 * 1.609344])
    assert out["offers"].accepted.sum() == 2 and len(out["probes"]) == 2


def test_metrics():
    m = metrics(np.array([10.0, 20.0]), np.array([12.0, 20.0]))
    assert m["mae_min"] == 1.0 and m["within_2min_pct"] == 100.0


def test_committed_sample_runs_through_feature_build():
    from pathlib import Path

    from ridehail_fs.features import build_feature_tables

    d = Path(__file__).resolve().parents[1] / "data" / "sample"
    tabs = {}
    for name, col in (("trips", ["request_ts", "dropoff_ts"]), ("offers", ["offer_ts"]), ("probes", ["probe_ts"])):
        tabs[name] = pd.read_csv(d / f"{name}_sample.csv.gz", parse_dates=col)
        for c in col:
            tabs[name][c] = pd.to_datetime(tabs[name][c], utc=True)
    feats = build_feature_tables(tabs)
    assert len(tabs["trips"]) > 100
    assert feats["zone_traffic"]["avg_speed_15m"].notna().sum() > 100
