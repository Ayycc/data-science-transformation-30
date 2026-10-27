"""Point-in-time correctness: Feast must return the latest row at or before the request, within TTL."""

import numpy as np
import pandas as pd
import pytest

from ridehail_fs.feature_repo import make_store
from ridehail_fs.features import hourly_snapshot_tables, write_feature_tables
from ridehail_fs.joins import ALL_FEATURES, compare_frames, feast_pit_join, naive_join, reference_asof

T0 = pd.Timestamp("2026-09-01 08:00", tz="UTC")


def _zone_rows(stamps_min, values):
    ts = [T0 + pd.Timedelta(minutes=m) for m in stamps_min]
    return pd.DataFrame(
        {
            "zone_id": 1,
            "event_timestamp": ts,
            "avg_speed_15m": values,
            "avg_speed_60m": values,
            "probe_count_15m": 5.0,
            "speed_trend": 1.0,
            "created": ts,
        }
    )


@pytest.fixture()
def tiny_store(tmp_path):
    zt = _zone_rows([0, 5, 10, 60], [10.0, 20.0, 30.0, 99.0])  # gap 10 -> 60 exceeds the 30-min TTL
    zd = pd.DataFrame({"zone_id": [1], "event_timestamp": [T0], "requests_15m": [1.0], "requests_60m": [4.0], "demand_index": [1.0], "created": [T0]})
    ds = pd.DataFrame(
        {
            "driver_id": [7, 7],
            "event_timestamp": [T0 - pd.Timedelta(hours=1), T0 + pd.Timedelta(hours=1)],
            "acceptance_rate_7d": [0.5, 0.9],
            "trips_24h": [3.0, 4.0],
            "slowness_ratio_7d": [1.5, 2.0],
            "created": [T0 - pd.Timedelta(hours=1), T0 + pd.Timedelta(hours=1)],
        }
    )
    feats = {"zone_traffic": zt, "zone_demand": zd, "driver_stats": ds}
    write_feature_tables(feats, tmp_path / "offline")
    store, _ = make_store(tmp_path / "store", tmp_path / "offline")
    return store, feats


def _trips(minutes):
    ts = [T0 + pd.Timedelta(minutes=m) for m in minutes]
    return pd.DataFrame(
        {
            "trip_id": np.arange(len(ts)),
            "request_ts": pd.to_datetime(ts, utc=True),
            "zone_id": 1,
            "dropoff_zone_id": 1,
            "driver_id": 7,
            "distance_km": 5.0,
        }
    )


def test_feast_pit_returns_latest_past_row_never_future(tiny_store):
    store, feats = tiny_store
    trips = _trips([-1, 0, 4.99, 5, 7, 39, 41, 61])
    out = feast_pit_join(store, trips).set_index("trip_id")["avg_speed_15m"]
    # -1: nothing yet; 0 and 4.99: row@0 (row@5 is in the future); 5, 7: row@5 ... 39: row@10 within TTL;
    # 41: row@10 is 31 min old -> expired; 61: row@60
    expected = [np.nan, 10.0, 10.0, 20.0, 20.0, 30.0, np.nan, 99.0]
    np.testing.assert_allclose(out.to_numpy(dtype=float), expected, equal_nan=True)


def test_driver_feature_is_as_of_request_not_latest(tiny_store):
    store, _ = tiny_store
    out = feast_pit_join(store, _trips([30])).iloc[0]
    assert out["acceptance_rate_7d"] == pytest.approx(0.5)  # the 09:00 row (0.9) is in the future


def test_feast_matches_independent_asof_reference(tiny_store):
    store, feats = tiny_store
    trips = _trips([-1, 0, 3, 5, 12, 39, 41, 61, 65])
    a, b = feast_pit_join(store, trips), reference_asof(trips, feats)
    assert compare_frames(a, b, ALL_FEATURES)["mismatched_cells"] == 0


def test_feast_matches_reference_on_simulated_city(small_store, small_tables, small_feats):
    trips = small_tables["trips"].sample(3000, random_state=0)
    rep = compare_frames(feast_pit_join(small_store, trips), reference_asof(trips, small_feats), ALL_FEATURES)
    assert rep["rows"] == 3000 and rep["mismatched_cells"] == 0


def test_naive_join_reads_the_future(small_tables, small_feats):
    trips = small_tables["trips"].iloc[5000:6000]
    snaps = hourly_snapshot_tables(small_tables, small_feats)
    naive = naive_join(trips, snaps)
    ref = reference_asof(trips, small_feats)
    # the naive join hands each trip the zone state at the END of its clock hour -> differs from the PIT value
    rep = compare_frames(naive, ref, ["avg_speed_15m", "slowness_ratio_7d"], rtol=1e-6)
    assert rep["per_column"]["avg_speed_15m"] > 0.9 * len(trips)
    assert rep["per_column"]["slowness_ratio_7d"] > 0.9 * len(trips)
