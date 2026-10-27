"""Training/serving skew and online latency on a small simulated city."""

import pandas as pd

from ridehail_fs.joins import entity_frame, feast_pit_join
from ridehail_fs.serving import checkpoint_times, latency_benchmark, push_demo, replay_online, skew_report


def test_online_features_equal_training_features(small_store, small_tables):
    trips = small_tables["trips"]
    start = trips.request_ts.min().normalize() + pd.Timedelta(days=1, hours=6)
    cps = checkpoint_times(start, start + pd.Timedelta(hours=12), 6, seed=1)
    online, rep = replay_online(small_store, trips, cps)
    assert rep["checkpoints"] >= 4 and len(online) > 20
    offline = feast_pit_join(small_store, trips[trips.trip_id.isin(online.trip_id)])
    sk = skew_report(online, offline)
    assert sk["mismatched_cells"] == 0, sk["per_column"]


def test_online_p99_under_20ms(small_store, small_tables):
    trips = small_tables["trips"]
    end = trips.request_ts.max().floor("5min")
    small_store.materialize(start_date=(end - pd.Timedelta(hours=4)).to_pydatetime(), end_date=end.to_pydatetime())
    lat = latency_benchmark(small_store, entity_frame(trips.tail(500)), n=300, warmup=30)
    assert lat["single_row"]["p99_ms"] < 20.0


def test_push_is_visible_online(small_store, small_tables):
    end = small_tables["trips"].request_ts.max().ceil("h") + pd.Timedelta(hours=1)
    res = push_demo(small_store, 0, end, 12.25)
    assert res["visible_online"]
