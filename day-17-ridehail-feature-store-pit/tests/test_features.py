"""Feature windows only use events strictly before the row's event_timestamp."""

import pandas as pd

from ridehail_fs.features import build_feature_tables

T0 = pd.Timestamp("2026-09-01 00:00", tz="UTC")


def _tables():
    m = lambda x: T0 + pd.Timedelta(minutes=x)  # noqa: E731
    trips = pd.DataFrame(
        {
            "trip_id": [0, 1],
            "request_ts": [m(1), m(50)],
            "dropoff_ts": [m(21), m(70)],
            "zone_id": [0, 0],
            "dropoff_zone_id": [0, 0],
            "driver_id": [3, 3],
            "distance_km": [5.0, 5.0],
            "duration_min": [20.0, 20.0],
            "free_flow_min": [10.0, 10.0],
        }
    )
    offers = pd.DataFrame({"driver_id": [3, 3, 3], "offer_ts": [m(1), m(49), m(50)], "accepted": [1, 0, 1]})
    probes = pd.DataFrame({"zone_id": [0, 0, 0], "probe_ts": [m(2), m(9), m(14.99)], "speed_kmh": [10.0, 20.0, 60.0]})
    return {"trips": trips, "offers": offers, "probes": probes}


def test_zone_traffic_window_excludes_current_and_future_probes():
    zt = build_feature_tables(_tables())["zone_traffic"].set_index("event_timestamp")
    at = lambda x: zt.loc[T0 + pd.Timedelta(minutes=x)]  # noqa: E731
    assert at(5)["avg_speed_15m"] == 10.0  # only the 00:02 probe
    assert at(10)["avg_speed_15m"] == 15.0  # 00:02 and 00:09
    assert at(15)["avg_speed_15m"] == 30.0  # + 00:14.99
    assert at(20)["avg_speed_15m"] == 40.0  # 00:02 has left the 15-min window


def test_driver_stats_count_trips_only_after_dropoff():
    ds = build_feature_tables(_tables())["driver_stats"]
    ds = ds[ds.driver_id == 3].set_index("event_timestamp")
    h1 = ds.loc[T0 + pd.Timedelta(hours=1)]
    assert h1["trips_24h"] == 1  # trip 1 is still in progress at 01:00
    assert h1["acceptance_rate_7d"] == 2 / 3
    assert ds.loc[T0 + pd.Timedelta(hours=2)]["trips_24h"] == 2
