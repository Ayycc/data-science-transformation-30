from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from bikeshare import config as C
from bikeshare import data, synthetic
from bikeshare import simulate as SIM


def test_sample_files_small_and_valid(sample_stations):
    total = sum(p.stat().st_size for p in C.DATA_SAMPLE.iterdir())
    assert total <= 1_000_000
    assert len(sample_stations) > 500
    assert sample_stations["lat"].between(38.6, 39.3).all() and sample_stations["lon"].between(-77.6, -76.7).all()
    assert sample_stations["is_metro"].sum() > 20
    w = pd.read_csv(C.DATA_SAMPLE / "weather_hourly.csv", parse_dates=["time"])
    assert len(w) == 24 * 60 and w["temp_c"].between(5, 45).all()


def test_parse_cabi_zip_roundtrip():
    csv = (
        '"ride_id","rideable_type","started_at","ended_at","start_station_name","start_station_id","end_station_name","end_station_id","start_lat","start_lng","end_lat","end_lng","member_casual"\n'
        '"X1","classic_bike","2024-09-03 14:02:18.985","2024-09-03 14:06:30.572","A St","31097","B St","31042",38.81,-77.04,38.80,-77.04,"member"\n'
        '"X2","electric_bike","2024-09-03 15:00:00","2024-09-03 15:10:00",,,,,38.81,-77.04,38.80,-77.04,"casual"\n'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("202409-capitalbikeshare-tripdata.csv", csv)
    t = data.parse_cabi_zip(buf.getvalue())
    assert list(t["ride_id"]) == ["X1"]  # dockless e-bike trip without stations dropped
    assert t.loc[0, "start_id"] == "31097" and t.loc[0, "end_time"] > t.loc[0, "start_time"]


def test_isd_parser_converts_utc_and_units():
    csv = 'DATE,REPORT_TYPE,TMP,AA1\n2024-09-16T12:52:00,FM-15,"+0200,5","01,0025,9,5"\n2024-09-16T13:00:00,FM-12,"+0999,1",\n'
    w = data.parse_isd(csv.encode())
    assert list(w.index) == [pd.Timestamp("2024-09-16 09:00")]  # 12:52 UTC -> 08:52 EDT -> hour ending 09:00
    assert w.iloc[0]["temp_c"] == 20.0 and w.iloc[0]["precip_mm"] == 2.5


def test_synthetic_generator_has_commute_structure(sample_stations):
    weather = data.synthetic_weather(pd.Timestamp("2024-09-16"), pd.Timestamp("2024-09-18"))
    trips = synthetic.generate_trips(sample_stations, weather, pd.Timestamp("2024-09-16"), pd.Timestamp("2024-09-18"), seed=7)
    again = synthetic.generate_trips(sample_stations, weather, pd.Timestamp("2024-09-16"), pd.Timestamp("2024-09-18"), seed=7)
    assert trips.equals(again)  # seeded
    assert 10_000 < len(trips) < 60_000
    core = (
        synthetic.haversine_matrix(
            sample_stations["lat"].to_numpy(), sample_stations["lon"].to_numpy(), np.array([38.9007]), np.array([-77.0365])
        )[:, 0]
        < 2000
    )
    core_ids = set(sample_stations.loc[core, "station_id"])
    h = trips["start_time"].dt.hour
    am_in = trips[(h >= 7) & (h <= 9)]["end_id"].isin(core_ids).mean()
    pm_in = trips[(h >= 16) & (h <= 18)]["end_id"].isin(core_ids).mean()
    assert am_in > pm_in  # downtown fills in the morning, empties in the evening


def test_committed_real_sample_day_simulates(sample_stations):
    trips = pd.read_csv(
        C.DATA_SAMPLE / "trips_2024-09-17.csv.gz", parse_dates=["start_time", "end_time"], dtype={"start_id": str, "end_id": str}
    )
    assert len(trips) > 5_000
    st = sample_stations.assign(capacity=data.estimate_capacity(sample_stations["activity_per_day"]))
    cfg = C.FleetConfig(shifts=())
    r = SIM.simulate(st, trips, SIM.Policy(), pd.Timestamp("2024-09-17"), pd.Timestamp("2024-09-18"), cfg)
    s = r.summary()
    assert s["lost_departures"] + s["diverted_arrivals"] > 0 and s["stockout_hours"] > 0
