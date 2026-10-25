from __future__ import annotations

import numpy as np
import pandas as pd

from bikeshare import band, geo
from bikeshare import features as F


def _offline_features(st, monkeypatch):
    monkeypatch.setattr(geo, "elevation_from_tiles", lambda lat, lon, z=12: (None, "offline test"))
    return geo.add_spatial_features(st, try_osm=False)


def test_haversine_known_distance():
    # one degree of latitude is ~111.2 km
    d = geo.haversine_matrix(np.array([38.0]), np.array([-77.0]), np.array([39.0]), np.array([-77.0]))
    assert abs(d[0, 0] - 111_195) < 200


def test_spatial_features_offline(toy_stations, monkeypatch):
    st, prov = _offline_features(toy_stations, monkeypatch)
    assert st["h3"].str.len().eq(15).all()
    assert st.loc[4, "metro_dist_m"] == 0  # the Metro dock itself
    assert (st["metro_dist_m"] >= 0).all() and st["metro_dist_m"].max() < 3000
    assert "committed sample" in prov["elevation"]  # falls back to provided elevations
    assert (st["stations_ring2"] >= st["stations_ring1"]).all() and (st["stations_ring1"] >= st["stations_in_cell"]).all()
    nb = geo.neighbour_lists(st)
    assert all(len(n) >= 3 and i not in n for i, n in enumerate(nb))
    cells = geo.h3_cells_gdf(st)
    assert cells["n_stations"].sum() == len(st)


def test_panel_lags_are_day_ahead(toy_stations, monkeypatch):
    st, _ = _offline_features(toy_stations, monkeypatch)
    rng = np.random.default_rng(1)
    start = pd.Timestamp("2024-08-01")
    n = 4000
    t0 = start + pd.to_timedelta(rng.uniform(0, 21 * 86400, n), unit="s")
    trips = pd.DataFrame(
        {
            "start_time": t0,
            "end_time": t0 + pd.Timedelta(minutes=15),
            "start_id": rng.choice(st["station_id"], n),
            "end_id": rng.choice(st["station_id"], n),
        }
    )
    dep, arr, hours = F.hourly_counts(trips, st["station_id"], start, start + pd.Timedelta(days=21))
    assert dep.sum() == n and n - 10 <= arr.sum() <= n  # a few arrivals fall after the window
    weather = pd.DataFrame({"temp_c": 20.0, "precip_mm": 0.0}, index=hours)
    panel = F.build_panel(dep, arr, hours, st, weather, geo.neighbour_lists(st))
    net = arr - dep
    s, t = 3, 200
    row = panel[(panel["station"] == s) & (panel["t"] == t)].iloc[0]
    assert row["lag24"] == net[s, t - 24] and row["lag168"] == net[s, t - 168]
    # expanding same-hour-of-week mean uses only earlier weeks
    assert np.isclose(row["ha_past_how"], net[s, [t - 168]].mean())
    assert np.isnan(panel.loc[panel["t"] < 168, "ha_past_how"]).all()
    assert set(F.FEATURES) <= set(panel.columns)


def test_band_table_drain_fill(toy_stations, monkeypatch):
    st, _ = _offline_features(toy_stations, monkeypatch)
    hours = pd.date_range("2024-08-01", periods=48, freq="h")
    net = np.zeros((len(st), 48), dtype=np.float32)
    net[0, 1:4] = -2  # drains 6 bikes overnight, then refills 4
    net[0, 6:8] = 2
    weather = pd.DataFrame({"temp_c": 20.0, "precip_mm": 0.0}, index=hours)
    tab = band.band_table(net, hours, weather, st, geo.neighbour_lists(st), shifts=(("overnight", 0, 300, 14), ("midday", 10, 240, 24)))
    r = tab[(tab["station"] == 0) & (tab["shift"] == "overnight")].iloc[0]
    assert r["drain"] == 6 and r["fill"] == 0
    assert tab[(tab["station"] == 1)][["drain", "fill"]].to_numpy().sum() == 0
