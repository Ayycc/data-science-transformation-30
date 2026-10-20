import numpy as np
import pandas as pd

from air_quality.aqi import category, daily_station_aqi, sub_index


def test_pm25_breakpoints_2024():
    vals = sub_index("pm25", np.array([0.0, 9.0, 9.1, 35.4, 35.5, 55.4, 125.4, 225.4, 400.0, np.nan]))
    assert vals[:9].tolist() == [0, 50, 51, 100, 101, 150, 200, 300, 500]
    assert np.isnan(vals[9])


def test_pm25_truncation_and_interpolation():
    # 35.49 truncates to 35.4 -> 100 ; 12.0 -> 51 + (49/26.3)*(2.9) = 56.4 -> 56
    assert sub_index("pm25", np.array([35.49]))[0] == 100
    assert sub_index("pm25", np.array([12.0]))[0] == 56


def test_no2_and_o3():
    assert sub_index("no2", np.array([53, 100, 101, 360.9])).tolist() == [50, 100, 101, 150]
    o3 = sub_index("o3", np.array([0.054, 0.0709, 0.071, 0.085, 0.25]))
    assert o3.tolist() == [50, 100, 101, 150, 300]


def test_category_edges():
    assert list(category(np.array([50, 51, 100, 101, 151, 201, 301]))) == [
        "Good",
        "Moderate",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]


def test_daily_station_aqi_coverage_rules():
    ts = pd.date_range("2026-09-01", periods=24, freq="h")
    a = pd.DataFrame({"station_id": "ST000", "ts": ts, "pm25": 40.0, "no2": 20.0, "o3": 30.0})
    b = a.assign(station_id="ST001")
    b.loc[b.index[:7], "pm25"] = np.nan  # only 17 valid PM2.5 hours -> no PM2.5 sub-index
    out = daily_station_aqi(pd.concat([a, b])).set_index("station_id")
    assert out.loc["ST000", "aqi_pm25"] == sub_index("pm25", np.array([40.0]))[0]
    assert out.loc["ST000", "dominant_pollutant"] == "pm25"
    assert np.isnan(out.loc["ST001", "aqi_pm25"])
    assert out.loc["ST001", "aqi"] == max(out.loc["ST001", "aqi_no2"], out.loc["ST001", "aqi_o3"])
