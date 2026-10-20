import pandas as pd

from air_quality.config import TINY
from air_quality.generate import simulate
from air_quality.sample import sample_frames

from .conftest import ROOT


def test_generator_is_deterministic():
    a, b = simulate(TINY), simulate(TINY)
    for p in ("pm25", "no2", "o3"):
        assert (a["meas"][p] == b["meas"][p]).all()
    assert (a["missing"] == b["missing"]).all()


def test_tiny_dataset_has_planted_faults(tiny_source):
    late = sorted(p.name for p in (tiny_source / "late_arrivals").iterdir())
    assert late == ["2026-09-06", "2026-09-12", "2026-09-20", "2026-09-27"]
    st = pd.read_parquet(tiny_source / "truth" / "stations_truth.parquet")
    assert (st["drift_start_day"] >= 0).sum() == 1 and st["unit_bug"].sum() > 0
    day24 = pd.read_parquet(tiny_source / "landing" / "2026-09-24" / "batch_000.parquet")
    assert (day24["no2"] > 2000).mean() > 0.1  # the firmware unit bug
    day12 = pd.read_parquet(tiny_source / "landing" / "2026-09-12" / "batch_000.parquet")
    assert day12[["station_id", "ts"]].drop_duplicates().shape[0] < 0.85 * 20 * 24  # backhaul outage


def test_committed_sample_matches_generator():
    stations, readings = sample_frames()
    committed_st = pd.read_csv(ROOT / "data" / "sample" / "stations.csv")
    committed = pd.read_csv(ROOT / "data" / "sample" / "readings_2026-09-01.csv", parse_dates=["ts", "landed_at"])
    assert (ROOT / "data" / "sample" / "readings_2026-09-01.csv").stat().st_size < 1_000_000
    pd.testing.assert_frame_equal(stations.reset_index(drop=True), committed_st, check_dtype=False)
    assert len(committed) == len(readings)
    pd.testing.assert_series_equal(committed["pm25"], readings["pm25"].round(1), check_dtype=False, check_names=False)
    assert (committed["station_id"] == readings["station_id"]).all()
