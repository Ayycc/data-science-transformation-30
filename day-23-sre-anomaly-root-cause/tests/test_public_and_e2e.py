import json
from pathlib import Path

import numpy as np
import pandas as pd

from sreanomaly.generate import SimConfig, generate
from sreanomaly.pipeline import run
from sreanomaly.public import evaluate_series, parse_series, window_mask

ROOT = Path(__file__).resolve().parents[1]


def test_nab_parser_and_evaluation_on_fixture():
    ts = pd.date_range("2024-01-01", periods=1000, freq="5min")
    v = np.random.default_rng(0).normal(50, 1, 1000)
    v[700:720] += 15
    raw = pd.DataFrame({"timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"), "value": v}).to_csv(index=False).encode()
    df = parse_series(raw)
    assert len(df) == 1000 and df["timestamp"].is_monotonic_increasing
    win = [[str(ts[690]), str(ts[730])]]
    assert window_mask(df["timestamp"], win).sum() == 41
    out = evaluate_series(df, win, seed=0)
    assert out["ecod"]["windows_hit"] == 1 and out["iforest"]["windows_hit"] == 1


def test_committed_sample_matches_generator():
    sample = ROOT / "data" / "sample"
    files = list(sample.glob("*"))
    assert files and sum(f.stat().st_size for f in files) <= 1_000_000
    tel = generate(SimConfig())
    assert (sample / "incidents_SYNTHETIC.csv").read_text() == tel.incidents.to_csv(index=False)
    assert (sample / "deploys_SYNTHETIC.csv").read_text() == tel.deploys.to_csv(index=False)
    wide = pd.read_csv(sample / "telemetry_6h_SYNTHETIC.csv")
    assert wide.shape == (360, 1 + 20 * 4)
    a = int(wide["minute"].iloc[0])
    np.testing.assert_allclose(wide.iloc[:, 1:].to_numpy(), tel.values[a : a + 360].reshape(360, -1), atol=1e-4)


def test_small_end_to_end(tmp_path):
    cfg = SimConfig(n_days=5, n_incidents=12, n_concurrent=2, n_surges=1)
    m = run(cfg, results_dir=tmp_path, data_dir=None, public=False, figures=True)
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["data"]["path_used"] == "synthetic_generator"
    assert saved["alert_volume"]["reduction_raw_to_incidents_pct"] > 50
    assert saved["root_cause"]["methods"]["ensemble"]["top3_detected"] >= 0.6
    assert (tmp_path / "figures" / "rca_topk.png").exists() and (tmp_path / "incident_timelines.md").exists()
    assert len(m["_rows"]) >= 8
