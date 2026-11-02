import numpy as np

from sreanomaly import detect


def test_seasonal_z_centered_in_training(small_tel):
    tr = 2 * 1440
    z = detect.seasonal_zscores(small_tel.values, tr)
    assert abs(np.median(z[:tr])) < 0.2
    assert 0.6 < np.median(np.abs(z[:tr])) < 1.0  # ~0.674 for a standard normal


def test_persistence_and_episodes():
    raw = np.zeros((20, 1), bool)
    raw[3:4] = True  # 1-minute blip
    raw[10:15] = True
    p = detect.persistence(raw, 3)
    assert not p[3, 0] and p[12, 0] and not p[11, 0]
    ep = detect.episodes(p, p.astype(float), ["svc"], gap=5)
    assert len(ep) == 1 and ep.iloc[0]["start"] == 12


def test_ecod_flags_injected_shift():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, (2000, 2, 4))
    z[1500:1530, 1, :2] += 8
    feats = detect.features(z)
    out = detect.multivariate(feats, 1000, "ecod", 0, quantile=0.999, k=3)
    assert out.flags[1510:1530, 1].mean() > 0.8
    assert out.flags[1000:1500, 0].mean() < 0.01


def test_raw_metric_alerts_counts_each_metric():
    z = np.zeros((100, 1, 2))
    z[10:20, 0, :] = 9
    raw = detect.raw_metric_alerts(z, ["a"], ["m1", "m2"])
    assert len(raw) == 2 and set(raw["metric"]) == {"m1", "m2"}
