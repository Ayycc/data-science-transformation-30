import json
from pathlib import Path

import numpy as np

from invopt import data as dm

ROOT = Path(__file__).resolve().parents[1]


def test_all_costs_parameterised(cfg):
    for k in ("holding_rate_annual", "ordering_cost_per_line", "shortage_goodwill_frac_price", "seasonal_salvage_frac_cost"):
        assert cfg["costs"][k] > 0
    assert set(cfg["service_targets"]) == {a + x for a in "ABC" for x in "XYZ"}
    assert all(0.5 < v < 1 for v in cfg["service_targets"].values())


def test_generator_is_deterministic_and_well_formed(cfg):
    a = dm.generate(cfg, n_skus=15)
    b = dm.generate(cfg, n_skus=15)
    assert a.demand.equals(b.demand) and a.skus.equals(b.skus)
    n_weeks = cfg["data"]["history_weeks"] + cfg["data"]["test_weeks"]
    assert len(a.items) == 15 * len(cfg["warehouses"])
    assert len(a.demand) == len(a.items) * n_weeks
    assert (a.demand["demand"] >= 0).all()
    assert (a.skus["moq"] % a.skus["pack_size"] == 0).all()
    assert (a.skus["price"] > a.skus["unit_cost"]).all()


def test_lead_time_samples_match_mean(cfg):
    ds = dm.generate(cfg, n_skus=30)
    lt = dm.lead_time_samples(ds.skus, ds.items, 400, cfg["seed"])
    mean = ds.items.merge(ds.skus, on="sku_id")["lt_mean_weeks"].to_numpy()
    assert (lt > 0).all()
    assert np.allclose(lt.mean(axis=1), mean, rtol=0.12)
    lt2 = dm.lead_time_samples(ds.skus, ds.items, 400, cfg["seed"], cv_multiplier=2.0)
    assert (lt2.std(axis=1) > lt.std(axis=1)).mean() > 0.9


def test_committed_sample_is_small_and_reproducible(cfg, tmp_path):
    sample = ROOT / "data" / "sample"
    files = list(sample.glob("*"))
    assert files and sum(f.stat().st_size for f in files) <= 1_000_000
    man = json.loads((sample / "manifest.json").read_text())
    assert man["label"] == "SYNTHETIC"
    s = dm.load(sample)
    full = dm.generate(cfg)
    keep = set(s.skus["sku_id"])
    sub = full.demand[full.demand["item_id"].str.split("|").str[0].isin(keep)].reset_index(drop=True)
    assert np.array_equal(sub["demand"].to_numpy(), s.demand["demand"].to_numpy())
