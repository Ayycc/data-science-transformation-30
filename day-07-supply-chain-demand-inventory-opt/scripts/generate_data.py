"""Regenerate the SYNTHETIC dataset (data/generated) and the committed 40-SKU sample (data/sample)."""

from __future__ import annotations

from pathlib import Path

from invopt import data as data_mod
from invopt.config import load_config

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    cfg = load_config()
    ok, msg = data_mod.probe_m5(cfg["data"]["m5_probe_url"], cfg["data"]["m5_probe_timeout_s"])
    print(f"M5 probe: {'reachable' if ok else 'unreachable'} ({msg}); using the seeded SYNTHETIC generator")
    ds = data_mod.generate(cfg)
    ds.source = "synthetic (M5 probe: " + ("reachable, adapter not implemented" if ok else "unreachable") + ")"
    data_mod.save(ds, ROOT / "data" / "generated")
    keep = ds.skus["sku_id"].iloc[:40]
    smp = data_mod.Dataset(
        skus=ds.skus[ds.skus["sku_id"].isin(keep)],
        items=ds.items[ds.items["sku_id"].isin(keep)],
        demand=ds.demand[ds.demand["item_id"].str.split("|").str[0].isin(keep)],
        source=ds.source,
    )
    data_mod.save(smp, ROOT / "data" / "sample")
    print(f"{len(ds.items)} items x {ds.demand['week'].nunique()} weeks written")
