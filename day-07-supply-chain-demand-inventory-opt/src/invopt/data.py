"""SYNTHETIC data generator for a 3-warehouse, multi-SKU distributor.

The M5 (Walmart) data is probed first; if it cannot be reached the seeded generator below is used.
Every random draw comes from one ``numpy.random.Generator`` seeded from the config.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PATTERNS = ("smooth", "erratic", "intermittent", "lumpy", "seasonal")


@dataclass
class Dataset:
    """SKU master, item (SKU x warehouse) master and long weekly demand table."""

    skus: pd.DataFrame
    items: pd.DataFrame
    demand: pd.DataFrame  # columns: item_id, week, demand
    source: str

    def history(self, n_weeks: int) -> pd.DataFrame:
        return self.demand[self.demand["week"] < n_weeks]

    def test(self, n_weeks: int) -> pd.DataFrame:
        return self.demand[self.demand["week"] >= n_weeks]


def probe_m5(url: str, timeout_s: float) -> tuple[bool, str]:
    """Return (reachable, message). Only a HEAD request; nothing is downloaded."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - fixed public URL from config
            return resp.status == 200, f"HTTP {resp.status}"
    except Exception as exc:  # network blocked, proxy 403, DNS, timeout...
        return False, f"{type(exc).__name__}: {exc}"[:200]


def _season_mask(n_weeks: int, start: int, length: int) -> np.ndarray:
    woy = np.arange(n_weeks) % 52
    return (woy >= start) & (woy < start + length)


def generate(cfg: dict, n_skus: int | None = None, seed: int | None = None) -> Dataset:
    """Generate SYNTHETIC demand, costs, MOQs and lead-time distributions."""
    dcfg = cfg["data"]
    n_skus = int(n_skus or dcfg["n_skus"])
    seed = int(cfg["seed"] if seed is None else seed)
    rng = np.random.default_rng(seed)
    n_weeks = int(dcfg["history_weeks"] + dcfg["test_weeks"])
    whs = cfg["warehouses"]
    wh_share = np.array([w["demand_share"] for w in whs], dtype=float)
    wh_share = wh_share / wh_share.sum()

    mix = dcfg["pattern_mix"]
    probs = np.array([mix[p] for p in PATTERNS], dtype=float)
    patterns = rng.choice(PATTERNS, size=n_skus, p=probs / probs.sum())

    unit_cost = np.round(np.exp(rng.normal(np.log(18.0), 0.9, n_skus)), 2).clip(0.8, 600)
    markup = rng.uniform(1.3, 2.1, n_skus)
    price = np.round(unit_cost * markup, 2)
    volume = np.round(np.exp(rng.normal(np.log(0.004), 0.8, n_skus)), 5)  # m3 per unit
    pack = rng.choice([1, 2, 6, 12, 24], size=n_skus, p=[0.25, 0.15, 0.3, 0.2, 0.1])
    lt_mean = np.round(rng.uniform(1.0, 5.0, n_skus), 2)  # weeks
    lt_cv = np.round(rng.uniform(0.10, 0.50, n_skus), 3)

    # Base weekly demand across all warehouses (units); heavy tailed so ABC emerges. Cheaper items sell more.
    base = np.exp(rng.normal(np.log(6.0), 1.2, n_skus)) * (18.0 / unit_cost) ** 0.35
    trend = rng.normal(0.0, 0.0015, n_skus)  # per-week multiplicative drift

    weeks = np.arange(n_weeks)
    season = _season_mask(n_weeks, int(dcfg["season_start_week"]), int(dcfg["season_length_weeks"]))
    rows_item, rows_demand = [], []
    sku_ids = [f"SKU{i:04d}" for i in range(n_skus)]
    item_rows = []
    for i in range(n_skus):
        share = rng.dirichlet(wh_share * 40.0)
        pat = patterns[i]
        level = base[i] * np.exp(trend[i] * weeks)
        for w, wh in enumerate(whs):
            lam = level * share[w]
            if pat == "smooth":
                lam = lam * 3.0
                mult = rng.gamma(1 / 0.15**2, 0.15**2, n_weeks)
                d = rng.poisson(lam * mult)
            elif pat == "erratic":
                lam = lam * 2.0
                k = 1.2  # negative binomial dispersion -> high CV of sizes
                d = rng.negative_binomial(k, k / (k + lam))
            elif pat == "intermittent":
                p = rng.uniform(0.15, 0.5)
                size = 1 + rng.poisson(np.maximum(lam * 0.6, 0.2))
                d = (rng.random(n_weeks) < p) * size
            elif pat == "lumpy":
                p = rng.uniform(0.06, 0.3)
                size = np.ceil(rng.lognormal(np.log(np.maximum(lam * 2.0, 1.0)), 0.9))
                d = (rng.random(n_weeks) < p) * size
            else:  # seasonal: demand only inside the season window, season total varies year to year
                yearly = rng.lognormal(0.0, 0.2, n_weeks // 52 + 1)[weeks // 52]
                d = np.where(season, rng.poisson(lam * 4.0 * yearly), 0)
            item_id = f"{sku_ids[i]}|{wh['name']}"
            item_rows.append((item_id, sku_ids[i], wh["name"]))
            rows_item.append(np.full(n_weeks, len(item_rows) - 1))
            rows_demand.append(d.astype(np.int64))

    # MOQ: supplier minimum in packs; for slow movers it can exceed many weeks of demand (a cause of overstock).
    moq_packs = rng.choice([1, 2, 4, 8], size=n_skus, p=[0.4, 0.3, 0.2, 0.1])
    moq = pack * moq_packs

    skus = pd.DataFrame(
        {
            "sku_id": sku_ids,
            "pattern": patterns,
            "unit_cost": unit_cost,
            "price": price,
            "volume_m3": volume,
            "pack_size": pack,
            "moq": moq,
            "lt_mean_weeks": lt_mean,
            "lt_cv": lt_cv,
        }
    )
    items = pd.DataFrame(item_rows, columns=["item_id", "sku_id", "warehouse"])
    idx = np.concatenate(rows_item)
    demand = pd.DataFrame(
        {
            "item_id": items["item_id"].to_numpy()[idx],
            "week": np.tile(weeks, len(item_rows)),
            "demand": np.concatenate(rows_demand),
        }
    )
    return Dataset(skus=skus, items=items, demand=demand, source="synthetic")


def lead_time_samples(skus: pd.DataFrame, items: pd.DataFrame, n_weeks: int, seed: int, cv_multiplier: float = 1.0) -> np.ndarray:
    """Gamma lead times (weeks) per item x order week, shared by every policy (common random numbers)."""
    rng = np.random.default_rng(seed + 991)
    m = items.merge(skus, on="sku_id", how="left")
    mean = m["lt_mean_weeks"].to_numpy()[:, None]
    cv = np.maximum(m["lt_cv"].to_numpy()[:, None] * cv_multiplier, 1e-3)
    shape = 1.0 / cv**2
    z = rng.standard_gamma(np.broadcast_to(shape, (len(m), n_weeks)))
    return np.maximum(z * mean / shape, 0.25)


def save(ds: Dataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.skus.to_csv(out_dir / "skus.csv", index=False)
    ds.items.to_csv(out_dir / "items.csv", index=False)
    ds.demand.to_csv(out_dir / "demand.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": ds.source,
                "label": "SYNTHETIC",
                "n_skus": len(ds.skus),
                "n_items": len(ds.items),
                "n_weeks": int(ds.demand["week"].max() + 1),
                "rows": len(ds.demand),
            },
            indent=2,
        )
    )


def load(in_dir: Path) -> Dataset:
    man = json.loads((in_dir / "manifest.json").read_text())
    return Dataset(
        skus=pd.read_csv(in_dir / "skus.csv"),
        items=pd.read_csv(in_dir / "items.csv"),
        demand=pd.read_csv(in_dir / "demand.csv.gz"),
        source=man["source"],
    )
