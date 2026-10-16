"""Stretch: resolution time by neighbourhood income (SYNTHETIC incomes), via a GeoPandas spatial join."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

from . import geo


def neighbourhoods_gdf() -> gpd.GeoDataFrame:
    inc = geo.neighbourhood_income()
    return gpd.GeoDataFrame(
        {"neighbourhood": np.arange(len(inc)), "median_income": inc},
        geometry=geo.neighbourhood_polygons(),
        crs="EPSG:4326",
    )


def assign_neighbourhood(res: pd.DataFrame) -> pd.Series:
    pts = gpd.GeoDataFrame(index=res.index, geometry=gpd.points_from_xy(res.longitude, res.latitude), crs="EPSG:4326")
    j = gpd.sjoin(pts, neighbourhoods_gdf()[["neighbourhood", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    return j.neighbourhood.reindex(res.index)


def equity_analysis(res: pd.DataFrame, hot_cells: set[str], seed: int = 7) -> tuple[pd.DataFrame, dict]:
    """Neighbourhood mean category-adjusted delay vs synthetic income; Q1 (lowest) vs Q4 gap with bootstrap CI."""
    hoods = neighbourhoods_gdf()[["neighbourhood", "median_income"]]
    hoods["income_quartile"] = pd.qcut(hoods.median_income, 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    out: dict = {}
    tables = []
    for label, sub in (("all_cells", res), ("excluding_hotspot_cells", res[~res.h3.isin(hot_cells)])):
        g = sub.groupby("neighbourhood").agg(n=("log_excess", "size"), mean_log_excess=("log_excess", "mean"))
        g = g.join(hoods.set_index("neighbourhood"))
        g["excess_pct"] = 100 * (np.exp(g.mean_log_excess - sub.log_excess.mean()) - 1)
        rho, p = stats.spearmanr(g.median_income, g.mean_log_excess)
        q = g.groupby("income_quartile").mean_log_excess.mean()
        gap = 100 * (np.exp(q["Q1"] - q["Q4"]) - 1)
        rng = np.random.default_rng(seed)
        boots = []
        q1, q4 = g[g.income_quartile == "Q1"], g[g.income_quartile == "Q4"]
        for _ in range(2000):
            a = q1.mean_log_excess.to_numpy()[rng.integers(0, len(q1), len(q1))].mean()
            b = q4.mean_log_excess.to_numpy()[rng.integers(0, len(q4), len(q4))].mean()
            boots.append(100 * (np.exp(a - b) - 1))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[label] = {
            "spearman_rho_income_vs_delay": round(float(rho), 3), "spearman_p": float(f"{p:.3g}"),
            "q1_vs_q4_gap_pct": round(float(gap), 1), "q1_vs_q4_gap_ci95": [round(float(lo), 1), round(float(hi), 1)],
            "requests": int(len(sub)),
        }  # fmt: skip
        tables.append(g.reset_index().assign(scope=label))
    return pd.concat(tables, ignore_index=True), out
