"""ABC (value), XYZ (variability) and Syntetos-Boylan-Croston (intermittency) segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def abc_classes(revenue: pd.Series, cutoffs: tuple[float, float] = (0.8, 0.95)) -> pd.Series:
    """A: top items covering ``cutoffs[0]`` of revenue, B: next up to ``cutoffs[1]``, C: the rest."""
    order = revenue.sort_values(ascending=False)
    total = order.sum()
    cum = order.cumsum() / total if total > 0 else pd.Series(1.0, index=order.index)
    prev = cum.shift(fill_value=0.0)  # share before this item -> the item that crosses a cut-off stays in the class
    cls = np.where(prev < cutoffs[0], "A", np.where(prev < cutoffs[1], "B", "C"))
    return pd.Series(cls, index=order.index).reindex(revenue.index)


def xyz_classes(cv: pd.Series, cutoffs: tuple[float, float] = (0.5, 1.0)) -> pd.Series:
    return pd.Series(np.where(cv < cutoffs[0], "X", np.where(cv < cutoffs[1], "Y", "Z")), index=cv.index)


def sbc_class(series: np.ndarray, adi_cut: float = 1.32, cv2_cut: float = 0.49) -> str:
    """Syntetos-Boylan-Croston demand class from average inter-demand interval and CV^2 of non-zero sizes."""
    nz = series[series > 0]
    if len(nz) == 0:
        return "intermittent"
    adi = len(series) / len(nz)
    cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else 0.0
    if adi < adi_cut:
        return "smooth" if cv2 < cv2_cut else "erratic"
    return "intermittent" if cv2 < cv2_cut else "lumpy"


def segment(
    history: pd.DataFrame,
    items: pd.DataFrame,
    skus: pd.DataFrame,
    cfg: dict,
    season_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """One row per item with ABC (computed per SKU across warehouses), XYZ, SBC and seasonal flag."""
    scfg = cfg["segmentation"]
    wide = history.pivot(index="item_id", columns="week", values="demand").sort_index(axis=1)
    m = items.set_index("item_id").join(skus.set_index("sku_id"), on="sku_id")
    m = m.loc[wide.index]
    mean = wide.mean(axis=1)
    std = wide.std(axis=1, ddof=1)
    cv = (std / mean.replace(0, np.nan)).fillna(np.inf)

    sku_rev = (mean * m["price"]).groupby(m["sku_id"]).sum() * 52
    abc = abc_classes(sku_rev, tuple(scfg["abc_cutoffs"]))
    out = pd.DataFrame(index=wide.index)
    out["sku_id"] = m["sku_id"]
    out["warehouse"] = m["warehouse"]
    out["mean_weekly"] = mean
    out["cv"] = cv
    out["annual_revenue_sku"] = m["sku_id"].map(sku_rev)
    out["abc"] = m["sku_id"].map(abc)
    out["xyz"] = xyz_classes(cv, tuple(scfg["xyz_cutoffs"]))
    out["sbc"] = [sbc_class(row, scfg["sbc_adi"], scfg["sbc_cv2"]) for row in wide.to_numpy()]
    if season_mask is not None:
        arr = wide.to_numpy()
        tot = arr.sum(axis=1)
        in_season = arr[:, season_mask[: arr.shape[1]]].sum(axis=1)
        mask = season_mask[: arr.shape[1]]
        years = np.arange(arr.shape[1]) // 52
        each_year = np.all([(arr[:, mask & (years == y)].sum(axis=1) > 0) for y in np.unique(years)], axis=0)
        # Seasonal: >= 90% of volume inside the season window in every history year and >= 5 non-zero weeks
        # (detected from data, not read from the generator).
        out["seasonal"] = (in_season >= 0.9 * tot) & each_year & ((arr > 0).sum(axis=1) >= 5)
    else:
        out["seasonal"] = False
    out["segment"] = out["abc"] + out["xyz"]
    return out
