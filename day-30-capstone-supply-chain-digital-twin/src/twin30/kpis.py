"""One KPI definition shared by the historical ledger and the twin, so comparisons are like for like."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    CATEGORIES,
    DAYS_PER_YEAR,
    DCS,
    DISCOUNT,
    FINANCE,
    FREIGHT,
    N_CATS,
    N_STORES,
    PRICE,
    STORES,
    SUPPLIER_COST,
    SUPPLIERS,
    UNIT_COST,
    geometry,
)

# KPIs that are observable in the historical record and used for twin validation
VALIDATION_KPIS = (
    "units_sold",
    "revenue",
    "gross_margin",
    "in_stock_rate",
    "avg_store_inventory_units",
    "avg_dc_inventory_units",
    "outbound_transport_cost",
    "operating_profit",
)


@dataclass
class Ledger:
    sales: np.ndarray  # (T, S, C)
    stockout: np.ndarray  # (T, S, C) 0/1
    store_oh: np.ndarray  # (T, S, C)
    dc_oh: np.ndarray  # (T, D, C)
    ship: np.ndarray  # (T, S, C) units shipped to each store
    assign: np.ndarray  # (S,) DC serving each store
    promo: np.ndarray  # (T, C)
    receipts: np.ndarray  # (K, C) units received from each supplier in the period
    open_dcs: np.ndarray  # (D,) bool
    demand: np.ndarray | None = None  # (T, S, C) only known inside a simulation


def compute(led: Ledger) -> dict[str, float]:
    T = led.sales.shape[0]
    years = T / DAYS_PER_YEAR
    geo = geometry()
    price = PRICE[None, :] * (1 - DISCOUNT[None, :] * led.promo)
    units_c = led.sales.sum(axis=(0, 1))
    revenue = float((led.sales * price[:, None, :]).sum())
    rec = led.receipts.sum(axis=0)
    blend = np.where(rec > 0, (led.receipts * SUPPLIER_COST[:, None]).sum(axis=0) / np.maximum(rec, 1e-9), 1.0)
    cogs = float((units_c * UNIT_COST * blend).sum())
    store_val = float((led.store_oh * UNIT_COST[None, None, :]).sum(axis=(1, 2)).mean())
    dc_val = float((led.dc_oh * UNIT_COST[None, None, :]).sum(axis=(1, 2)).mean())
    holding = (store_val + dc_val) * FINANCE.holding_rate_per_year * years
    dist = geo.dist[led.assign, np.arange(N_STORES)]
    outbound = float((led.ship.sum(axis=0) * dist[:, None] * FREIGHT[None, :]).sum())
    handling = float(led.ship.sum() * FINANCE.dc_handling_per_unit)
    fixed = float(np.sum(np.array(FINANCE.dc_fixed_cost_per_year) * led.open_dcs) * years)
    gm = revenue - cogs
    out = {
        "units_sold": float(units_c.sum()),
        "revenue": revenue,
        "gross_margin": gm,
        "in_stock_rate": float(1 - led.stockout.mean()),
        "avg_store_inventory_units": float(led.store_oh.sum(axis=(1, 2)).mean()),
        "avg_dc_inventory_units": float(led.dc_oh.sum(axis=(1, 2)).mean()),
        "holding_cost": holding,
        "outbound_transport_cost": outbound,
        "dc_handling_cost": handling,
        "dc_fixed_cost": fixed,
        "operating_profit": gm - holding - outbound - handling - fixed,
    }
    if led.demand is not None:
        out["fill_rate"] = float(led.sales.sum() / max(led.demand.sum(), 1.0))
        out["lost_sales_units"] = float(led.demand.sum() - led.sales.sum())
    return out


def ledger_from_tables(tabs: dict[str, pd.DataFrame], day_from: int, day_to: int) -> Ledger:
    """Build the ledger for days [day_from, day_to) from the observable history tables only."""
    cats = {c.name: i for i, c in enumerate(CATEGORIES)}
    stores = {s.name: i for i, s in enumerate(STORES)}
    dcs = {d.name: i for i, d in enumerate(DCS)}
    sups = {s.name: i for i, s in enumerate(SUPPLIERS)}
    T = day_to - day_from
    s = tabs["sales"]
    s = s[(s.day >= day_from) & (s.day < day_to)]
    t, si, ci = (s.day - day_from).to_numpy(), s.store.map(stores).to_numpy(), s.category.map(cats).to_numpy()

    def cube(col: str) -> np.ndarray:
        a = np.zeros((T, N_STORES, N_CATS))
        a[t, si, ci] = s[col].to_numpy()
        return a

    promo = np.zeros((T, N_CATS))
    promo[t, ci] = s.promo.to_numpy()
    sh = tabs["shipments"]
    sh = sh[(sh.ship_day >= day_from) & (sh.ship_day < day_to)]
    ship = np.zeros((T, N_STORES, N_CATS))
    np.add.at(
        ship, ((sh.ship_day - day_from).to_numpy(), sh.store.map(stores).to_numpy(), sh.category.map(cats).to_numpy()), sh.units.to_numpy()
    )
    assign = np.zeros(N_STORES, dtype=int)
    for st, dc in sh.groupby("store").dc.agg(lambda x: x.mode().iloc[0]).items():
        assign[stores[st]] = dcs[dc]
    dci = tabs["dc_inventory"]
    dci = dci[(dci.day >= day_from) & (dci.day < day_to)]
    dc_oh = np.zeros((T, len(DCS), N_CATS))
    dc_oh[(dci.day - day_from).to_numpy(), dci.dc.map(dcs).to_numpy(), dci.category.map(cats).to_numpy()] = dci.on_hand_end.to_numpy()
    po = tabs["purchase_orders"]
    po = po[(po.receipt_day >= day_from) & (po.receipt_day < day_to)]
    receipts = np.zeros((len(SUPPLIERS), N_CATS))
    np.add.at(receipts, (po.supplier.map(sups).to_numpy(), po.category.map(cats).to_numpy()), po.units.to_numpy())
    open_dcs = np.zeros(len(DCS), dtype=bool)
    open_dcs[np.unique(assign)] = True
    return Ledger(
        sales=cube("units_sold"),
        stockout=cube("stockout"),
        store_oh=cube("on_hand_end"),
        dc_oh=dc_oh,
        ship=ship,
        assign=assign,
        promo=promo,
        receipts=receipts,
        open_dcs=open_dcs,
    )
