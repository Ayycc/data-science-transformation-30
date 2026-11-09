"""SYNTHETIC "real world" for the capstone: hidden truth plus the historical record it leaves.

This is a plain NumPy day loop, deliberately independent of the SimPy twin in ``twin.py``. It has
features the twin does not model (AR(1) demand shocks, a Poisson-lognormal demand process, gamma
lead times, random one-day truck delays), so the twin's ability to reproduce historical KPIs is a
real out-of-sample test rather than a tautology. The twin only sees the tables written by
``history_tables`` (sales, shipments, purchase orders, DC stock, promo calendar, outage log).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import policy as pol
from . import promo as promo_mod
from .config import (
    BASELINE_POLICY,
    CATEGORIES,
    DAYS_PER_YEAR,
    DCS,
    DISCOUNT,
    FINANCE,
    HISTORICAL_ASSIGNMENT,
    HISTORY_DAYS,
    HISTORY_START,
    N_CATS,
    N_DCS,
    N_STORES,
    PRICE,
    SEED,
    STORES,
    SUPPLIERS,
    Policy,
    geometry,
)


@dataclass
class WorldTruth:
    """Hidden parameters of the synthetic world (never read by the twin)."""

    base: list[float] = field(default_factory=lambda: [1200.0, 400.0, 100.0])
    store_size: list[float] = field(default_factory=lambda: [1.10, 0.85, 0.95, 1.20, 1.05, 0.80, 0.90, 1.15, 1.00, 0.75, 0.95, 1.10])
    weekday: list[float] = field(default_factory=lambda: [0.86, 0.88, 0.93, 0.98, 1.10, 1.25, 1.00])
    # per category: amplitude 1, peak day-of-year 1, amplitude 2, peak day-of-year 2 (log scale)
    season: list[list[float]] = field(
        default_factory=lambda: [[0.06, 350.0, 0.02, 200.0], [0.10, 100.0, 0.03, 280.0], [0.45, 355.0, 0.12, 345.0]]
    )
    trend_per_year: float = 0.03
    promo_lift: list[float] = field(default_factory=lambda: [1.30, 1.55, 1.80])
    post_promo_dip: list[float] = field(default_factory=lambda: [0.93, 0.88, 0.92])
    ar_phi: float = 0.5
    ar_sd: float = 0.10
    transit_delay_p: float = 0.08


def calendar(start: str, n_days: int) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    dates = pd.date_range(start, periods=n_days, freq="D")
    return dates.dayofyear.to_numpy(), dates.dayofweek.to_numpy(), dates


def log_season(doy: np.ndarray, season: np.ndarray) -> np.ndarray:
    """(T,) day-of-year -> (T, C) log seasonal index."""
    s = np.asarray(season)
    w = 2 * np.pi / 365.25
    return s[:, 0] * np.cos(w * (doy[:, None] - s[:, 1])) + s[:, 2] * np.cos(2 * w * (doy[:, None] - s[:, 3]))


def true_mean(day0: int, n_days: int, promo: np.ndarray, post: np.ndarray, truth: WorldTruth) -> np.ndarray:
    """Expected demand (T, S, C) before AR shocks. day0 is the offset from HISTORY_START."""
    start = pd.Timestamp(HISTORY_START) + pd.Timedelta(days=day0)
    doy, dow, _ = calendar(str(start.date()), n_days)
    t_years = (day0 + np.arange(n_days)) / DAYS_PER_YEAR
    log_mu = (
        np.log(np.asarray(truth.base))[None, None, :]
        + np.log(np.asarray(truth.store_size))[None, :, None]
        + np.log(np.asarray(truth.weekday))[dow][:, None, None]
        + log_season(doy, np.asarray(truth.season))[:, None, :]
        + truth.trend_per_year * t_years[:, None, None]
        + (promo * np.log(truth.promo_lift))[:, None, :]
        + (post * np.log(truth.post_promo_dip))[:, None, :]
    )
    return np.exp(log_mu)


def weekly_season_index(day0: int, truth: WorldTruth) -> np.ndarray:
    """(52, C) true weekly seasonal index used by the merchandisers' promo rule in the history."""
    start = pd.Timestamp(HISTORY_START) + pd.Timedelta(days=day0)
    doy, _, _ = calendar(str(start.date()), DAYS_PER_YEAR)
    s = np.exp(log_season(doy, np.asarray(truth.season)))
    return s.reshape(52, 7, -1).mean(axis=1)


def draw_outages(n_days: int, rng: np.random.Generator, offset: int = 0) -> list[tuple[int, int, int]]:
    """Poisson supplier outages: list of (supplier, start_day, end_day) with end exclusive."""
    out = []
    for k, sup in enumerate(SUPPLIERS):
        n = rng.poisson(sup.outage_rate_per_year * n_days / DAYS_PER_YEAR)
        for start in np.sort(rng.integers(0, n_days, size=n)):
            dur = max(2, int(round(rng.gamma(2.0, sup.outage_mean_days / 2.0))))
            out.append((k, int(start) + offset, int(start) + offset + dur))
    return sorted(out, key=lambda r: (r[1], r[0]))


@dataclass
class WorldRun:
    truth: WorldTruth
    demand: np.ndarray  # (T, S, C) true demand (hidden)
    sales: np.ndarray  # (T, S, C)
    stockout: np.ndarray  # (T, S, C) bool, demand exceeded stock
    store_oh: np.ndarray  # (T, S, C) end of day
    dc_oh: np.ndarray  # (T, D, C) end of day
    ship: np.ndarray  # (T, S, C) units shipped DC -> store that day
    promo: np.ndarray  # (T, C)
    post: np.ndarray  # (T, C)
    shipments: list = field(repr=False, default_factory=list)
    pos: list = field(repr=False, default_factory=list)
    outages: list = field(repr=False, default_factory=list)
    snapshots: dict = field(repr=False, default_factory=dict)


def simulate_world(
    seed: int = SEED,
    n_days: int = HISTORY_DAYS,
    burn_in: int = 56,
    truth: WorldTruth | None = None,
    policy: Policy = BASELINE_POLICY,
    snapshot_days: tuple[int, ...] = (DAYS_PER_YEAR, HISTORY_DAYS),
) -> WorldRun:
    """Run the hidden world for ``n_days`` after a burn-in and return the full record."""
    truth = truth or WorldTruth()
    rng = np.random.default_rng(seed)
    geo = geometry()
    assign = np.array(HISTORICAL_ASSIGNMENT)
    cover = geo.transit[assign, np.arange(N_STORES)] + 1
    cap = np.array(FINANCE.dc_capacity_units_per_day)
    N = burn_in + n_days
    day0 = -burn_in

    # promo calendars, one 364-day block at a time (no promos during burn-in)
    promo = np.zeros((N, N_CATS))
    n_weeks = [c.promo_weeks_per_year for c in CATEGORIES]
    for b in range(int(np.ceil(n_days / DAYS_PER_YEAR))):
        idx = burn_in + b * DAYS_PER_YEAR
        weeks = promo_mod.current_rule(weekly_season_index(b * DAYS_PER_YEAR, truth), n_weeks, rng)
        block = promo_mod.weekly_to_daily(weeks)
        promo[idx : idx + DAYS_PER_YEAR] = block[: N - idx]
    post = np.vstack([promo_mod.post_promo(promo[i : i + DAYS_PER_YEAR]) for i in range(0, N, DAYS_PER_YEAR)])[:N]
    # carry the post-promo flag across block boundaries
    for i in range(DAYS_PER_YEAR, N, DAYS_PER_YEAR):
        last = promo[i - 7]
        post[i : i + 7] = np.where((last > 0) & (promo[i] == 0), 1.0, post[i : i + 7])

    mu = true_mean(day0, N, promo, post, truth)
    ar = np.zeros_like(mu)
    eps = rng.normal(0.0, truth.ar_sd * np.sqrt(1 - truth.ar_phi**2), size=mu.shape)
    ar[0] = rng.normal(0.0, truth.ar_sd, size=mu.shape[1:])
    for i in range(1, N):
        ar[i] = truth.ar_phi * ar[i - 1] + eps[i]
    demand = rng.poisson(mu * np.exp(ar)).astype(float)

    outages = draw_outages(N, rng, offset=day0)

    # state
    W = policy.forecast_window
    sales = np.zeros((N, N_STORES, N_CATS))
    stockout = np.zeros((N, N_STORES, N_CATS), dtype=bool)
    store_oh_rec = np.zeros_like(sales)
    dc_oh_rec = np.zeros((N, N_DCS, N_CATS))
    ship_rec = np.zeros_like(sales)
    outflow = np.zeros((N, N_DCS, N_CATS))
    store_oh = mu[0] * cover[:, None] * 1.3
    dc_oh = np.zeros((N_DCS, N_CATS))
    for d in range(N_DCS):
        if (assign == d).any():
            dc_oh[d] = mu[0][assign == d].sum(axis=0) * 30
    store_arr: dict[int, np.ndarray] = {}
    dc_arr: dict[int, np.ndarray] = {}
    in_transit = np.zeros((N_STORES, N_CATS))
    on_order = np.zeros((N_DCS, N_CATS))
    # warm-start the forecast windows with expected values
    sales[:W] = mu[:W]
    outflow_init = np.zeros((N_DCS, N_CATS))
    for d in range(N_DCS):
        outflow_init[d] = mu[0][assign == d].sum(axis=0)
    shipments: list[tuple] = []
    pos: list[tuple] = []
    snapshots: dict = {}
    snap_idx = {burn_in + s: s for s in snapshot_days}

    for i in range(N):
        if i in snap_idx:
            snapshots[snap_idx[i]] = _snapshot(i, store_oh, dc_oh, store_arr, dc_arr, sales, outflow, W, pos, day0)
        if i in store_arr:
            a = store_arr.pop(i)
            store_oh += a
            in_transit -= a
        if i in dc_arr:
            a = dc_arr.pop(i)
            dc_oh += a
            on_order -= a
        dem = demand[i]
        s_i = np.minimum(dem, store_oh)
        stockout[i] = dem > store_oh
        store_oh = store_oh - s_i
        sales[i] = s_i
        hist = sales[max(0, i - W + 1) : i + 1]
        soon = pol.promo_in_window(promo, i, cover)
        req = pol.store_orders(store_oh, in_transit, hist, cover, soon, policy)
        ship = pol.ration(req, assign, dc_oh, cap)
        for d in range(N_DCS):
            dc_oh[d] -= ship[assign == d].sum(axis=0)
            outflow[i, d] = req[assign == d].sum(axis=0)
        delay = rng.random(N_STORES) < truth.transit_delay_p
        transit = geo.transit[assign, np.arange(N_STORES)] + delay.astype(int)
        for s in range(N_STORES):
            if ship[s].sum() > 0:
                arr = i + int(transit[s])
                store_arr.setdefault(arr, np.zeros((N_STORES, N_CATS)))[s] += ship[s]
                shipments.append((i + day0, arr + day0, int(assign[s]), s, *ship[s].tolist()))
        in_transit += ship
        ship_rec[i] = ship
        if (i + day0) % 7 == 0:  # Monday
            oh_hist = outflow[max(0, i - W + 1) : i + 1]
            if i < W:
                oh_hist = np.vstack([np.repeat(outflow_init[None], W - i - 1, axis=0), oh_hist])
            orders = pol.dc_orders(dc_oh, on_order, oh_hist, policy)
            for d in range(N_DCS):
                if not (assign == d).any():
                    continue
                for k, sup in enumerate(SUPPLIERS):
                    q = orders[d, :, k]
                    if q.sum() <= 0:
                        continue
                    dispatch = i
                    for kk, st, en in outages:
                        if kk == k and st - day0 <= i < en - day0:
                            dispatch = max(dispatch, en - day0)
                    shape = 1.0 / sup.lead_cv**2
                    lead = max(1, int(np.ceil(rng.gamma(shape, sup.lead_mean_days / shape))))
                    receipt = dispatch + lead
                    dc_arr.setdefault(receipt, np.zeros((N_DCS, N_CATS)))[d] += q
                    on_order[d] += q
                    pos.append((i + day0, dispatch + day0, receipt + day0, k, d, *q.tolist()))
        store_oh_rec[i] = store_oh
        dc_oh_rec[i] = dc_oh

    if N in snap_idx:
        snapshots[snap_idx[N]] = _snapshot(N, store_oh, dc_oh, store_arr, dc_arr, sales, outflow, W, pos, day0)
    keep = slice(burn_in, N)
    return WorldRun(
        truth=truth,
        demand=demand[keep],
        sales=sales[keep],
        stockout=stockout[keep],
        store_oh=store_oh_rec[keep],
        dc_oh=dc_oh_rec[keep],
        ship=ship_rec[keep],
        promo=promo[keep],
        post=post[keep],
        shipments=[r for r in shipments if r[0] >= 0],
        pos=[r for r in pos if r[2] >= 0],
        outages=[o for o in outages if o[2] > 0],
        snapshots=snapshots,
    )


def _snapshot(i, store_oh, dc_oh, store_arr, dc_arr, sales, outflow, W, pos, day0) -> dict:
    """Start-of-day state at absolute index i (relative day numbers for pipelines)."""
    return {
        "store_on_hand": store_oh.tolist(),
        "dc_on_hand": dc_oh.tolist(),
        "store_pipeline": [[k - i, v.tolist()] for k, v in sorted(store_arr.items()) if k >= i],
        "dc_pipeline": [[r[2] - day0 - i, r[3], r[4], list(r[5:])] for r in pos if r[0] - day0 < i <= r[2] - day0],
        "sales_hist": sales[i - W : i].tolist(),
        "outflow_hist": outflow[i - W : i].tolist(),
    }


def history_tables(run: WorldRun) -> dict[str, pd.DataFrame]:
    """The observable historical record (what the retailer's systems would hold)."""
    T = run.sales.shape[0]
    _, _, dates = calendar(HISTORY_START, T)
    cats = [c.name for c in CATEGORIES]
    stores = [s.name for s in STORES]
    dcs = [d.name for d in DCS]
    tt, ss, cc = np.meshgrid(np.arange(T), np.arange(N_STORES), np.arange(N_CATS), indexing="ij")
    price = PRICE[None, :] * (1 - DISCOUNT[None, :] * run.promo)
    sales = pd.DataFrame(
        {
            "day": tt.ravel(),
            "date": dates[tt.ravel()].strftime("%Y-%m-%d"),
            "store": np.array(stores)[ss.ravel()],
            "category": np.array(cats)[cc.ravel()],
            "units_sold": run.sales.ravel().astype(int),
            "price": np.round(price[tt.ravel(), cc.ravel()], 2),
            "promo": run.promo[tt.ravel(), cc.ravel()].astype(int),
            "post_promo": run.post[tt.ravel(), cc.ravel()].astype(int),
            "stockout": run.stockout.ravel().astype(int),
            "on_hand_end": run.store_oh.ravel().astype(int),
        }
    )
    sh = pd.DataFrame(run.shipments, columns=["ship_day", "arrive_day", "dc_idx", "store_idx", *cats])
    sh = sh.melt(id_vars=["ship_day", "arrive_day", "dc_idx", "store_idx"], var_name="category", value_name="units")
    sh = sh[sh.units > 0]
    sh["dc"] = np.array(dcs)[sh.dc_idx]
    sh["store"] = np.array(stores)[sh.store_idx]
    sh = sh[["ship_day", "arrive_day", "dc", "store", "category", "units"]].sort_values(["ship_day", "store", "category"])
    po = pd.DataFrame(run.pos, columns=["order_day", "dispatch_day", "receipt_day", "sup_idx", "dc_idx", *cats])
    po = po.melt(id_vars=["order_day", "dispatch_day", "receipt_day", "sup_idx", "dc_idx"], var_name="category", value_name="units")
    po = po[po.units > 0]
    po["supplier"] = np.array([s.name for s in SUPPLIERS])[po.sup_idx]
    po["dc"] = np.array(dcs)[po.dc_idx]
    po = po[["order_day", "dispatch_day", "receipt_day", "supplier", "dc", "category", "units"]].sort_values(
        ["order_day", "dc", "supplier", "category"]
    )
    dd, dk, dc_ = np.meshgrid(np.arange(T), np.arange(N_DCS), np.arange(N_CATS), indexing="ij")
    dcinv = pd.DataFrame(
        {
            "day": dd.ravel(),
            "dc": np.array(dcs)[dk.ravel()],
            "category": np.array(cats)[dc_.ravel()],
            "on_hand_end": run.dc_oh.ravel().astype(int),
        }
    )
    dcinv = dcinv[dcinv.dc != "DC-East"]
    outages = pd.DataFrame([(SUPPLIERS[k].name, s, e) for k, s, e in run.outages], columns=["supplier", "start_day", "end_day"])
    for df in (sh, po):
        df["units"] = df["units"].astype(int)
    return {
        "sales": sales.reset_index(drop=True),
        "shipments": sh.reset_index(drop=True),
        "purchase_orders": po.reset_index(drop=True),
        "dc_inventory": dcinv.reset_index(drop=True),
        "outages": outages,
    }


def write_history(run: WorldRun, out_dir: Path) -> dict[str, int]:
    """Write observable tables (gzipped CSV), snapshots, and the hidden truth to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, df in history_tables(run).items():
        p = out_dir / f"{name}.csv.gz"
        df.to_csv(p, index=False, compression={"method": "gzip", "mtime": 0})
        sizes[name] = len(df)
    (out_dir / "snapshots.json").write_text(json.dumps({str(k): v for k, v in run.snapshots.items()}))
    truth_dir = out_dir / "_truth"
    truth_dir.mkdir(exist_ok=True)
    (truth_dir / "params.json").write_text(json.dumps(asdict(run.truth), indent=2))
    np.savez_compressed(truth_dir / "demand.npz", demand=run.demand)
    return sizes


def write_sample(run: WorldRun, out_dir: Path, n_days: int = 56) -> int:
    """Committed sample: first ``n_days`` of every observable table (<= 1 MB)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tabs = history_tables(run)
    filt = {
        "sales": tabs["sales"][tabs["sales"].day < n_days],
        "shipments": tabs["shipments"][tabs["shipments"].ship_day < n_days],
        "purchase_orders": tabs["purchase_orders"][tabs["purchase_orders"].order_day < n_days],
        "dc_inventory": tabs["dc_inventory"][tabs["dc_inventory"].day < n_days],
        "outages": tabs["outages"],
    }
    total = 0
    for name, df in filt.items():
        p = out_dir / f"{name}.csv.gz"
        df.to_csv(p, index=False, compression={"method": "gzip", "mtime": 0})
        total += p.stat().st_size
    return total
