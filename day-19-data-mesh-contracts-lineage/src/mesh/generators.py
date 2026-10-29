"""Three independent, seeded SYNTHETIC domain generators (orders, payments, logistics).

Each domain has its own source-system release cadence and its own *raw* schema evolution. The raw
feeds are what an upstream operational system emits; the domain team's dbt project absorbs those
changes so the published data product (governed by its contract) stays stable.

Raw schema timeline (day index from START):

* orders    (daily files)          r1 day 0  | r2 day 21 adds ``sales_channel`` |
                                   r3 day 49 renames ``order_total`` -> ``order_amount`` and starts
                                   reporting status ``returned`` | r4 day 70 adds ``discount_amount``
* payments  (daily files)          r1 day 0 dollars ``amount`` | r2 day 35 switches to integer
                                   ``amount_cents`` and adds ``payment_method``
* logistics (batch every 3 days)   r1 day 0 | r2 day 56 renames ``carrier`` -> ``carrier_code`` and
                                   adds ``carrier_service``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from mesh.config import N_DAYS, SEED, START


@dataclass(frozen=True)
class RawRelease:
    day: int
    tag: str
    columns: tuple[str, ...]
    note: str


RAW_RELEASES: dict[str, list[RawRelease]] = {
    "orders": [
        RawRelease(0, "r1", ("order_id", "customer_id", "order_ts", "order_total", "currency", "status"), "initial feed"),
        RawRelease(
            21, "r2", ("order_id", "customer_id", "order_ts", "order_total", "currency", "status", "sales_channel"), "adds sales_channel"
        ),
        RawRelease(
            49,
            "r3",
            ("order_id", "customer_id", "order_ts", "order_amount", "currency", "status", "sales_channel"),
            "renames order_total -> order_amount; status gains 'returned'",
        ),
        RawRelease(
            70,
            "r4",
            ("order_id", "customer_id", "order_ts", "order_amount", "discount_amount", "currency", "status", "sales_channel"),
            "adds discount_amount",
        ),
    ],
    "payments": [
        RawRelease(0, "r1", ("payment_id", "order_id", "paid_at", "amount", "currency", "status"), "initial feed, amount in dollars"),
        RawRelease(
            35,
            "r2",
            ("payment_id", "order_id", "paid_at", "amount_cents", "currency", "status", "payment_method"),
            "amount -> integer amount_cents; adds payment_method",
        ),
    ],
    "logistics": [
        RawRelease(0, "r1", ("shipment_id", "order_id", "carrier", "shipped_at", "delivered_at", "promised_date"), "initial feed"),
        RawRelease(
            56,
            "r2",
            ("shipment_id", "order_id", "carrier_code", "carrier_service", "shipped_at", "delivered_at", "promised_date"),
            "renames carrier -> carrier_code; adds carrier_service",
        ),
    ],
}

# Logistics lands in batches every 3 days (covering the previous 3 days), so it is structurally staler.
LOGISTICS_BATCH_DAYS = 3


def raw_release_on(domain: str, day: int) -> RawRelease:
    rel = RAW_RELEASES[domain][0]
    for r in RAW_RELEASES[domain]:
        if r.day <= day:
            rel = r
    return rel


def _ts(day: int, seconds: np.ndarray) -> pd.Series:
    base = START + timedelta(days=day)
    return pd.Series(pd.to_datetime(base) + pd.to_timedelta(seconds, unit="s"))


def simulate_orders(n_days: int = N_DAYS, seed: int = SEED, base_rate: float = 420.0) -> pd.DataFrame:
    """Canonical (truth) order table; raw files are projections of it per release."""
    rng = np.random.default_rng(seed + 1)
    frames = []
    next_id = 1
    for d in range(n_days):
        dow = (START + timedelta(days=d)).weekday()
        rate = base_rate * (1.0 + 0.004 * d) * (1.15 if dow >= 5 else 1.0)
        n = int(rng.poisson(rate))
        secs = np.sort(rng.integers(0, 86400, n))
        amount = np.round(np.exp(rng.normal(3.9, 0.55, n)), 2)
        status = rng.choice(["completed", "cancelled", "returned"], n, p=[0.87, 0.05, 0.08])
        channel = rng.choice(["web", "app", "marketplace"], n, p=[0.5, 0.35, 0.15])
        discount = np.where(rng.random(n) < 0.2, np.round(amount * rng.uniform(0.05, 0.2, n), 2), 0.0)
        frames.append(
            pd.DataFrame(
                {
                    "order_id": [f"O{next_id + i:07d}" for i in range(n)],
                    "customer_id": [f"C{c:05d}" for c in rng.integers(1, 8001, n)],
                    "order_ts": _ts(d, secs),
                    "amount": amount,
                    "discount_amount": discount,
                    "currency": "USD",
                    "status": status,
                    "sales_channel": channel,
                    "day": d,
                }
            )
        )
        next_id += n
    return pd.concat(frames, ignore_index=True)


def simulate_payments(orders: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)
    n = len(orders)
    rows = []
    delay = rng.integers(5, 900, n)
    failed_first = rng.random(n) < 0.03
    method = rng.choice(["card", "wallet", "bnpl"], n, p=[0.7, 0.22, 0.08])
    refund_delay_days = rng.integers(3, 15, n)
    orphan = rng.random(n) < 0.002  # upstream bug: a few payments lose their order reference
    pid = 1
    for i, o in enumerate(orders.itertuples(index=False)):
        paid = o.order_ts + timedelta(seconds=int(delay[i]))
        oid = None if orphan[i] else o.order_id
        if failed_first[i]:
            rows.append((f"P{pid:08d}", oid, paid, o.amount, "failed", method[i]))
            pid += 1
            paid = paid + timedelta(seconds=120)
        if o.status == "cancelled":
            rows.append((f"P{pid:08d}", oid, paid, o.amount, "failed", method[i]))
            pid += 1
            continue
        rows.append((f"P{pid:08d}", oid, paid, o.amount, "captured", method[i]))
        pid += 1
        if o.status == "returned":
            rows.append((f"P{pid:08d}", oid, paid + timedelta(days=int(refund_delay_days[i])), o.amount, "refunded", method[i]))
            pid += 1
    df = pd.DataFrame(rows, columns=["payment_id", "order_id", "paid_at", "amount", "status", "payment_method"])
    df["currency"] = "USD"
    df["day"] = (df["paid_at"] - pd.Timestamp(START)).dt.days
    return df.sort_values("paid_at", ignore_index=True)


def simulate_shipments(orders: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 3)
    ship = orders[orders["status"] != "cancelled"].reset_index(drop=True)
    n = len(ship)
    carrier = rng.choice(["UPS", "FDX", "DHL", "USPS"], n, p=[0.35, 0.3, 0.15, 0.2])
    lateness = {"UPS": 0.0, "FDX": 0.2, "DHL": 0.5, "USPS": 0.8}
    shipped = ship["order_ts"] + pd.to_timedelta(rng.integers(2 * 3600, 30 * 3600, n), unit="s")
    transit_days = rng.gamma(4.0, 0.6, n) + np.array([lateness[c] for c in carrier])
    delivered = shipped + pd.to_timedelta(np.round(transit_days * 86400), unit="s")
    service = np.where(rng.random(n) < 0.18, "express", "ground")
    df = pd.DataFrame(
        {
            "shipment_id": [f"S{i + 1:07d}" for i in range(n)],
            "order_id": ship["order_id"],
            "carrier": carrier,
            "carrier_service": service,
            "shipped_at": shipped,
            "delivered_at": delivered,
            "promised_date": (ship["order_ts"].dt.normalize() + pd.Timedelta(days=4)).dt.date,
        }
    )
    df["day"] = (df["shipped_at"] - pd.Timestamp(START)).dt.days
    return df


def _loaded_at(domain: str, day: int) -> datetime | None:
    """When the file covering ``day`` lands in the warehouse (None if not yet landed by end of horizon)."""
    if domain == "logistics":
        batch_end = (day // LOGISTICS_BATCH_DAYS) * LOGISTICS_BATCH_DAYS + LOGISTICS_BATCH_DAYS - 1
        if batch_end >= N_DAYS - 1:
            return None  # batch not complete yet at the end of the horizon
        return START + timedelta(days=batch_end + 1, hours=2)
    return START + timedelta(days=day + 1, hours=1)


def project_raw(domain: str, df: pd.DataFrame, day: int) -> pd.DataFrame:
    """Project the canonical rows of one day into that day's raw source schema."""
    rel = raw_release_on(domain, day)
    out = df.copy()
    if domain == "orders":
        out["order_total"] = out["amount"]
        out["order_amount"] = out["amount"]
        if rel.tag in ("r1", "r2"):
            out["status"] = out["status"].replace({"returned": "completed"})  # legacy system did not report returns
    elif domain == "payments":
        out["amount_cents"] = np.round(out["amount"] * 100).astype("int64")
    elif domain == "logistics":
        out["carrier_code"] = out["carrier"]
    return out[list(rel.columns)]


def write_raw(out_dir: Path, n_days: int = N_DAYS, seed: int = SEED, base_rate: float = 420.0) -> dict[str, int]:
    """Write one CSV per domain per landed file. Returns row counts per domain."""
    orders = simulate_orders(n_days, seed, base_rate)
    tables = {"orders": orders, "payments": simulate_payments(orders, seed), "logistics": simulate_shipments(orders, seed)}
    counts: dict[str, int] = {}
    for domain, df in tables.items():
        ddir = out_dir / domain
        ddir.mkdir(parents=True, exist_ok=True)
        for f in ddir.glob("*.csv"):
            f.unlink()
        df = df[df["day"] < n_days]
        # Orders feed retries: 0.4% of rows are re-sent (duplicates) in the same file.
        rng = np.random.default_rng(seed + 10)
        total = 0
        for day, part in df.groupby("day", sort=True):
            loaded = _loaded_at(domain, int(day))
            if loaded is None:
                continue
            raw = project_raw(domain, part, int(day))
            if domain == "orders":
                dup = raw[rng.random(len(raw)) < 0.004]
                raw = pd.concat([raw, dup], ignore_index=True)
            raw = raw.copy()
            raw["_loaded_at"] = loaded
            raw.to_csv(ddir / f"{domain}_{(START + timedelta(days=int(day))).date()}.csv", index=False)
            total += len(raw)
        counts[domain] = total
    return counts
