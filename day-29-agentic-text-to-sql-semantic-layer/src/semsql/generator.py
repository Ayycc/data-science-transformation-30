"""Seeded generator for a SYNTHETIC B2B SaaS warehouse in DuckDB.

The warehouse is built to contain the semantics that break naive text-to-SQL:

* fiscal calendar: the fiscal year starts on 1 February and is named after the
  calendar year in which it ends (FY2025 = 2024-02-01 .. 2025-01-31);
* multi-currency billing: invoices and subscriptions are in the account's billing
  currency (USD, EUR, GBP, JPY, AUD, BRL) and must be converted with monthly FX rates;
* plan changes close a subscription with status 'cancelled' and open a new one the
  same day, so "cancelled subscriptions" is not churn;
* win-backs: some churned accounts return months later;
* void invoices and internal test accounts that must be excluded;
* PII columns (contact name, e-mail, phone) on the accounts table.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DATA_START = dt.date(2023, 2, 1)  # first day of FY2024
DATA_END = dt.date(2026, 1, 31)  # last day of FY2026
SEED = 20261108

REGIONS = {
    "NA": (0.40, {"USD": 1.0}),
    "EMEA": (0.30, {"EUR": 0.65, "GBP": 0.35}),
    "APAC": (0.18, {"JPY": 0.45, "AUD": 0.35, "USD": 0.20}),
    "LATAM": (0.12, {"BRL": 0.70, "USD": 0.30}),
}
SEGMENTS = {"SMB": 0.55, "Mid-Market": 0.32, "Enterprise": 0.13}
INDUSTRIES = [
    "Software",
    "Financial Services",
    "Healthcare",
    "Retail",
    "Manufacturing",
    "Education",
]
PLANS = ["Starter", "Growth", "Scale"]
PLAN_USD_MRR = {"Starter": 400.0, "Growth": 1500.0, "Scale": 6000.0}
PLAN_MIX = {
    "SMB": [0.70, 0.27, 0.03],
    "Mid-Market": [0.25, 0.60, 0.15],
    "Enterprise": [0.05, 0.35, 0.60],
}
SEAT_FACTOR = {"SMB": 1.0, "Mid-Market": 2.5, "Enterprise": 6.0}
MONTHLY_CHURN = {"SMB": 0.030, "Mid-Market": 0.016, "Enterprise": 0.007}
FX_BASE = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "JPY": 0.0068,
    "AUD": 0.66,
    "BRL": 0.19,
}
PRIORITIES = ["low", "medium", "high", "urgent"]
FIRST = [
    "Alex",
    "Sam",
    "Priya",
    "Chen",
    "Maria",
    "Tom",
    "Aisha",
    "Lars",
    "Yuki",
    "Diego",
    "Nora",
    "Omar",
]
LAST = [
    "Smith",
    "Garcia",
    "Kumar",
    "Li",
    "Muller",
    "Rossi",
    "Tanaka",
    "Silva",
    "Okafor",
    "Novak",
]


@dataclass(frozen=True)
class GenConfig:
    n_accounts: int = 2500
    seed: int = SEED
    test_account_share: float = 0.03


def _months(start: dt.date, end: dt.date) -> list[dt.date]:
    out, y, m = [], start.year, start.month
    while dt.date(y, m, 1) <= end:
        out.append(dt.date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _add_months(d: dt.date, k: int) -> dt.date:
    y, m = divmod(d.month - 1 + k, 12)
    y += d.year
    m += 1
    day = min(d.day, 28)
    return dt.date(y, m, day)


def fiscal_year(d: dt.date) -> int:
    return d.year + 1 if d.month >= 2 else d.year


def fiscal_quarter_num(d: dt.date) -> int:
    return ((d.month - 2) % 12) // 3 + 1


def make_dim_date() -> pd.DataFrame:
    days = pd.date_range("2023-01-01", "2026-12-31", freq="D")
    dates = [x.date() for x in days]
    fy = [fiscal_year(d) for d in dates]
    fq = [fiscal_quarter_num(d) for d in dates]
    return pd.DataFrame(
        {
            "date_day": dates,
            "calendar_year": [d.year for d in dates],
            "calendar_month": [f"{d.year}-{d.month:02d}" for d in dates],
            "fiscal_year": fy,
            "fiscal_quarter": [f"FY{y}-Q{q}" for y, q in zip(fy, fq)],
            "fiscal_quarter_num": fq,
        }
    )


def make_fx(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    months = _months(dt.date(2023, 1, 1), dt.date(2026, 12, 1))
    for cur, base in FX_BASE.items():
        level = 1.0
        for mth in months:
            if cur != "USD":
                level *= float(np.exp(rng.normal(0, 0.02)))
            rows.append({"currency": cur, "month": mth, "rate_to_usd": round(base * level, 6)})
    return pd.DataFrame(rows)


def generate(cfg: GenConfig | None = None) -> dict[str, pd.DataFrame]:
    """Return all warehouse tables as DataFrames (SYNTHETIC)."""
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)
    fx = make_fx(rng)
    fx_lookup = {(r.currency, r.month): r.rate_to_usd for r in fx.itertuples()}
    regions = list(REGIONS)
    region_p = [REGIONS[r][0] for r in regions]
    n_days = (dt.date(2025, 12, 31) - DATA_START).days

    accounts, subs, invoices, usage, tickets = [], [], [], [], []
    sub_id = inv_id = tkt_id = 0
    for acc_id in range(1, cfg.n_accounts + 1):
        region = str(rng.choice(regions, p=region_p))
        curs = REGIONS[region][1]
        currency = str(rng.choice(list(curs), p=list(curs.values())))
        segment = str(rng.choice(list(SEGMENTS), p=list(SEGMENTS.values())))
        is_test = bool(rng.random() < cfg.test_account_share)
        # signups accelerate over time (growing company)
        signup = DATA_START + dt.timedelta(days=int(n_days * rng.random() ** 0.8))
        first, last = str(rng.choice(FIRST)), str(rng.choice(LAST))
        accounts.append(
            {
                "account_id": acc_id,
                "account_name": f"{'Internal QA' if is_test else 'Customer'} {acc_id:05d}",
                "region": region,
                "segment": segment,
                "industry": str(rng.choice(INDUSTRIES)),
                "billing_currency": currency,
                "signup_date": signup,
                "is_test": is_test,
                "contact_name": f"{first} {last}",
                "contact_email": f"{first.lower()}.{last.lower()}{acc_id}@example.com",
                "contact_phone": f"+1-555-{rng.integers(1000, 9999)}",
            }
        )
        # subscription timeline
        plan_idx = int(rng.choice(3, p=PLAN_MIX[segment]))
        start = signup
        month_i = 0
        active_until: dt.date | None = None
        timeline: list[tuple[dt.date, dt.date | None, str]] = []
        cur_start = start
        churn_p = MONTHLY_CHURN[segment] * (3.0 if is_test else 1.0)
        while True:
            month_i += 1
            d = _add_months(start, month_i)
            if d > DATA_END:
                timeline.append((cur_start, None, PLANS[plan_idx]))
                break
            u = rng.random()
            if u < churn_p:
                timeline.append((cur_start, d, PLANS[plan_idx]))
                active_until = d
                # win-back
                if rng.random() < 0.15:
                    back = _add_months(d, int(rng.integers(2, 7)))
                    if back <= DATA_END - dt.timedelta(days=30):
                        start, cur_start, month_i = back, back, 0
                        active_until = None
                        continue
                break
            if u < churn_p + 0.012:  # plan change: close + reopen same day
                new_idx = min(2, plan_idx + 1) if rng.random() < 0.75 else max(0, plan_idx - 1)
                if new_idx != plan_idx:
                    timeline.append((cur_start, d, PLANS[plan_idx]))
                    plan_idx, cur_start = new_idx, d
        del active_until
        for s_start, s_end, plan in timeline:
            sub_id += 1
            mrr_usd = PLAN_USD_MRR[plan] * SEAT_FACTOR[segment] * float(rng.uniform(0.8, 1.25))
            if is_test:
                mrr_usd *= 20  # QA accounts carry absurd fake prices
            rate0 = fx_lookup[(currency, s_start.replace(day=1))]
            mrr_local = round(mrr_usd / rate0, 2)
            subs.append(
                {
                    "subscription_id": sub_id,
                    "account_id": acc_id,
                    "plan": plan,
                    "currency": currency,
                    "mrr_local": mrr_local,
                    "start_date": s_start,
                    "end_date": s_end,
                    "status": "active" if s_end is None else "cancelled",
                }
            )
            k = 0
            while True:
                inv_date = _add_months(s_start, k)
                if inv_date > DATA_END or (s_end is not None and inv_date >= s_end):
                    break
                inv_id += 1
                u = rng.random()
                status = "void" if u < 0.03 else ("open" if u < 0.07 else "paid")
                invoices.append(
                    {
                        "invoice_id": inv_id,
                        "account_id": acc_id,
                        "subscription_id": sub_id,
                        "invoice_date": inv_date,
                        "currency": currency,
                        "amount_local": mrr_local,
                        "status": status,
                    }
                )
                mth = inv_date.replace(day=1)
                seats = SEAT_FACTOR[segment] * {"Starter": 5, "Growth": 15, "Scale": 40}[plan]
                usage.append(
                    {
                        "account_id": acc_id,
                        "usage_month": mth,
                        "active_users": int(max(1, rng.poisson(seats))),
                        "api_calls": int(rng.poisson(seats * 900)),
                    }
                )
                n_t = rng.poisson(0.25 * SEAT_FACTOR[segment] ** 0.5)
                for _ in range(int(n_t)):
                    tkt_id += 1
                    opened = dt.datetime.combine(inv_date, dt.time()) + dt.timedelta(minutes=int(rng.integers(0, 28 * 24 * 60)))
                    if opened.date() > DATA_END:
                        continue
                    pr = str(rng.choice(PRIORITIES, p=[0.35, 0.35, 0.22, 0.08]))
                    base_h = {"low": 40, "medium": 24, "high": 10, "urgent": 4}[pr]
                    resolved = None
                    if rng.random() > 0.05:
                        resolved = opened + dt.timedelta(hours=float(rng.gamma(2.0, base_h / 2.0)))
                    csat = None
                    if resolved is not None and rng.random() < 0.6:
                        p5 = 0.45 if segment == "Enterprise" else 0.35
                        csat = int(
                            rng.choice(
                                [1, 2, 3, 4, 5],
                                p=[0.05, 0.08, 0.17, 0.35 - (p5 - 0.35), p5],
                            )
                        )
                    tickets.append(
                        {
                            "ticket_id": tkt_id,
                            "account_id": acc_id,
                            "opened_at": opened,
                            "resolved_at": resolved,
                            "priority": pr,
                            "csat_score": csat,
                        }
                    )
                k += 1
    usage_df = pd.DataFrame(usage).groupby(["account_id", "usage_month"], as_index=False).sum()
    return {
        "accounts": pd.DataFrame(accounts),
        "subscriptions": pd.DataFrame(subs),
        "invoices": pd.DataFrame(invoices),
        "fx_rates": fx,
        "usage_monthly": usage_df,
        "support_tickets": pd.DataFrame(tickets),
        "dim_date": make_dim_date(),
    }


DDL = {
    "accounts": """CREATE TABLE accounts (account_id INTEGER PRIMARY KEY, account_name VARCHAR,
        region VARCHAR, segment VARCHAR, industry VARCHAR, billing_currency VARCHAR,
        signup_date DATE, is_test BOOLEAN, contact_name VARCHAR, contact_email VARCHAR,
        contact_phone VARCHAR)""",
    "subscriptions": """CREATE TABLE subscriptions (subscription_id INTEGER PRIMARY KEY,
        account_id INTEGER, plan VARCHAR, currency VARCHAR, mrr_local DOUBLE, start_date DATE,
        end_date DATE, status VARCHAR)""",
    "invoices": """CREATE TABLE invoices (invoice_id INTEGER PRIMARY KEY, account_id INTEGER,
        subscription_id INTEGER, invoice_date DATE, currency VARCHAR, amount_local DOUBLE,
        status VARCHAR)""",
    "fx_rates": "CREATE TABLE fx_rates (currency VARCHAR, month DATE, rate_to_usd DOUBLE)",
    "usage_monthly": """CREATE TABLE usage_monthly (account_id INTEGER, usage_month DATE,
        active_users INTEGER, api_calls BIGINT)""",
    "support_tickets": """CREATE TABLE support_tickets (ticket_id INTEGER PRIMARY KEY,
        account_id INTEGER, opened_at TIMESTAMP, resolved_at TIMESTAMP, priority VARCHAR,
        csat_score INTEGER)""",
    "dim_date": """CREATE TABLE dim_date (date_day DATE PRIMARY KEY, calendar_year INTEGER,
        calendar_month VARCHAR, fiscal_year INTEGER, fiscal_quarter VARCHAR,
        fiscal_quarter_num INTEGER)""",
}


def write_duckdb(tables: dict[str, pd.DataFrame], path: Path) -> dict[str, int]:
    """Write the tables into a fresh DuckDB file and return row counts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for p in (path, path.with_suffix(path.suffix + ".wal")):
        if p.exists():
            p.unlink()
    con = duckdb.connect(str(path))
    counts = {}
    for name, df in tables.items():
        con.execute(DDL[name])
        con.register("_df", df)
        cols = ", ".join(df.columns)
        con.execute(f"INSERT INTO {name} ({cols}) SELECT {cols} FROM _df")
        con.unregister("_df")
        counts[name] = int(con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
    con.close()
    return counts


def write_sample(tables: dict[str, pd.DataFrame], out_dir: Path, n_accounts: int = 60) -> int:
    """Write a small CSV sample (first n accounts) and return total bytes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, df in tables.items():
        if name in ("dim_date", "fx_rates"):
            sub = df if name == "fx_rates" else df.iloc[0:0]
            if name == "dim_date":
                continue
        else:
            sub = df[df["account_id"] <= n_accounts]
        p = out_dir / f"{name}_SYNTHETIC_sample.csv"
        sub.to_csv(p, index=False)
        total += p.stat().st_size
    return total


def build_warehouse(path: Path, cfg: GenConfig | None = None) -> dict[str, int]:
    return write_duckdb(generate(cfg), path)
