"""Seeded synthetic generator for an omnichannel retailer (SYNTHETIC DATA).

Writes the bronze *landing zone*: Parquet files exactly as two source systems
(store POS and e-commerce) plus master-data exports would deliver them, with
realistic mess injected on purpose:

* duplicate events (the same order line re-sent in a later batch),
* late-arriving records (stores that were offline upload days or weeks late),
* timezone drift (POS exports switch from local time to UTC after a firmware rollout),
* price changes, product renames and SKU re-codes (catalog history for SCD2),
* returns (some arriving before their order line has landed),
* training-mode transactions, unmapped SKUs and inconsistent SKU casing,
* duplicate customer records from the CRM export.

A ground-truth file (`_truth/order_line_truth.parquet`) records the true UTC
timestamp of every unique order line. The pipeline never reads it; it is only
used to *measure* how well the silver layer normalised timestamps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

from retail_lakehouse.config import (
    CLOCK_CHANGES_SEED,
    END,
    SEED,
    START,
    TRAINING_STORE_ID,
    WEB_STORE_ID,
    Scale,
)

CITIES = [
    ("New York", "NY", "America/New_York"),
    ("Boston", "MA", "America/New_York"),
    ("Philadelphia", "PA", "America/New_York"),
    ("Atlanta", "GA", "America/New_York"),
    ("Miami", "FL", "America/New_York"),
    ("Charlotte", "NC", "America/New_York"),
    ("Pittsburgh", "PA", "America/New_York"),
    ("Chicago", "IL", "America/Chicago"),
    ("Dallas", "TX", "America/Chicago"),
    ("Houston", "TX", "America/Chicago"),
    ("Minneapolis", "MN", "America/Chicago"),
    ("St. Louis", "MO", "America/Chicago"),
    ("Nashville", "TN", "America/Chicago"),
    ("Austin", "TX", "America/Chicago"),
    ("Denver", "CO", "America/Denver"),
    ("Salt Lake City", "UT", "America/Denver"),
    ("Albuquerque", "NM", "America/Denver"),
    ("Phoenix", "AZ", "America/Phoenix"),
    ("Tucson", "AZ", "America/Phoenix"),
    ("Los Angeles", "CA", "America/Los_Angeles"),
    ("San Francisco", "CA", "America/Los_Angeles"),
    ("San Diego", "CA", "America/Los_Angeles"),
    ("Seattle", "WA", "America/Los_Angeles"),
    ("Portland", "OR", "America/Los_Angeles"),
    ("Las Vegas", "NV", "America/Los_Angeles"),
]

# category: (median price $, lognormal sigma, base return rate, nouns)
CATEGORIES: dict[str, tuple[float, float, float, list[str]]] = {
    "Apparel": (34.0, 0.45, 0.12, ["Tee", "Hoodie", "Jacket", "Chinos", "Dress", "Sweater"]),
    "Footwear": (79.0, 0.40, 0.10, ["Sneaker", "Boot", "Sandal", "Loafer", "Runner"]),
    "Home": (45.0, 0.60, 0.05, ["Throw", "Lamp", "Cushion", "Rug", "Vase", "Frame"]),
    "Kitchen": (29.0, 0.60, 0.04, ["Skillet", "Knife Set", "Mug", "Blender", "Kettle"]),
    "Electronics": (119.0, 0.70, 0.07, ["Earbuds", "Speaker", "Charger", "Tablet Case"]),
    "Beauty": (18.0, 0.45, 0.03, ["Serum", "Cleanser", "Lip Balm", "Moisturiser"]),
    "Toys": (24.0, 0.55, 0.04, ["Puzzle", "Building Set", "Plush", "Board Game"]),
    "Sports": (39.0, 0.60, 0.06, ["Yoga Mat", "Water Bottle", "Dumbbell", "Racket"]),
    "Grocery": (7.0, 0.40, 0.01, ["Coffee", "Granola", "Olive Oil", "Tea", "Snack Box"]),
    "Books": (16.0, 0.35, 0.02, ["Novel", "Cookbook", "Journal", "Atlas"]),
}
BRANDS = [
    "Northwind",
    "Harbor & Pine",
    "Lumen",
    "Copperfield",
    "Bluestem",
    "Kestrel",
    "Mapleworks",
    "Tidewater",
    "Ironbark",
    "Saltmarsh",
    "Fernhill",
    "Brightline",
    "Oakhaven",
    "Riverstone",
    "Summit Co",
    "Wildrose",
    "Granite Bay",
    "Pebble",
    "Juniper",
    "Atlas & Oak",
]
ADJECTIVES = [
    "Classic",
    "Essential",
    "Everyday",
    "Premium",
    "Heritage",
    "Urban",
    "Coastal",
    "Trail",
    "Studio",
    "Signature",
    "Compact",
    "Deluxe",
    "Organic",
    "Pro",
    "Lite",
    "Original",
]
RETURN_REASONS = ["DAMAGED", "WRONG_SIZE", "NOT_AS_DESCRIBED", "CHANGED_MIND", "OTHER"]

DOW_FACTOR = np.array([0.90, 0.90, 0.95, 1.00, 1.10, 1.30, 1.15])  # Mon..Sun
MONTH_FACTOR = {1: 0.80, 2: 0.85, 3: 0.95, 4: 0.95, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.05,
                9: 0.95, 10: 1.0, 11: 1.30, 12: 1.60}  # fmt: skip
POS_HOUR_W = np.array([2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 5, 4, 3, 2], dtype=float)  # 08..21 local
WEB_HOUR_W = np.array(
    [2, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 6, 6, 5, 5, 5, 6, 7, 8, 9, 9, 7, 4], dtype=float
)  # 00..23 in America/New_York (the web store's business-day timezone)


def _money_str(cents: np.ndarray) -> pd.Series:
    c = pd.Series(cents, dtype="int64")
    return (c // 100).astype(str) + "." + (c % 100).astype(str).str.zfill(2)


def _months_after(ts: pd.Timestamp) -> int:
    return (ts.year - START.year) * 12 + ts.month - START.month


def _make_stores(rng: np.random.Generator, n_stores: int) -> pd.DataFrame:
    ids = np.arange(1, n_stores + 1)
    city = [CITIES[(i * 7) % len(CITIES)] for i in ids]
    open_days = rng.integers(0, 365 * 15, n_stores)
    stores = pd.DataFrame(
        {
            "store_id": ids,
            "store_name": [f"{c[0]} #{i:03d}" for i, c in zip(ids, city)],
            "city": [c[0] for c in city],
            "state": [c[1] for c in city],
            "timezone": [c[2] for c in city],
            "store_format": rng.choice(["Flagship", "Standard", "Outlet"], n_stores, p=[0.1, 0.7, 0.2]),
            "opened_on": (pd.Timestamp("2009-01-01") + pd.to_timedelta(open_days, "D")).date,
        }
    )
    extra = pd.DataFrame(
        {
            "store_id": [WEB_STORE_ID, TRAINING_STORE_ID],
            "store_name": ["Web Store", "POS Training Lane"],
            "city": [None, "Chicago"],
            "state": [None, "IL"],
            "timezone": ["America/New_York", "America/Chicago"],
            "store_format": ["Online", "Training"],
            "opened_on": [pd.Timestamp("2012-03-01").date(), pd.Timestamp("2012-03-01").date()],
        }
    )
    return pd.concat([extra, stores], ignore_index=True)


def _make_products(rng: np.random.Generator, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (products, versions). versions holds every catalog state change."""
    cats = list(CATEGORIES)
    cat_w = np.array([0.20, 0.08, 0.12, 0.10, 0.08, 0.10, 0.08, 0.08, 0.10, 0.06])
    pid = np.arange(1, n + 1)
    category = rng.choice(cats, n, p=cat_w)
    brand = rng.choice(BRANDS, n)
    adj = rng.choice(ADJECTIVES, n)
    noun = [rng.choice(CATEGORIES[c][3]) for c in category]
    med = np.array([CATEGORIES[c][0] for c in category])
    sig = np.array([CATEGORIES[c][1] for c in category])
    price = np.maximum(2.0, np.round(med * np.exp(rng.normal(0, sig)))) - 0.01
    price_cents = np.round(price * 100).astype(np.int64)

    launched_late = rng.random(n) < 0.10
    pre = pd.Timestamp("2022-01-01") + pd.to_timedelta(
        rng.integers(0, (START - pd.Timestamp("2022-01-01")).days * 86400, n), "s"
    )
    late = START + pd.to_timedelta(rng.integers(0, (END - START).days * 86400 - 60 * 86400, n), "s")
    created = pd.Series(np.where(launched_late, late, pre)).astype("datetime64[s]")

    products = pd.DataFrame(
        {
            "product_id": pid,
            "category": category,
            "brand": brand,
            "created_at": created,
            "popularity": rng.permutation(np.arange(1, n + 1)),  # rank 1 = best seller
        }
    )

    # Pick which products get renamed / re-coded: exact counts, from products that
    # existed before the window so their sales straddle the change.
    pre_ids = pid[~launched_late]
    n_recode = max(3, round(0.015 * n))
    n_rename = max(3, round(0.03 * n))
    chosen = rng.choice(pre_ids, n_recode + n_rename, replace=False)
    recode_ids, rename_ids = set(chosen[:n_recode]), set(chosen[n_recode:])

    n_months = _months_after(END)  # 24 month buckets, extracts happen at month starts
    rows: list[tuple] = []
    for i in range(n):
        p = int(pid[i])
        name = f"{brand[i]} {adj[i]} {noun[i]}"
        sku = f"SKU-{100000 + p}"
        cents = int(price_cents[i])
        rows.append((p, created.iloc[i], sku, name, cents, "created"))
        first_m = max(0, _months_after(created.iloc[i]) + 1)
        months = np.arange(first_m, n_months)
        if len(months) == 0:
            continue
        kinds = ["price"] * int(min(rng.poisson(1.2), len(months)))
        if p in recode_ids:
            kinds.append("recode")
        if p in rename_ids:
            kinds.append("rename")
        kinds = kinds[: len(months)]
        if not kinds:
            continue
        ev_months = np.sort(rng.choice(months, len(kinds), replace=False))
        rng.shuffle(kinds)
        for m, kind in zip(ev_months, kinds):
            m_start = START + pd.DateOffset(months=int(m))
            m_len = ((m_start + pd.DateOffset(months=1)) - m_start).total_seconds()
            ts = (m_start + pd.Timedelta(seconds=int(rng.integers(0, int(m_len))))).floor("s")
            if kind == "price":
                f = rng.choice([0.85, 0.90, 0.95, 1.05, 1.08, 1.10, 1.15])
                cents = max(199, int(round(cents * f / 100.0)) * 100 - 1)
            elif kind == "rename":
                new_adj = rng.choice([a for a in ADJECTIVES if a not in name])
                name = f"{brand[i]} {new_adj} {noun[i]}"
            else:  # recode: new SKU code and a rebranded name ("renamed SKU")
                sku = f"SKU-{900000 + p}"
                name = f"{brand[i]} {noun[i]} 2.0"
            rows.append((p, ts, sku, name, cents, kind))
    versions = pd.DataFrame(
        rows, columns=["product_id", "valid_from", "sku_code", "product_name", "price_cents", "change"]
    )
    versions["valid_from"] = versions["valid_from"].astype("datetime64[s]")
    versions = versions.sort_values(["product_id", "valid_from"], kind="stable").reset_index(drop=True)
    return products, versions


def _catalog_extracts(products: pd.DataFrame, versions: pd.DataFrame) -> pd.DataFrame:
    meta = products.set_index("product_id")[["category", "brand"]]
    frames = []
    for m in range(_months_after(END) + 1):
        extract = START + pd.DateOffset(months=m)
        v = versions[versions["valid_from"] <= extract].groupby("product_id").tail(1)
        frames.append(
            pd.DataFrame(
                {
                    "extract_date": extract.date(),
                    "product_id": v["product_id"].to_numpy(),
                    "sku_code": v["sku_code"].to_numpy(),
                    "product_name": v["product_name"].to_numpy(),
                    "category": meta.loc[v["product_id"], "category"].to_numpy(),
                    "brand": meta.loc[v["product_id"], "brand"].to_numpy(),
                    "list_price": _money_str(v["price_cents"].to_numpy()).to_numpy(),
                    "updated_at": v["valid_from"].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _make_customers(rng: np.random.Generator, n: int, seed: int) -> pd.DataFrame:
    fake = Faker("en_US")
    fake.seed_instance(seed)
    span = (END - START).days
    active_from = START - pd.Timedelta(days=365) + pd.to_timedelta(rng.integers(0, (365 + span - 30) * 86400, n), "s")
    active_to = active_from + pd.to_timedelta(30 + rng.exponential(450, n), "D")
    city = [CITIES[j] for j in rng.integers(0, len(CITIES), n)]
    first = [fake.first_name() for _ in range(n)]
    last = [fake.last_name() for _ in range(n)]
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "first_name": first,
            "last_name": last,
            "city": [c[0] for c in city],
            "state": [c[1] for c in city],
            "active_from": pd.Series(active_from).astype("datetime64[s]"),
            "active_to": pd.Series(active_to).astype("datetime64[s]"),
            "weight": rng.gamma(0.7, 1.0, n),
            "loyalty_tier": rng.choice(["Bronze", "Silver", "Gold"], n, p=[0.6, 0.3, 0.1]),
        }
    )


def _localize(local: pd.Series, tz: pd.Series) -> pd.Series:
    """Convert naive local wall-clock times to naive UTC, per row timezone."""
    out = pd.Series(pd.NaT, index=local.index, dtype="datetime64[s]")
    for zone, idx in local.groupby(tz).groups.items():
        di = pd.DatetimeIndex(local.loc[idx])
        # DST: treat ambiguous wall times as standard time, shift non-existent ones forward
        loc = di.tz_localize(zone, ambiguous=np.zeros(len(di), dtype=bool), nonexistent="shift_forward")
        utc = loc.tz_convert("UTC").tz_localize(None)
        out.loc[idx] = utc.astype("datetime64[s]")
    return out


def generate(out_dir: Path, scale: Scale, seed: int = SEED) -> dict:
    """Generate the landing zone for `scale` into `out_dir`. Returns a manifest dict."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    stores = _make_stores(rng, scale.n_stores)
    products, versions = _make_products(rng, scale.n_products)
    customers = _make_customers(rng, scale.n_customers, seed)
    clock = pd.read_csv(CLOCK_CHANGES_SEED, parse_dates=["change_date"])

    # ---- orders -------------------------------------------------------------------------
    days = pd.date_range(START, END, freq="D", inclusive="left")
    t = np.arange(len(days)) / 365.0
    w = (1 + 0.12 * t) * DOW_FACTOR[days.dayofweek] * np.array([MONTH_FACTOR[m] for m in days.month])
    n_orders = int(round(scale.n_lines / 2.4))
    day_idx = rng.choice(len(days), n_orders, p=w / w.sum())
    is_web = rng.random(n_orders) < (0.25 + 0.10 * t[day_idx])
    phys = stores[(stores.store_id != WEB_STORE_ID) & (stores.store_id != TRAINING_STORE_ID)]
    sw = np.exp(rng.normal(0, 0.5, len(phys)))
    store_id = np.where(is_web, WEB_STORE_ID, rng.choice(phys.store_id.to_numpy(), n_orders, p=sw / sw.sum()))
    training = (~is_web) & (rng.random(n_orders) < 0.001)
    store_id = np.where(training, TRAINING_STORE_ID, store_id)

    day_ts = days[day_idx]
    secs = rng.integers(0, 3600, n_orders)
    pos_hour = 8 + rng.choice(len(POS_HOUR_W), n_orders, p=POS_HOUR_W / POS_HOUR_W.sum())
    web_hour = rng.choice(24, n_orders, p=WEB_HOUR_W / WEB_HOUR_W.sum())
    hour = np.where(is_web, web_hour, pos_hour)
    wall = pd.Series(day_ts + pd.to_timedelta(hour * 3600 + secs, "s")).astype("datetime64[s]")
    tz = pd.Series(store_id).map(stores.set_index("store_id")["timezone"])
    order_utc = _localize(wall, tz)  # every order happens on a local business day in the window

    orders = pd.DataFrame(
        {
            "order_seq": np.arange(n_orders),
            "is_web": is_web,
            "store_id": store_id,
            "order_utc": order_utc.to_numpy(),
            "local_wall": wall.to_numpy(),
        }  # fmt: skip
    )

    # customers: web always identified, POS only when a loyalty card is scanned
    need = is_web | (rng.random(n_orders) < 0.55)
    cust = np.zeros(n_orders, dtype=np.int64)
    cw = customers["weight"].to_numpy()
    af = customers["active_from"].to_numpy()
    at = customers["active_to"].to_numpy()
    todo = np.flatnonzero(need)
    ts_np = orders["order_utc"].to_numpy()
    for _ in range(40):
        if len(todo) == 0:
            break
        cand = rng.choice(len(cw), len(todo), p=cw / cw.sum())
        ok = (af[cand] <= ts_np[todo]) & (ts_np[todo] <= at[cand])
        cust[todo[ok]] = cand[ok] + 1
        todo = todo[~ok]
    if len(todo):  # rare leftovers: anyone already active
        cust[todo] = rng.choice(np.flatnonzero(af <= START.to_datetime64()), len(todo)) + 1
    orders["customer_id"] = cust

    # ---- order lines --------------------------------------------------------------------
    n_lines_per = 1 + rng.poisson(1.4, n_orders)
    li = np.repeat(np.arange(n_orders), n_lines_per)
    n_lines = len(li)
    line_no = np.arange(n_lines) - np.repeat(np.cumsum(n_lines_per) - n_lines_per, n_lines_per) + 1
    pop_w = 1.0 / products["popularity"].to_numpy() ** 0.8
    pop_w /= pop_w.sum()
    created = products["created_at"].to_numpy()
    line_ts = ts_np[li]
    prod_idx = rng.choice(len(products), n_lines, p=pop_w)
    bad = np.flatnonzero(created[prod_idx] > line_ts)
    for _ in range(20):
        if len(bad) == 0:
            break
        prod_idx[bad] = rng.choice(len(products), len(bad), p=pop_w)
        bad = bad[created[prod_idx[bad]] > line_ts[bad]]
    if len(bad):
        always = np.flatnonzero(created <= START.to_datetime64())
        prod_idx[bad] = always[np.argmin(products["popularity"].to_numpy()[always])]
    product_id = products["product_id"].to_numpy()[prod_idx]

    # catalog state (sku code, list price) at the moment of sale
    vsec = versions["valid_from"].to_numpy().astype("datetime64[s]").astype(np.int64)
    vkey = versions["product_id"].to_numpy().astype(np.int64) * 10**11 + vsec
    lkey = product_id.astype(np.int64) * 10**11 + line_ts.astype("datetime64[s]").astype(np.int64)
    vi = np.searchsorted(vkey, lkey, side="right") - 1
    assert (versions["product_id"].to_numpy()[vi] == product_id).all()
    sku = versions["sku_code"].to_numpy()[vi].astype(object)
    price = versions["price_cents"].to_numpy()[vi]
    qty = np.minimum(rng.geometric(0.72, n_lines), 6)
    disc_pct = rng.choice([0.0, 0.10, 0.15, 0.25], n_lines, p=[0.70, 0.15, 0.10, 0.05])
    gross = qty * price
    disc = np.round(gross * disc_pct).astype(np.int64)
    net = gross - disc

    lines = pd.DataFrame(
        {
            "order_line_id": "OL" + pd.Series(np.arange(1, n_lines + 1)).astype(str).str.zfill(9),
            "order_seq": li,
            "line_number": line_no,
            "product_id": product_id,
            "sku_code": sku,
            "quantity": qty,
            "unit_price_cents": price,
            "discount_cents": disc,
            "net_cents": net,
        }
    )
    lines = lines.merge(orders, on="order_seq", how="left")
    lines["order_id"] = np.where(
        lines["is_web"], "WO" + lines["order_seq"].astype(str).str.zfill(8),
        "SO" + lines["order_seq"].astype(str).str.zfill(8),
    )  # fmt: skip

    # ingestion: normal lag 0.5-30h; offline stores upload 2-45 days late; web backlog 2-10 days
    oseq = orders["order_seq"].to_numpy()
    lag_h = rng.uniform(0.5, 30, n_orders)
    late_pos = (~is_web) & (rng.random(n_orders) < 0.03)
    late_web = is_web & (rng.random(n_orders) < 0.01)
    lag_h = np.where(late_pos, rng.uniform(48, 45 * 24, n_orders), lag_h)
    lag_h = np.where(late_web, rng.uniform(48, 10 * 24, n_orders), lag_h)
    ing = pd.Series(ts_np + pd.to_timedelta(lag_h * 3600, "s").to_numpy()).astype("datetime64[s]")
    lines["_ingested_at"] = ing.to_numpy()[lines["order_seq"].to_numpy()]
    assert (oseq == np.arange(n_orders)).all()

    # unmapped SKUs (scanner/manual-entry errors)
    unmapped = rng.random(n_lines) < 0.001
    lines.loc[unmapped & lines["is_web"].to_numpy(), "sku_code"] = None
    lines.loc[unmapped & ~lines["is_web"].to_numpy(), "sku_code"] = "SKU-UNKNOWN"
    lines["is_unmapped"] = unmapped
    lines["is_training"] = lines["store_id"].to_numpy() == TRAINING_STORE_ID

    truth = lines[["order_line_id", "order_utc"]].rename(columns={"order_utc": "true_order_ts_utc"})

    # duplicates: 1.5% of lines re-sent 1-10 days later
    dup_mask = rng.random(n_lines) < 0.015
    dups = lines[dup_mask].copy()
    dups["_ingested_at"] = (dups["_ingested_at"] + pd.to_timedelta(rng.uniform(1, 10, len(dups)), "D")).astype(
        "datetime64[s]"
    )
    raw = pd.concat([lines, dups], ignore_index=True)
    raw["batch"] = raw["_ingested_at"].dt.strftime("%Y-%m")

    # ---- POS export (strings everywhere, local wall-clock time, clock drift) -------------
    pos = raw[~raw["is_web"]].copy()
    drift = pos[["store_id"]].merge(clock[["store_id", "change_date"]], on="store_id", how="left")
    drifted = (
        drift["change_date"].notna() & (pos["local_wall"].to_numpy() >= drift["change_date"].to_numpy())
    ).to_numpy()
    shown_ts = np.where(drifted, pos["order_utc"].to_numpy(), pos["local_wall"].to_numpy())
    pos_sku = pos["sku_code"].astype(object)
    messy = rng.random(len(pos)) < 0.02
    pos_sku = np.where(messy & pos_sku.notna().to_numpy(), pos_sku.str.lower() + " ", pos_sku)
    pos_out = pd.DataFrame(
        {
            "order_line_id": pos["order_line_id"].to_numpy(),
            "receipt_no": pos["order_id"].to_numpy(),
            "line_no": pos["line_number"].to_numpy().astype(np.int32),
            "store_id": pos["store_id"].to_numpy().astype(np.int32),
            "loyalty_card": np.where(
                pos["customer_id"].to_numpy() > 0,
                "C" + pos["customer_id"].astype(str).str.zfill(7),
                None,
            ),  # fmt: skip
            "sku": pos_sku,
            "qty": pos["quantity"].to_numpy().astype(np.int32),
            "unit_price": _money_str(pos["unit_price_cents"].to_numpy()).to_numpy(),
            "discount": _money_str(pos["discount_cents"].to_numpy()).to_numpy(),
            "line_total": _money_str(pos["net_cents"].to_numpy()).to_numpy(),
            "txn_ts": pd.Series(shown_ts).astype("datetime64[s]").dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
            "_ingested_at": pos["_ingested_at"].to_numpy(),
            "batch": pos["batch"].to_numpy(),
        }
    )

    # ---- web export (integers in cents, epoch-millisecond event time in UTC) ------------
    web = raw[raw["is_web"]]
    web_out = pd.DataFrame(
        {
            "order_line_id": web["order_line_id"].to_numpy(),
            "order_id": web["order_id"].to_numpy(),
            "line_number": web["line_number"].to_numpy().astype(np.int32),
            "customer_id": web["customer_id"].to_numpy().astype(np.int64),
            "sku_code": web["sku_code"].to_numpy(),
            "quantity": web["quantity"].to_numpy().astype(np.int32),
            "unit_price_cents": web["unit_price_cents"].to_numpy().astype(np.int64),
            "discount_cents": web["discount_cents"].to_numpy().astype(np.int64),
            "event_time_ms": web["order_utc"].to_numpy().astype("datetime64[ms]").astype(np.int64),
            "_ingested_at": web["_ingested_at"].to_numpy(),
            "batch": web["batch"].to_numpy(),
        }
    )

    # ---- returns ------------------------------------------------------------------------
    cat_of = products.set_index("product_id")["category"]
    rr = cat_of.loc[lines["product_id"]].map({k: v[2] for k, v in CATEGORIES.items()}).to_numpy()
    rr = rr * np.where(lines["is_web"].to_numpy(), 1.6, 1.0)
    eligible = ~(lines["is_unmapped"].to_numpy() | lines["is_training"].to_numpy())
    ret_mask = eligible & (rng.random(n_lines) < rr)
    r = lines[ret_mask]
    r_ts = (r["order_utc"] + pd.to_timedelta(rng.uniform(1, 60, len(r)) * 86400, "s")).astype("datetime64[s]")
    keep = (r_ts < END).to_numpy()
    r, r_ts = r[keep], r_ts[keep]
    r_qty = np.array([rng.integers(1, q + 1) for q in r["quantity"].to_numpy()], dtype=np.int32)
    refund = np.round(r["net_cents"].to_numpy() * r_qty / r["quantity"].to_numpy()).astype(np.int64)
    returns = pd.DataFrame(
        {
            "return_id": "RT" + pd.Series(np.arange(1, len(r) + 1)).astype(str).str.zfill(8),
            "order_line_id": r["order_line_id"].to_numpy(),
            "return_channel": np.where(r["is_web"].to_numpy(), "web", "store"),
            "returned_qty": r_qty,
            "refund_cents": refund,
            "reason_code": rng.choice(RETURN_REASONS, len(r), p=[0.2, 0.3, 0.2, 0.25, 0.05]),
            "return_ts": r_ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ").to_numpy(),
            "_ingested_at": (r_ts + pd.to_timedelta(rng.uniform(1, 48, len(r)) * 3600, "s"))
            .astype("datetime64[s]")
            .to_numpy(),
        }
    )
    rdup = returns[rng.random(len(returns)) < 0.01].copy()
    rdup["_ingested_at"] = (rdup["_ingested_at"] + pd.to_timedelta(rng.uniform(1, 5, len(rdup)), "D")).astype(
        "datetime64[s]"
    )
    returns = pd.concat([returns, rdup], ignore_index=True)
    returns["batch"] = pd.Series(returns["_ingested_at"]).dt.strftime("%Y-%m")

    # ---- customer master (CRM export with duplicate records) ----------------------------
    first_order = pd.Series(ts_np[cust > 0]).groupby(cust[cust > 0]).min()
    signup = customers["active_from"].copy()
    fo = customers["customer_id"].map(first_order)
    signup = signup.where(fo.isna() | (signup <= fo), fo - pd.Timedelta(days=1)).astype("datetime64[s]")
    email = (customers["first_name"].str.lower() + "." + customers["last_name"].str.lower()
             + customers["customer_id"].astype(str) + "@example.com").str.replace(" ", "")  # fmt: skip
    crm = pd.DataFrame(
        {
            "customer_id": customers["customer_id"].to_numpy(),
            "customer_ref": "C" + customers["customer_id"].astype(str).str.zfill(7),
            "first_name": customers["first_name"],
            "last_name": customers["last_name"],
            "email": email,
            "city": customers["city"],
            "state": customers["state"],
            "loyalty_tier": customers["loyalty_tier"],
            "signup_ts": signup.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": (END - pd.Timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    old = crm[rng.random(len(crm)) < 0.02].copy()
    old["email"] = old["email"].str.replace("@example.com", "@oldmail.example.com", regex=False)
    old["updated_at"] = (START - pd.Timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    crm = pd.concat([crm, old], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    catalog = _catalog_extracts(products, versions)

    # ---- write landing zone -------------------------------------------------------------
    for sub in ["pos_order_lines", "web_order_lines", "order_returns", "_truth"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
        for f in (out_dir / sub).glob("*.parquet"):
            f.unlink()
    for name, df in [("pos_order_lines", pos_out), ("web_order_lines", web_out), ("order_returns", returns)]:
        for batch, part in df.groupby("batch", sort=True):
            part.drop(columns="batch").to_parquet(
                out_dir / name / f"{name}_{batch}.parquet", index=False, compression="zstd"
            )
    catalog.to_parquet(out_dir / "product_catalog.parquet", index=False, compression="zstd")
    crm.to_parquet(out_dir / "customers.parquet", index=False, compression="zstd")
    stores.to_parquet(out_dir / "stores.parquet", index=False, compression="zstd")
    truth.to_parquet(out_dir / "_truth" / "order_line_truth.parquet", index=False, compression="zstd")

    manifest = {
        "generator": "retail_lakehouse.generate (SYNTHETIC)",
        "scale": scale.name,
        "seed": seed,
        "window_utc": [str(START), str(END)],
        "orders": int(n_orders),
        "unique_order_lines": int(n_lines),
        "raw_order_line_rows": int(len(raw)),
        "duplicate_rows_injected": int(len(dups)),
        "late_pos_orders_injected": int(late_pos.sum()),
        "late_web_orders_injected": int(late_web.sum()),
        "training_lines_injected": int(lines["is_training"].sum()),
        "unmapped_sku_lines_injected": int(unmapped.sum()),
        "pos_rows_with_clock_drift": int(drifted.sum()),
        "pos_rows_with_messy_sku_format": int(messy.sum()),
        "returns_unique": int(len(returns) - len(rdup)),
        "return_duplicate_rows_injected": int(len(rdup)),
        "customers": int(scale.n_customers),
        "customer_duplicate_rows_injected": int(len(old)),
        "products": int(scale.n_products),
        "catalog_versions": int(len(versions)),
        "price_changes": int((versions["change"] == "price").sum()),
        "renames": int((versions["change"] == "rename").sum()),
        "sku_recodes": int((versions["change"] == "recode").sum()),
        "catalog_extracts": int(catalog["extract_date"].nunique()),
        "stores_physical": int(scale.n_stores),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
