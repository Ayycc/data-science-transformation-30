"""Seeded SYNTHETIC card-transaction stream with injected fraud campaigns.

Legitimate behaviour
- Each cardholder has a home city, a personal spend level, a transaction rate, an online share and 1-3 devices.
- 20% of cardholders take one trip (2-10 days, 35% abroad); their transactions move to the trip city.
- 20% of cardholders start using a new phone part-way through the window (legit "new device" events).
- 5% of cardholders make bursts of 3-10 tiny in-app / ride-share payments minutes apart (legit look-alikes of
  card testing).
- Merchants belong to a city and a merchant category (MCC); online merchants have no fixed location and the
  transaction location is the IP geolocation of the device.

Fraud patterns (labelled is_fraud=1, attack_type says which; campaigns start at a uniform random time)
- card_testing: 3-15 small ($0.50-$9.99) online digital-goods authorisations from a new device, spaced ~45 s,
  ~4 min or ~15 min apart, followed (65%) by 1-3 electronics/gift-card cash-out purchases ($200-$1,500) within
  hours. Half of campaigns use an IP in the cardholder's home city.
- account_takeover: 2-8 online purchases, ~20 min or ~3 h apart. 70% from a never-seen device, 30% from the
  cardholder's own device (session hijack). 40% are "stealthy": home-city IP, everyday categories and ticket sizes
  close to the cardholder's normal spend; the rest buy electronics/gift cards/apparel ($30+, median ~$320) from
  a far-away IP.
- geo_impossible: 2-6 counterfeit card-present purchases in a city >800 km from home (40% abroad), starting
  0.5-10 hours after a genuine transaction by the cardholder.
Fraudsters favour a small pool of weak-control "high-risk" merchants (30-35% of their volume), which is what the
merchant target encoding can learn.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from card_fraud.config import DAYS, MCC_NAMES, MCC_TABLE, SEED, WINDOW_START, Scale

# (name, lat, lon, relative size). The first 40 are domestic (US), the last 10 foreign.
CITIES: list[tuple[str, float, float, float]] = [
    ("New York", 40.71, -74.01, 8.3),
    ("Los Angeles", 34.05, -118.24, 3.9),
    ("Chicago", 41.88, -87.63, 2.7),
    ("Houston", 29.76, -95.37, 2.3),
    ("Phoenix", 33.45, -112.07, 1.6),
    ("Philadelphia", 39.95, -75.17, 1.6),
    ("San Antonio", 29.42, -98.49, 1.5),
    ("San Diego", 32.72, -117.16, 1.4),
    ("Dallas", 32.78, -96.80, 1.3),
    ("Austin", 30.27, -97.74, 1.0),
    ("Jacksonville", 30.33, -81.66, 0.95),
    ("San Jose", 37.34, -121.89, 1.0),
    ("Fort Worth", 32.76, -97.33, 0.92),
    ("Columbus", 39.96, -83.00, 0.9),
    ("Charlotte", 35.23, -80.84, 0.88),
    ("Indianapolis", 39.77, -86.16, 0.88),
    ("San Francisco", 37.77, -122.42, 0.87),
    ("Seattle", 47.61, -122.33, 0.75),
    ("Denver", 39.74, -104.99, 0.72),
    ("Nashville", 36.16, -86.78, 0.69),
    ("Washington", 38.91, -77.04, 0.69),
    ("Oklahoma City", 35.47, -97.52, 0.68),
    ("Boston", 42.36, -71.06, 0.65),
    ("El Paso", 31.76, -106.49, 0.68),
    ("Portland", 45.52, -122.68, 0.65),
    ("Las Vegas", 36.17, -115.14, 0.64),
    ("Detroit", 42.33, -83.05, 0.63),
    ("Memphis", 35.15, -90.05, 0.63),
    ("Louisville", 38.25, -85.76, 0.62),
    ("Baltimore", 39.29, -76.61, 0.58),
    ("Milwaukee", 43.04, -87.91, 0.57),
    ("Albuquerque", 35.08, -106.65, 0.56),
    ("Tucson", 32.22, -110.97, 0.54),
    ("Fresno", 36.74, -119.79, 0.54),
    ("Sacramento", 38.58, -121.49, 0.52),
    ("Atlanta", 33.75, -84.39, 0.5),
    ("Kansas City", 39.10, -94.58, 0.5),
    ("Miami", 25.76, -80.19, 0.45),
    ("Minneapolis", 44.98, -93.27, 0.43),
    ("Anchorage", 61.22, -149.90, 0.29),
    ("London", 51.51, -0.13, 1.0),
    ("Paris", 48.86, 2.35, 1.0),
    ("Lagos", 6.52, 3.38, 1.0),
    ("Sao Paulo", -23.55, -46.63, 1.0),
    ("Bucharest", 44.43, 26.10, 1.0),
    ("Manila", 14.60, 120.98, 1.0),
    ("Mumbai", 19.08, 72.88, 1.0),
    ("Mexico City", 19.43, -99.13, 1.0),
    ("Bangkok", 13.76, 100.50, 1.0),
    ("Istanbul", 41.01, 28.98, 1.0),
]
N_DOMESTIC = 40
FRAUD_TARGET_MCCS = ("electronics", "gift_cards", "digital_goods", "jewelry", "apparel")
NEW_DEVICE_BASE = 10_000_000  # device ids >= this are attacker / new-phone devices
HOURLY = np.array(
    [1, 0.6, 0.4, 0.3, 0.3, 0.5, 1.2, 2.5, 3.5, 4, 4.5, 5, 5.5, 5, 4.6, 4.5, 4.8, 5.3, 5.5, 5, 4.2, 3.3, 2.4, 1.6]
)


def haversine_np(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (numpy, vectorised)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=np.float64)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _merchants(rng: np.random.Generator, n_cards: int) -> dict[str, np.ndarray]:
    """Build in-person merchants per (city, mcc) and online merchants per mcc, as flat arrays + group offsets."""
    n_city, n_mcc = len(CITIES), len(MCC_NAMES)
    size = np.array([c[3] for c in CITIES])
    size[N_DOMESTIC:] = 0.6
    per = max(n_cards / 5.0, 60.0)  # merchant density scales with the portfolio
    counts = np.zeros(n_city * n_mcc + n_mcc, dtype=np.int64)
    for ci in range(n_city):
        for mi, m in enumerate(MCC_NAMES):
            w = MCC_TABLE[m][1]
            if w > 0:
                counts[ci * n_mcc + mi] = max(2, int(rng.poisson(per * w * size[ci] / size.sum())))
    for mi, m in enumerate(MCC_NAMES):
        w = MCC_TABLE[m][2]
        if w > 0:
            counts[n_city * n_mcc + mi] = max(5, int(rng.poisson(per * 0.15 * w)))
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    n = int(counts.sum())
    group = np.repeat(np.arange(len(counts)), counts)
    is_online = group >= n_city * n_mcc
    city = np.where(is_online, -1, group // n_mcc)
    mcc = np.where(is_online, group - n_city * n_mcc, group % n_mcc)
    clat = np.array([c[1] for c in CITIES])
    clon = np.array([c[2] for c in CITIES])
    lat = np.where(is_online, np.nan, clat[np.maximum(city, 0)] + rng.normal(0, 0.06, n))
    lon = np.where(is_online, np.nan, clon[np.maximum(city, 0)] + rng.normal(0, 0.06, n))
    target = np.isin(mcc, [MCC_NAMES.index(m) for m in FRAUD_TARGET_MCCS])
    high_risk = target & (rng.random(n) < 0.06)
    return {
        "counts": counts, "offsets": offsets, "city": city, "mcc": mcc, "lat": lat, "lon": lon,
        "is_online": is_online, "high_risk": high_risk, "n": n,
    }  # fmt: skip


def _pick_merchant(rng: np.random.Generator, m: dict, group: np.ndarray) -> np.ndarray:
    """Pick a merchant inside each requested group, skewed towards popular (low index) merchants."""
    size = m["counts"][group]
    if np.any(size == 0):
        raise ValueError("requested an empty merchant group")
    k = np.minimum((size * rng.random(len(group)) ** 1.6).astype(np.int64), size - 1)
    return m["offsets"][group] + k


def generate(scale: Scale, seed: int = SEED) -> tuple[pl.DataFrame, dict]:
    """Return (transactions, manifest). Fully determined by (scale, seed)."""
    rng = np.random.default_rng(seed)
    n_cards, n_city, n_mcc = scale.n_cards, len(CITIES), len(MCC_NAMES)
    secs = DAYS * 86_400
    clat = np.array([c[1] for c in CITIES])
    clon = np.array([c[2] for c in CITIES])
    city_w = np.array([c[3] for c in CITIES[:N_DOMESTIC]])
    city_w = city_w / city_w.sum()
    m = _merchants(rng, n_cards)

    # ---- cardholders -------------------------------------------------------------------------------
    home = rng.choice(N_DOMESTIC, n_cards, p=city_w)
    sigma = 0.6
    rate = scale.txn_rate_per_day * rng.lognormal(-0.5 * sigma**2, sigma, n_cards)
    spend = rng.lognormal(0.0, 0.4, n_cards)
    p_online = rng.beta(2, 5, n_cards)
    n_dev = 1 + rng.binomial(2, 0.3, n_cards)
    trip = rng.random(n_cards) < 0.20
    trip_start = rng.integers(0, secs - 11 * 86_400, n_cards)
    trip_len = rng.integers(2, 11, n_cards) * 86_400
    trip_abroad = rng.random(n_cards) < 0.35
    trip_city = np.where(
        trip_abroad,
        rng.integers(N_DOMESTIC, n_city, n_cards),
        (home + rng.integers(1, N_DOMESTIC, n_cards)) % N_DOMESTIC,
    )
    new_phone = rng.random(n_cards) < 0.20
    new_phone_at = rng.integers(0, secs, n_cards)

    # ---- legit transactions ------------------------------------------------------------------------
    n_txn = rng.poisson(rate * DAYS)
    card = np.repeat(np.arange(n_cards), n_txn)
    n = len(card)
    t = (
        rng.integers(0, DAYS, n) * 86_400
        + rng.choice(24, n, p=HOURLY / HOURLY.sum()) * 3_600
        + rng.integers(0, 3_600, n)
    )
    in_trip = trip[card] & (t >= trip_start[card]) & (t < trip_start[card] + trip_len[card])
    city = np.where(in_trip, trip_city[card], home[card])
    online = rng.random(n) < p_online[card]
    w_ip = np.array([MCC_TABLE[x][1] for x in MCC_NAMES])
    w_on = np.array([MCC_TABLE[x][2] for x in MCC_NAMES])
    mcc = np.where(online, rng.choice(n_mcc, n, p=w_on / w_on.sum()), rng.choice(n_mcc, n, p=w_ip / w_ip.sum()))
    group = np.where(online, n_city * n_mcc + mcc, city * n_mcc + mcc)
    merchant = _pick_merchant(rng, m, group)
    med = np.array([MCC_TABLE[x][0] for x in MCC_NAMES])
    amount = np.maximum(1.0, np.round(med[mcc] * spend[card] * rng.lognormal(0.0, 0.6, n), 2))
    lat = np.where(online, clat[city] + rng.normal(0, 0.08, n), m["lat"][merchant])
    lon = np.where(online, clon[city] + rng.normal(0, 0.08, n), m["lon"][merchant])
    dev_k = np.minimum(rng.geometric(0.6, n) - 1, n_dev[card] - 1)
    device = card.astype(np.int64) * 4 + dev_k + 1
    use_new = new_phone[card] & (t >= new_phone_at[card]) & (rng.random(n) < 0.7)
    device = np.where(use_new, card.astype(np.int64) * 4 + 4, device)
    device = np.where(online, device, 0)
    # legit micro-payment bursts (in-app / gaming purchases): look like card testing but are genuine
    burster = np.flatnonzero(rng.random(n_cards) < 0.05)
    n_bursts = rng.poisson(4, len(burster))
    b_card = np.repeat(burster, n_bursts)
    b_size = rng.integers(3, 11, len(b_card))
    bc = np.repeat(b_card, b_size)
    b_start = np.repeat(rng.integers(0, secs - 3_600, len(b_card)), b_size)
    b_off = np.concatenate([np.cumsum(rng.exponential(90.0, k)) for k in b_size]) if len(b_size) else np.empty(0)
    bt = (b_start + b_off).astype(np.int64)
    nb = len(bc)
    b_mcc = rng.choice([MCC_NAMES.index("digital_goods"), MCC_NAMES.index("ride_share")], nb, p=[0.85, 0.15])
    b_merch = _pick_merchant(rng, m, n_city * n_mcc + b_mcc)
    b_in_trip = trip[bc] & (bt >= trip_start[bc]) & (bt < trip_start[bc] + trip_len[bc])
    b_city = np.where(b_in_trip, trip_city[bc], home[bc])
    b_dev = bc.astype(np.int64) * 4 + 1
    t = np.concatenate([t, bt])
    card = np.concatenate([card, bc])
    merchant = np.concatenate([merchant, b_merch])
    mcc = np.concatenate([mcc, b_mcc])
    online = np.concatenate([online, np.ones(nb, bool)])
    amount = np.concatenate([amount, np.maximum(0.5, np.round(rng.lognormal(np.log(3.0), 0.6, nb), 2))])
    lat = np.concatenate([lat, clat[b_city] + rng.normal(0, 0.08, nb)])
    lon = np.concatenate([lon, clon[b_city] + rng.normal(0, 0.08, nb)])
    device = np.concatenate([device, b_dev])
    n = len(t)
    legit = {
        "t": t, "card": card, "merchant": merchant, "mcc": mcc, "online": online, "amount": amount,
        "lat": lat, "lon": lon, "device": device, "fraud": np.zeros(n, np.int8), "attack": np.zeros(n, np.int8),
    }  # fmt: skip

    # ---- fraud campaigns ---------------------------------------------------------------------------
    n_per_card = np.bincount(card, minlength=n_cards)
    order = np.lexsort((t, card))
    t_sorted, card_sorted = t[order], card[order]
    first_idx = np.concatenate([[0], np.cumsum(n_per_card)[:-1]])
    eligible = np.flatnonzero(n_txn >= 3)
    n_events = int(round(scale.fraud_events_per_card * n_cards))
    kinds = rng.choice(3, n_events, p=[0.35, 0.35, 0.30]) + 1  # 1 testing, 2 ATO, 3 geo
    victims = rng.choice(eligible, n_events, replace=False if n_events <= len(eligible) else True)
    hr_by_group: dict[int, np.ndarray] = {}
    for g in np.flatnonzero(m["counts"]):
        idx = np.arange(m["offsets"][g], m["offsets"][g] + m["counts"][g])
        hr_by_group[int(g)] = idx[m["high_risk"][idx]]

    def fraud_merchant(groups: np.ndarray, p_hr: float) -> np.ndarray:
        out = _pick_merchant(rng, m, groups)
        for i, g in enumerate(groups):
            pool = hr_by_group.get(int(g), np.empty(0, np.int64))
            if len(pool) and rng.random() < p_hr:
                out[i] = pool[rng.integers(len(pool))]
        return out

    def far_city(h: int, abroad_p: float) -> int:
        if rng.random() < abroad_p:
            return int(rng.integers(N_DOMESTIC, n_city))
        d = haversine_np(clat[h], clon[h], clat[:N_DOMESTIC], clon[:N_DOMESTIC])
        return int(rng.choice(np.flatnonzero(d > 800)))

    mcc_idx = {x: i for i, x in enumerate(MCC_NAMES)}
    med_amt = np.array([MCC_TABLE[x][0] for x in MCC_NAMES])
    rows: list[dict[str, np.ndarray]] = []
    next_dev = NEW_DEVICE_BASE + 0
    for e in range(n_events):
        c, kind = int(victims[e]), int(kinds[e])
        h = int(home[c])
        if kind == 1:  # card testing
            k = int(rng.integers(3, 16))
            t0 = int(rng.integers(3 * 86_400, secs - 86_400))
            tt = t0 + np.cumsum(rng.exponential(rng.choice([45.0, 240.0, 900.0]), k)).astype(np.int64)
            mm = np.full(k, mcc_idx["digital_goods"])
            amt = np.round(rng.uniform(0.5, 9.99, k), 2)
            if rng.random() < 0.65:
                kc = int(rng.integers(1, 4))
                tt = np.concatenate([tt, tt[-1] + np.cumsum(rng.uniform(1_800, 8 * 3_600, kc)).astype(np.int64)])
                mm = np.concatenate([mm, rng.choice([mcc_idx["electronics"], mcc_idx["gift_cards"]], kc)])
                amt = np.concatenate([amt, np.round(rng.uniform(200, 1_500, kc), 2)])
            k = len(tt)
            geo_city = far_city(h, 0.5) if rng.random() < 0.5 else h
            merch = fraud_merchant(n_city * n_mcc + mm, 0.35)
            online_f, dev = np.ones(k, bool), np.full(k, next_dev)
            next_dev += 1
            flat, flon = clat[geo_city] + rng.normal(0, 0.08, k), clon[geo_city] + rng.normal(0, 0.08, k)
        elif kind == 2:  # account takeover
            k = int(rng.integers(2, 9))
            t0 = int(rng.integers(3 * 86_400, secs - 86_400))
            tt = t0 + np.cumsum(rng.exponential(rng.choice([20 * 60.0, 3 * 3_600.0]), k)).astype(np.int64)
            mm = rng.choice(
                [mcc_idx["electronics"], mcc_idx["gift_cards"], mcc_idx["jewelry"], mcc_idx["apparel"],
                 mcc_idx["home_improvement"], mcc_idx["entertainment"]],
                k,
                p=[0.35, 0.2, 0.05, 0.2, 0.1, 0.1],
            )  # fmt: skip
            amt = np.round(np.maximum(30.0, rng.lognormal(np.log(320), 0.8, k)), 2)
            stealth = rng.random() < 0.4  # blends in: home-city IP, everyday categories and ticket sizes
            geo_city = h if stealth else far_city(h, 0.4)
            if stealth:
                mm = rng.choice([mcc_idx["apparel"], mcc_idx["restaurants"], mcc_idx["entertainment"],
                                 mcc_idx["electronics"], mcc_idx["digital_goods"]], k)  # fmt: skip
                amt = np.round(np.maximum(5.0, med_amt[mm] * spend[c] * rng.lognormal(0.4, 0.6, k)), 2)
            merch = fraud_merchant(n_city * n_mcc + mm, 0.35)
            online_f = np.ones(k, bool)
            if rng.random() < 0.3:  # session hijack / malware on the cardholder's own device
                dev = np.full(k, c * 4 + 1, dtype=np.int64)
            else:
                dev = np.full(k, next_dev)
                next_dev += 1
            flat, flon = clat[geo_city] + rng.normal(0, 0.08, k), clon[geo_city] + rng.normal(0, 0.08, k)
        else:  # geo-impossible counterfeit card-present
            k = int(rng.integers(2, 7))
            j = int(rng.integers(0, n_per_card[c]))
            anchor = int(t_sorted[first_idx[c] + j])
            assert card_sorted[first_idx[c] + j] == c
            t0 = anchor + int(rng.uniform(0.5, 10.0) * 3_600)
            tt = t0 + np.concatenate([[0], np.cumsum(rng.exponential(2 * 3_600.0, k - 1))]).astype(np.int64)
            mm = rng.choice(
                [mcc_idx["electronics"], mcc_idx["jewelry"], mcc_idx["apparel"], mcc_idx["gift_cards"],
                 mcc_idx["grocery"]],
                k,
                p=[0.35, 0.2, 0.2, 0.15, 0.1],
            )  # fmt: skip
            geo_city = far_city(h, 0.4)
            merch = fraud_merchant(geo_city * n_mcc + mm, 0.3)
            amt = np.round(np.maximum(10.0, rng.lognormal(np.log(140), 0.9, k)), 2)
            online_f, dev = np.zeros(k, bool), np.zeros(k, np.int64)
            flat, flon = m["lat"][merch], m["lon"][merch]
        keep = tt < secs
        rows.append({
            "t": tt[keep], "card": np.full(keep.sum(), c), "merchant": merch[keep], "mcc": mm[keep],
            "online": online_f[keep], "amount": amt[keep], "lat": flat[keep], "lon": flon[keep],
            "device": dev[keep], "fraud": np.ones(keep.sum(), np.int8), "attack": np.full(keep.sum(), kind, np.int8),
        })  # fmt: skip

    allrows = {key: np.concatenate([legit[key]] + [r[key] for r in rows]) for key in legit}
    attack_names = np.array(["none", "card_testing", "account_takeover", "geo_impossible"])
    mcc_names = np.array(MCC_NAMES)
    df = _to_frame(allrows, attack_names, mcc_names)
    manifest = _manifest(df, scale, seed, n_events, m["n"])
    return df, manifest


def _to_frame(allrows: dict, attack_names: np.ndarray, mcc_names: np.ndarray) -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "t_s": allrows["t"].astype(np.int64),
            "card_id": allrows["card"].astype(np.int32),
            "merchant_id": allrows["merchant"].astype(np.int32),
            "mcc": mcc_names[allrows["mcc"]],
            "channel": np.where(allrows["online"], "online", "in_person"),
            "amount": allrows["amount"].astype(np.float64),
            "lat": allrows["lat"].astype(np.float32),
            "lon": allrows["lon"].astype(np.float32),
            "device_id": allrows["device"].astype(np.int64),
            "is_fraud": allrows["fraud"].astype(np.int8),
            "attack_type": attack_names[allrows["attack"]],
        }
    )
    start = pl.lit(WINDOW_START).cast(pl.Datetime("ms"))
    df = (
        df.with_columns(ts=start + pl.duration(seconds=pl.col("t_s")))
        .sort(["ts", "card_id", "merchant_id", "amount"])
        .with_row_index("txn_id")
        .with_columns(pl.col("txn_id").cast(pl.Int64))
        .select(
            "txn_id",
            "ts",
            "card_id",
            "merchant_id",
            "mcc",
            "channel",
            "amount",
            "lat",
            "lon",
            "device_id",
            "is_fraud",
            "attack_type",
        )  # fmt: skip
    )
    return df


def _manifest(df: pl.DataFrame, scale: Scale, seed: int, n_events: int, n_merchants: int) -> dict:
    by_attack = df.group_by("attack_type").agg(pl.len().alias("n"), pl.col("amount").sum().alias("usd")).sort("n")
    return {
        "generator": "card_fraud.generate (SYNTHETIC)",
        "scale": asdict(scale),
        "seed": seed,
        "window": [str(df["ts"].min()), str(df["ts"].max())],
        "transactions": df.height,
        "cardholders_with_txns": int(df["card_id"].n_unique()),
        "merchants": n_merchants,
        "fraud_campaigns": n_events,
        "fraud_transactions": int(df["is_fraud"].sum()),
        "fraud_rate": float(df["is_fraud"].mean()),
        "by_attack_type": {
            r["attack_type"]: {"transactions": r["n"], "amount_usd": round(r["usd"], 2)} for r in by_attack.to_dicts()
        },
    }


def write(df: pl.DataFrame, manifest: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "transactions.parquet"
    df.write_parquet(path, compression="zstd", compression_level=10, statistics=False)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return path
