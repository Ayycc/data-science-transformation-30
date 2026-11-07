"""Seeded SYNTHETIC insurer extracts: policy versions, premium transactions, claim valuations.

Everything here is synthetic. The generator plants the data conditions that trigger
the legacy quirks: endorsements that change a policy's region, a backfill job that
re-loaded some versions out of order, inconsistently formatted LOB codes, late
bookings and missing case reserves.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LOBS, QUARTERS, REGIONS, SEED, to_sas

LOB_MIX = np.array([0.14, 0.30, 0.08, 0.38, 0.10])
BASE_PREMIUM = {"COMMERCIAL": 380_000, "HOME": 95_000, "LIABILITY": 240_000, "MOTOR": 72_000, "TRAVEL": 18_000}
CLAIM_FREQ = {"COMMERCIAL": 0.16, "HOME": 0.10, "LIABILITY": 0.07, "MOTOR": 0.18, "TRAVEL": 0.12, "PET": 0.30}
SEVERITY = {"COMMERCIAL": 900_000, "HOME": 420_000, "LIABILITY": 1_400_000, "MOTOR": 300_000, "TRAVEL": 60_000, "PET": 45_000}


@dataclass
class Extracts:
    policy_versions: pd.DataFrame
    premium_txns: pd.DataFrame
    claim_valuations: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {"policy_versions": self.policy_versions, "premium_txns": self.premium_txns, "claim_valuations": self.claim_valuations}


def _raw_code(lob: str, rng: np.random.Generator) -> str:
    u = rng.random()
    if u < 0.035:
        return lob.title()  # "Motor" from the broker portal
    if u < 0.06:
        return lob.lower() + " "  # "home " from the old mainframe
    return lob


def generate(n_policies: int = 30_000, seed: int = SEED) -> Extracts:
    """Generate SYNTHETIC extracts for 2023-2025 policy inception and 2024-2025 reporting."""
    rng = np.random.default_rng(seed)
    first_day = to_sas(dt.date(2023, 1, 1))
    last_inception = to_sas(dt.date(2025, 12, 15))
    last_day = QUARTERS[-1].end

    versions, txns, vals = [], [], []
    txn_id = 0
    claim_id = 0
    for pid in range(1, n_policies + 1):
        is_pet = rng.random() < 0.015
        lob = "PET" if is_pet else LOBS[rng.choice(5, p=LOB_MIX)]
        region = REGIONS[rng.integers(4)]
        start = int(rng.integers(first_day, last_inception))
        term = 365 if lob != "TRAVEL" else int(rng.choice([30, 90, 180, 365]))
        expiry = start + term  # exclusive
        prem = int(BASE_PREMIUM.get(lob, 30_000) * rng.lognormal(0.0, 0.35)) + 1
        code = lob if is_pet else _raw_code(lob, rng)
        versions.append([pid, 1, start, code, region, start])
        book_lag = int(rng.integers(0, 8)) if rng.random() > 0.03 else int(rng.integers(20, 75))
        txn_id += 1
        txns.append([txn_id, pid, "NB", start, expiry, prem, start + book_lag])

        # Endorsements: change region (customer moved) and add premium for the rest of the term.
        n_end = rng.poisson(0.22) if term > 60 else 0
        v = 1
        cur_region = region
        for e_day in sorted(rng.integers(start + 10, expiry - 5, size=min(n_end, 3))):
            v += 1
            if rng.random() < 0.6:
                cur_region = REGIONS[(REGIONS.index(cur_region) + int(rng.integers(1, 4))) % 4]
            code_v = lob if is_pet else _raw_code(lob, rng)
            versions.append([pid, v, int(e_day), code_v, cur_region, int(e_day)])
            add = int(prem * rng.uniform(0.03, 0.25) * (expiry - e_day) / term) + 1
            lag = int(rng.integers(0, 8)) if rng.random() > 0.05 else int(rng.integers(20, 75))
            txn_id += 1
            txns.append([txn_id, pid, "END", int(e_day), expiry, add, int(e_day) + lag])

        # Claims with quarterly valuations from the report quarter to the last quarter.
        exp_claims = CLAIM_FREQ[lob] * term / 365
        lo = max(start, QUARTERS[0].start)
        hi = min(expiry, last_day)
        if hi <= lo:
            continue
        for _ in range(rng.poisson(exp_claims)):
            loss = int(rng.integers(start, expiry))
            if loss < lo or loss > hi:
                continue
            report = loss + int(rng.exponential(20 if lob != "LIABILITY" else 70))
            if report > last_day:
                continue
            claim_id += 1
            ultimate = max(1_000, int(SEVERITY[lob] * rng.lognormal(0.0, 0.9)))
            q0 = next(q.idx for q in QUARTERS if q.start <= report <= q.end)
            settle = q0 + int(rng.integers(0, 5 if lob != "LIABILITY" else 9))
            paid = 0
            for q in QUARTERS[q0 - 1 :]:
                if q.idx >= settle:
                    paid = ultimate
                    reserve = 0
                else:
                    paid = min(ultimate, paid + int(ultimate * rng.uniform(0.05, 0.4)))
                    reserve = max(0, int((ultimate - paid) * rng.uniform(0.8, 1.3)))
                reserve_val = None if rng.random() < 0.02 else reserve
                vals.append([claim_id, pid, loss, report, q.idx, paid, reserve_val])

    pv = pd.DataFrame(versions, columns=["policy_id", "version_no", "version_eff", "lob_code_raw", "region", "load_date"])
    # Merge-order quirk: a migration backfill re-wrote ~15% of multi-version policies so
    # that their later versions sit *ahead of* version 1 in the extract file and carry
    # version 1's load date. load_seq is the physical file order.
    multi = pv.groupby("policy_id")["version_no"].transform("max") > 1
    backfilled = set(pv.loc[multi, "policy_id"].drop_duplicates().sample(frac=0.15, random_state=seed % 2**32).tolist())
    moved = pv["policy_id"].isin(backfilled) & (pv["version_no"] > 1)
    pv.loc[moved, "load_date"] = pv.groupby("policy_id")["load_date"].transform("min")[moved]
    pv = pv.assign(_k=np.where(moved, -pv["version_no"], pv["version_no"]))
    pv = pv.sort_values(["load_date", "policy_id", "_k"], kind="stable").drop(columns="_k").reset_index(drop=True)
    pv.insert(0, "load_seq", np.arange(1, len(pv) + 1))

    pt = pd.DataFrame(txns, columns=["txn_id", "policy_id", "txn_type", "txn_eff", "txn_exp", "premium_cents", "booked_date"])
    cv = pd.DataFrame(vals, columns=["claim_id", "policy_id", "loss_date", "report_date", "valuation_q", "cum_paid_cents", "case_reserve_cents"])
    cv["case_reserve_cents"] = cv["case_reserve_cents"].astype("Int64")
    for df in (pv, pt, cv):
        for c in df.columns:
            if df[c].dtype == np.int64 or str(df[c].dtype) == "int64":
                df[c] = df[c].astype("int64")
    return Extracts(pv, pt, cv)


def as_of_close(ex: Extracts, close_q: int) -> Extracts:
    """What the source systems held at the close of quarter `close_q` (booked/loaded on or before quarter end)."""
    end = QUARTERS[close_q - 1].end
    pv = ex.policy_versions[ex.policy_versions["load_date"] <= end].reset_index(drop=True)
    pt = ex.premium_txns[ex.premium_txns["booked_date"] <= end].reset_index(drop=True)
    cv = ex.claim_valuations[ex.claim_valuations["valuation_q"] <= close_q].reset_index(drop=True)
    return Extracts(pv, pt, cv)
