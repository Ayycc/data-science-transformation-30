"""Data acquisition: public IBM AML dataset if reachable, otherwise the seeded SYNTHETIC generator."""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import IBM_AML_URL, IBM_LOCAL_CSV, SimConfig
from .simulate import SimulatedBank, simulate


@dataclass
class DataBundle:
    bank: SimulatedBank
    source: str  # "ibm_public" or "synthetic"
    note: str


def probe_ibm(url: str = IBM_AML_URL, timeout: float = 15.0) -> tuple[bool, str]:
    """Check whether the public IBM AML archive is downloadable from this network."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:  # proxy refusal, DNS, timeout
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def load_ibm_csv(path: Path) -> SimulatedBank:
    """Map the IBM AML transaction CSV (HI-/LI- *_Trans.csv) onto this project's schema.

    The IBM file has no per-account typology labels in the transaction table, so typology
    is recorded as ``"unknown"`` for laundering accounts and per-typology recall is skipped.
    """
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    src_key = raw["From Bank"].astype(str) + ":" + raw["Account"].astype(str)
    dst_key = raw["To Bank"].astype(str) + ":" + raw["Account.1"].astype(str)
    keys, inv = np.unique(np.concatenate([src_key.to_numpy(), dst_key.to_numpy()]), return_inverse=True)
    n = len(raw)
    ts = pd.to_datetime(raw["Timestamp"])
    t = (ts - ts.min()).dt.total_seconds().to_numpy()
    lab = raw["Is Laundering"].astype(int).to_numpy().astype(bool)
    tx = pd.DataFrame(
        {
            "tx_id": np.arange(n, dtype=np.int64),
            "src": inv[:n].astype(np.int64),
            "dst": inv[n:].astype(np.int64),
            "ts": t,
            "amount": raw["Amount Paid"].astype(float).to_numpy(),
            "channel": raw["Payment Format"].astype(str).to_numpy(),
            "typology": np.where(lab, "unknown", "none"),
            "instance_id": -1,
            "is_laundering": lab,
        }
    ).sort_values("ts", kind="stable")
    bad = np.zeros(len(keys), dtype=bool)
    bad[tx.loc[tx["is_laundering"], "src"].to_numpy()] = True
    bad[tx.loc[tx["is_laundering"], "dst"].to_numpy()] = True
    acc = pd.DataFrame(
        {
            "account_id": np.arange(len(keys), dtype=np.int64),
            "role": "personal",
            "region": 0,
            "region_risk": 0,
            "is_payroll": False,
            "open_day": 0.0,
            "typology": np.where(bad, "unknown", "none"),
            "instance_id": np.where(bad, np.arange(len(keys)), -1),
            "is_laundering": bad,
        }
    )
    inst = pd.DataFrame(columns=["instance_id", "typology", "account_id", "scheme_role", "slow"])
    return SimulatedBank(accounts=acc, transactions=tx.reset_index(drop=True), instances=inst)


def acquire(cfg: SimConfig, try_public: bool = True) -> DataBundle:
    """Public IBM data when a local copy exists or it can be fetched; SYNTHETIC fallback otherwise."""
    if IBM_LOCAL_CSV.exists():
        return DataBundle(load_ibm_csv(IBM_LOCAL_CSV), "ibm_public", f"local copy {IBM_LOCAL_CSV.name}")
    note = "public probe skipped"
    if try_public:
        ok, note = probe_ibm()
        if ok:
            note += " (archive reachable but requires Kaggle credentials to unpack; using synthetic generator)"
    return DataBundle(simulate(cfg), "synthetic", f"IBM AML probe: {note}; fell back to seeded SYNTHETIC generator (seed {cfg.seed})")
