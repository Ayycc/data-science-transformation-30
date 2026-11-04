"""Traditional statistical disclosure control: k-anonymity by generalisation plus local suppression."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import QUASI_IDENTIFIERS

AGE_BANDS = [17, 25, 35, 45, 55, 65, 75, 91]


def k_anonymity_report(df: pd.DataFrame, qis: list[str] | None = None, k: int = 5) -> dict:
    qis = qis or QUASI_IDENTIFIERS
    sizes = df.groupby(qis, observed=True).size()
    per_row = df.groupby(qis, observed=True)[qis[0]].transform("size")
    return {
        "quasi_identifiers": qis,
        "equivalence_classes": int(len(sizes)),
        "min_k": int(sizes.min()),
        "share_records_unique": float((per_row == 1).mean()),
        f"share_records_k_lt_{k}": float((per_row < k).mean()),
    }


def generalise(df: pd.DataFrame) -> pd.DataFrame:
    """Age to 10-year bands (released as the band midpoint) and every non-US region to ``Other``."""
    out = df.copy()
    edges = np.asarray(AGE_BANDS)
    band = np.clip(np.searchsorted(edges, out["age"].to_numpy(), side="right") - 1, 0, len(edges) - 2)
    out["age"] = ((edges[band] + edges[band + 1] - 1) // 2).astype(int)
    out["native_region"] = np.where(out["native_region"] == "United-States", "United-States", "Other")
    return out


def k_anonymise(df: pd.DataFrame, k: int = 5, qis: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Generalise, then suppress (drop) records in equivalence classes smaller than k."""
    qis = qis or QUASI_IDENTIFIERS
    g = generalise(df)
    size = g.groupby(qis, observed=True)[qis[0]].transform("size")
    keep = size >= k
    rel = g[keep].reset_index(drop=True)
    return rel, {
        "k": k,
        "suppressed_records": int((~keep).sum()),
        "suppressed_share": float((~keep).mean()),
        "after": k_anonymity_report(rel, qis, k),
    }
