"""Segment (heterogeneous treatment effect) slices with Benjamini-Hochberg correction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from expengine.cuped import cuped_estimate


def segment_effects(
    df: pd.DataFrame,
    metrics: dict[str, str],
    segment_cols: Sequence[str],
    treat_col: str = "treat",
    alpha: float = 0.05,
    min_per_arm: int = 200,
) -> pd.DataFrame:
    """CUPED effect per (segment column, level, metric); BH-adjusted across all slices.

    ``metrics`` maps metric column -> pre-period covariate column.
    """
    rows = []
    for col in segment_cols:
        for level, g in df.groupby(col, observed=True):
            t = g[treat_col].to_numpy(bool)
            if t.sum() < min_per_arm or (~t).sum() < min_per_arm:
                continue
            for metric, cov in metrics.items():
                est = cuped_estimate(g[metric].to_numpy(), g[cov].to_numpy(), t, alpha, metric)
                rows.append(
                    {
                        "segment": col,
                        "level": str(level),
                        "metric": metric,
                        "n": len(g),
                        "rel_lift": est.rel_lift,
                        "diff": est.diff,
                        "se": est.se,
                        "p_value": est.p_value,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    reject, p_adj, _, _ = multipletests(out["p_value"].to_numpy(), alpha=alpha, method="fdr_bh")
    out["p_bh"] = p_adj
    out["significant_bh"] = reject
    out["significant_unadjusted"] = out["p_value"].to_numpy() < alpha
    return out


def interaction_test(d1: float, se1: float, d2: float, se2: float) -> float:
    """Two-sided p-value that two independent segment effects differ."""
    from scipy import stats

    z = (d1 - d2) / np.sqrt(se1**2 + se2**2)
    return float(2 * stats.norm.sf(abs(z)))
