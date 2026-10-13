"""SHAP explanations and analyst-facing reason codes."""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import polars as pl
import shap

from card_fraud.config import FEATURES, MCC_NAMES

REASON_TEXT: dict[str, str] = {
    "amount": "transaction amount",
    "hour": "time of day",
    "dow": "day of week",
    "is_online": "card-not-present (online) channel",
    "mcc_code": "merchant category",
    "card_txn_count_1h": "card transactions in the last hour",
    "card_txn_count_24h": "card transactions in the last 24h",
    "card_txn_count_7d": "card transactions in the last 7 days",
    "card_amount_sum_1h": "card spend in the last hour",
    "card_amount_sum_24h": "card spend in the last 24h",
    "card_amount_mean_30d": "card's 30-day average ticket",
    "amount_to_mean_30d": "amount vs card's 30-day average",
    "card_small_txn_count_1h": "small (<$5) authorisations in the last hour",
    "secs_since_prev": "time since the card's previous transaction",
    "dist_prev_km": "distance from the previous transaction",
    "speed_kmh": "implied travel speed since the previous transaction",
    "dist_from_centroid_km": "distance from the card's usual location",
    "is_new_device": "first use of this device on the card",
    "device_prior_txn_count": "prior transactions on this device",
    "is_new_merchant_for_card": "first purchase at this merchant",
    "card_age_days": "days since the card was first seen",
    "card_txn_index": "number of prior card transactions",
    "merchant_risk_te": "merchant's historical fraud rate",
    "merchant_hist_txn_count": "merchant's historical volume",
}


def shap_values(booster: lgb.Booster, x: np.ndarray) -> tuple[np.ndarray, float]:
    """SHAP values (log-odds) from shap.TreeExplainer, plus the expected value."""
    explainer = shap.TreeExplainer(booster)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # notice about list output format for LightGBM
        sv = explainer.shap_values(x)
    if isinstance(sv, list):  # older shap returns [neg, pos]
        sv = sv[1]
    ev = explainer.expected_value
    ev = float(np.atleast_1d(ev)[-1])
    return np.asarray(sv), ev


def global_importance(sv: np.ndarray) -> pl.DataFrame:
    return (
        pl.DataFrame({"feature": list(FEATURES), "mean_abs_shap": np.abs(sv).mean(axis=0)})
        .sort("mean_abs_shap", descending=True)
        .with_columns(share=pl.col("mean_abs_shap") / pl.col("mean_abs_shap").sum())
    )


def _fmt(feature: str, value: float) -> str:
    if np.isnan(value):
        return "n/a"
    if feature in ("amount", "card_amount_sum_1h", "card_amount_sum_24h", "card_amount_mean_30d"):
        return f"${value:,.2f}"
    if feature in ("dist_prev_km", "dist_from_centroid_km"):
        return f"{value:,.0f} km"
    if feature == "speed_kmh":
        return f"{value:,.0f} km/h"
    if feature == "secs_since_prev":
        return f"{value / 60:,.1f} min"
    if feature == "merchant_risk_te":
        return f"{value:.2%}"
    if feature == "amount_to_mean_30d":
        return f"{value:.1f}x"
    if feature == "mcc_code":
        return MCC_NAMES[int(value)]
    return f"{value:g}"


def reason_codes(sv: np.ndarray, x: np.ndarray, k: int = 3) -> list[list[dict]]:
    """Top-k features pushing each row towards fraud (largest positive SHAP values)."""
    out = []
    top = np.argsort(-sv, axis=1)[:, :k]
    for i in range(sv.shape[0]):
        row = []
        for j in top[i]:
            f = FEATURES[j]
            row.append(
                {
                    "feature": f,
                    "value": _fmt(f, float(x[i, j])),
                    "shap": float(sv[i, j]),
                    "text": f"{REASON_TEXT[f]} = {_fmt(f, float(x[i, j]))}",
                }
            )
        out.append(row)
    return out
