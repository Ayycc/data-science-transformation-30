"""pandas baseline: the same features as ``features_polars``, written idiomatically in pandas.

Used only to benchmark Polars and to cross-check that both implementations agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from card_fraud.config import MCC_TO_CODE, N_TRAIN_FOLDS
from card_fraud.features_polars import SMALL_AMOUNT, TE_ALPHA
from card_fraud.generate import haversine_np


def card_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.sort_values(["card_id", "ts", "txn_id"], kind="mergesort").reset_index(drop=True)
    df["lat"] = df["lat"].astype("float64")
    df["lon"] = df["lon"].astype("float64")
    df["_one"] = 1.0
    df["_small"] = (df["amount"] < SMALL_AMOUNT).astype("float64")
    df["_ts_s"] = (df["ts"] - pd.Timestamp("1970-01-01")).dt.total_seconds()
    df["fold"] = df["ts"].dt.month.astype("int32")

    g = df.groupby("card_id", sort=True)
    for w in ("1h", "24h", "7d", "30d"):
        r = g.rolling(w, on="ts", closed="left")[["_one", "amount", "_small"]].sum()
        r = r.reset_index(drop=True).fillna(0.0)
        df[f"_cnt_{w}"] = r["_one"].to_numpy()
        df[f"_sum_{w}"] = r["amount"].to_numpy()
        df[f"_small_{w}"] = r["_small"].to_numpy()
    df["card_txn_count_1h"] = df["_cnt_1h"]
    df["card_txn_count_24h"] = df["_cnt_24h"]
    df["card_txn_count_7d"] = df["_cnt_7d"]
    df["card_amount_sum_1h"] = df["_sum_1h"]
    df["card_amount_sum_24h"] = df["_sum_24h"]
    df["card_small_txn_count_1h"] = df["_small_1h"]
    df["card_amount_mean_30d"] = (df["_sum_30d"] / df["_cnt_30d"]).where(df["_cnt_30d"] > 0)

    prev_ts = g["_ts_s"].shift(1)
    prev_lat = g["lat"].shift(1)
    prev_lon = g["lon"].shift(1)
    df["secs_since_prev"] = df["_ts_s"] - prev_ts
    df["dist_prev_km"] = haversine_np(prev_lat, prev_lon, df["lat"], df["lon"])
    idx = g.cumcount().astype("float64")
    df["card_txn_index"] = idx
    lat_c = (g["lat"].cumsum() - df["lat"]) / idx
    lon_c = (g["lon"].cumsum() - df["lon"]) / idx
    df["dist_from_centroid_km"] = pd.Series(haversine_np(lat_c, lon_c, df["lat"], df["lon"])).where(idx > 0)
    df["card_age_days"] = (df["_ts_s"] - g["_ts_s"].transform("first")) / 86_400.0
    online = df["channel"] == "online"
    df["is_online"] = online.astype("float64")
    dev_prior = df.groupby(["card_id", "device_id"]).cumcount()
    df["is_new_device"] = (online & (dev_prior == 0)).astype("float64")
    df["device_prior_txn_count"] = dev_prior.astype("float64").where(online)
    df["is_new_merchant_for_card"] = (df.groupby(["card_id", "merchant_id"]).cumcount() == 0).astype("float64")
    df["hour"] = df["ts"].dt.hour.astype("float64")
    df["dow"] = (df["ts"].dt.dayofweek + 1).astype("float64")
    df["mcc_code"] = df["mcc"].map(MCC_TO_CODE).astype("int32")
    df["amount_to_mean_30d"] = df["amount"] / df["card_amount_mean_30d"]
    df["speed_kmh"] = df["dist_prev_km"] / (np.maximum(df["secs_since_prev"], 60.0) / 3_600.0)
    return df


def merchant_target_encoding(df: pd.DataFrame, n_train_folds: int = N_TRAIN_FOLDS) -> pd.DataFrame:
    stats = df.groupby(["merchant_id", "fold"]).agg(_n=("is_fraud", "size"), _f=("is_fraud", "sum")).reset_index()
    enc_parts, priors = [], {}
    for f in range(1, 13):
        hist = stats[stats["fold"] < min(f, n_train_folds + 1)]
        e = hist.groupby("merchant_id")[["_n", "_f"]].sum().reset_index()
        e["fold"] = f
        enc_parts.append(e)
        priors[f] = hist["_f"].sum() / hist["_n"].sum() if len(hist) else np.nan
    enc = pd.concat(enc_parts, ignore_index=True).astype({"fold": "int32", "_n": "float64", "_f": "float64"})
    out = df.merge(enc, on=["merchant_id", "fold"], how="left")
    prior = out["fold"].map(priors).astype("float64")
    n = out["_n"].fillna(0.0)
    out["merchant_hist_txn_count"] = n.where(prior.notna())
    out["merchant_risk_te"] = (out["_f"].fillna(0.0) + TE_ALPHA * prior) / (n + TE_ALPHA)
    return out


def build_features(raw: pd.DataFrame, n_train_folds: int = N_TRAIN_FOLDS) -> pd.DataFrame:
    out = merchant_target_encoding(card_features(raw), n_train_folds)
    out = out.drop(columns=[c for c in out.columns if c.startswith("_")])
    return out.sort_values("txn_id").reset_index(drop=True)
