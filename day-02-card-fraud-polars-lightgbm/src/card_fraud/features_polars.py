"""Polars LazyFrame feature pipeline.

Every feature for a transaction uses only that transaction's own authorisation-time attributes plus
transactions (and fraud labels) that happened strictly before it:
- rolling windows are ``closed="left"``: the window for time t is [t - w, t), so the current and any later
  transaction are excluded;
- "previous transaction" features use ``shift(1)`` over the card, ordered by time;
- merchant target encoding is expanding-window out-of-fold: rows in month f see labels from months < f only,
  capped at the last training month, so validation/test rows only ever see training labels.
"""

from __future__ import annotations

import polars as pl

from card_fraud.config import MCC_TO_CODE, N_TRAIN_FOLDS

EARTH_KM = 6371.0
TE_ALPHA = 200.0  # m-estimate smoothing strength for the merchant target encoding
SMALL_AMOUNT = 5.0
WINDOWS = {"1h": "1h", "24h": "24h", "7d": "7d", "30d": "30d"}


def _haversine(lat1: pl.Expr, lon1: pl.Expr, lat2: pl.Expr, lon2: pl.Expr) -> pl.Expr:
    rad = 0.017453292519943295
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = (dlat / 2).sin() ** 2 + (lat1 * rad).cos() * (lat2 * rad).cos() * (dlon / 2).sin() ** 2
    return 2 * EARTH_KM * a.clip(0.0, 1.0).sqrt().arcsin()


def _roll(col: str, window: str) -> pl.Expr:
    return pl.col(col).rolling_sum_by("ts", window_size=window, closed="left").over("card_id")


def card_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Velocity, recency, novelty and geo-velocity features per card (lazy)."""
    by_card = ["card_id"]
    return (
        lf.sort(["card_id", "ts", "txn_id"])
        .with_columns(
            pl.col("lat").cast(pl.Float64),
            pl.col("lon").cast(pl.Float64),
            _one=pl.lit(1.0),
            _small=(pl.col("amount") < SMALL_AMOUNT).cast(pl.Float64),
            _ts_s=pl.col("ts").dt.epoch("s").cast(pl.Float64),
            fold=pl.col("ts").dt.month().cast(pl.Int32),
        )
        .with_columns(
            card_txn_count_1h=_roll("_one", "1h"),
            card_txn_count_24h=_roll("_one", "24h"),
            card_txn_count_7d=_roll("_one", "7d"),
            _cnt_30d=_roll("_one", "30d"),
            card_amount_sum_1h=_roll("amount", "1h"),
            card_amount_sum_24h=_roll("amount", "24h"),
            _sum_30d=_roll("amount", "30d"),
            card_small_txn_count_1h=_roll("_small", "1h"),
            _prev_ts=pl.col("_ts_s").shift(1).over(by_card),
            _prev_lat=pl.col("lat").shift(1).over(by_card),
            _prev_lon=pl.col("lon").shift(1).over(by_card),
            _first_ts=pl.col("_ts_s").first().over(by_card),
            card_txn_index=pl.int_range(pl.len(), dtype=pl.Int64).over(by_card).cast(pl.Float64),
            _lat_prior_sum=(pl.col("lat").cum_sum() - pl.col("lat")).over(by_card),
            _lon_prior_sum=(pl.col("lon").cum_sum() - pl.col("lon")).over(by_card),
            _dev_prior=pl.int_range(pl.len(), dtype=pl.Int64).over(["card_id", "device_id"]),
            _merch_prior=pl.int_range(pl.len(), dtype=pl.Int64).over(["card_id", "merchant_id"]),
        )
        .with_columns(
            card_amount_mean_30d=pl.when(pl.col("_cnt_30d") > 0).then(pl.col("_sum_30d") / pl.col("_cnt_30d")),
            secs_since_prev=pl.col("_ts_s") - pl.col("_prev_ts"),
            dist_prev_km=_haversine(pl.col("_prev_lat"), pl.col("_prev_lon"), pl.col("lat"), pl.col("lon")),
            dist_from_centroid_km=pl.when(pl.col("card_txn_index") > 0).then(
                _haversine(
                    pl.col("_lat_prior_sum") / pl.col("card_txn_index"),
                    pl.col("_lon_prior_sum") / pl.col("card_txn_index"),
                    pl.col("lat"),
                    pl.col("lon"),
                )
            ),
            card_age_days=(pl.col("_ts_s") - pl.col("_first_ts")) / 86_400.0,
            is_online=(pl.col("channel") == "online").cast(pl.Float64),
            is_new_device=((pl.col("channel") == "online") & (pl.col("_dev_prior") == 0)).cast(pl.Float64),
            device_prior_txn_count=pl.when(pl.col("channel") == "online").then(pl.col("_dev_prior").cast(pl.Float64)),
            is_new_merchant_for_card=(pl.col("_merch_prior") == 0).cast(pl.Float64),
            hour=pl.col("ts").dt.hour().cast(pl.Float64),
            dow=pl.col("ts").dt.weekday().cast(pl.Float64),
            mcc_code=pl.col("mcc").replace_strict(MCC_TO_CODE, return_dtype=pl.Int32),
        )
        .with_columns(
            amount_to_mean_30d=pl.col("amount") / pl.col("card_amount_mean_30d"),
            speed_kmh=pl.col("dist_prev_km") / (pl.max_horizontal(pl.col("secs_since_prev"), 60.0) / 3_600.0),
        )
    )


def merchant_target_encoding(lf: pl.LazyFrame, n_train_folds: int = N_TRAIN_FOLDS) -> pl.LazyFrame:
    """Expanding-window out-of-fold merchant fraud-rate encoding.

    A row in fold f gets the smoothed fraud rate of its merchant over folds g < min(f, n_train_folds + 1).
    Fold 1 has no history (null). Validation/test folds only see training-fold labels.
    """
    folds = range(1, 13)  # calendar months; fixed so the query plan does not depend on the data
    stats = lf.group_by(["merchant_id", "fold"]).agg(_n=pl.len().cast(pl.Float64), _f=pl.col("is_fraud").sum())
    enc_parts, prior_parts = [], []
    for f in folds:
        hist = stats.filter(pl.col("fold") < min(f, n_train_folds + 1))
        fold = pl.lit(f, dtype=pl.Int32)
        enc_parts.append(
            hist.group_by("merchant_id")
            .agg(pl.col("_n").sum(), pl.col("_f").sum().cast(pl.Float64))
            .with_columns(fold=fold)
        )
        prior_parts.append(
            hist.select(
                pl.when(pl.col("_n").sum() > 0).then(pl.col("_f").sum() / pl.col("_n").sum()).alias("_prior_fold")
            ).with_columns(fold=fold)
        )
    enc, priors = pl.concat(enc_parts), pl.concat(prior_parts)
    return (
        lf.join(enc, on=["merchant_id", "fold"], how="left")
        .join(priors, on="fold", how="left")
        .with_columns(
            merchant_hist_txn_count=pl.when(pl.col("_prior_fold").is_not_null()).then(pl.col("_n").fill_null(0.0)),
            merchant_risk_te=(pl.col("_f").fill_null(0.0) + TE_ALPHA * pl.col("_prior_fold"))
            / (pl.col("_n").fill_null(0.0) + TE_ALPHA),
        )
    )


def build_features_lazy(lf: pl.LazyFrame, n_train_folds: int = N_TRAIN_FOLDS) -> pl.LazyFrame:
    """Full lazy feature query; drop helper columns and return rows ordered by txn_id."""
    out = merchant_target_encoding(card_features(lf), n_train_folds)
    helpers = [c for c in out.collect_schema().names() if c.startswith("_")]
    return out.drop(helpers).sort("txn_id")


def build_features(df: pl.DataFrame | pl.LazyFrame, n_train_folds: int = N_TRAIN_FOLDS) -> pl.DataFrame:
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    return build_features_lazy(lf, n_train_folds).collect()
