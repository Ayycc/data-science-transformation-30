"""Paths, scales and constants shared by the generator, features and model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
SAMPLE_DIR = DATA_DIR / "sample"
RESULTS_DIR = PROJECT_ROOT / "results"
COST_MATRIX_PATH = PROJECT_ROOT / "configs" / "cost_matrix.json"

SEED = 42
WINDOW_START = datetime(2025, 1, 1)
DAYS = 181  # 2025-01-01 .. 2025-06-30 (six calendar months)

# Time-based split on calendar months (fold = month number 1..6).
TRAIN_FOLDS = (1, 2, 3, 4)  # Jan-Apr 2025
VAL_FOLDS = (5,)  # May 2025
TEST_FOLDS = (6,)  # Jun 2025
N_TRAIN_FOLDS = len(TRAIN_FOLDS)

ATTACK_TYPES = ("none", "card_testing", "account_takeover", "geo_impossible")

# mcc name -> (median legit amount USD, in-person weight, online weight)
MCC_TABLE: dict[str, tuple[float, float, float]] = {
    "grocery": (45.0, 0.22, 0.04),
    "restaurants": (30.0, 0.18, 0.08),
    "fuel": (40.0, 0.12, 0.0),
    "pharmacy": (25.0, 0.06, 0.02),
    "apparel": (70.0, 0.08, 0.12),
    "electronics": (180.0, 0.03, 0.08),
    "home_improvement": (90.0, 0.05, 0.03),
    "entertainment": (35.0, 0.06, 0.08),
    "travel_air": (350.0, 0.0, 0.04),
    "hotels": (220.0, 0.03, 0.04),
    "ride_share": (18.0, 0.0, 0.10),
    "digital_goods": (12.0, 0.0, 0.20),
    "gift_cards": (75.0, 0.02, 0.03),
    "jewelry": (250.0, 0.01, 0.01),
    "utilities": (110.0, 0.0, 0.13),
}
MCC_NAMES: tuple[str, ...] = tuple(MCC_TABLE)
MCC_TO_CODE: dict[str, int] = {m: i for i, m in enumerate(MCC_NAMES)}

# Features handed to the models (all computed from information available at authorisation time).
FEATURES: tuple[str, ...] = (
    "amount",
    "hour",
    "dow",
    "is_online",
    "mcc_code",
    "card_txn_count_1h",
    "card_txn_count_24h",
    "card_txn_count_7d",
    "card_amount_sum_1h",
    "card_amount_sum_24h",
    "card_amount_mean_30d",
    "amount_to_mean_30d",
    "card_small_txn_count_1h",
    "secs_since_prev",
    "dist_prev_km",
    "speed_kmh",
    "dist_from_centroid_km",
    "is_new_device",
    "device_prior_txn_count",
    "is_new_merchant_for_card",
    "card_age_days",
    "card_txn_index",
    "merchant_risk_te",
    "merchant_hist_txn_count",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("mcc_code",)


@dataclass(frozen=True)
class Scale:
    name: str
    n_cards: int
    txn_rate_per_day: float  # mean legit transactions per card per day
    fraud_events_per_card: float  # attack campaigns per card over the window
    n_trials: int  # Optuna trials
    neg_sample_rate: float  # share of legit training rows kept (all fraud rows are kept)


SCALES: dict[str, Scale] = {
    # ~5.1M transactions, 100k cardholders, 6 months, ~0.3% fraud
    "full": Scale("full", 100_000, 0.28, 0.0195, n_trials=12, neg_sample_rate=0.1),
    # ~1M transactions: a quick end-to-end check of the full code path
    "medium": Scale("medium", 20_000, 0.28, 0.0195, n_trials=6, neg_sample_rate=0.2),
    # committed sample (<1 MB); fraud is boosted so tests have enough positives
    "sample": Scale("sample", 600, 0.28, 0.12, n_trials=3, neg_sample_rate=1.0),
}


def load_cost_matrix(path: Path = COST_MATRIX_PATH) -> dict:
    """Read the configurable cost matrix (USD)."""
    return json.loads(Path(path).read_text())
