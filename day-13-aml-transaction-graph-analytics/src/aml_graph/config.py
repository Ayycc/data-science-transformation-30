"""Central configuration: paths, scale, seeds and detector thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
ALERT_DIR = RESULTS_DIR / "alerts"

SEED = 20261023

# Public IBM "Transactions for Anti Money Laundering" (Altman et al., NeurIPS 2023) on Kaggle.
IBM_AML_URL = "https://www.kaggle.com/api/v1/datasets/download/ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
IBM_LOCAL_CSV = RAW_DIR / "HI-Small_Trans.csv"

TYPOLOGIES = ("fan_in", "fan_out", "cycle", "scatter_gather", "mule_chain")

HOUR = 3600.0
DAY = 86400.0


@dataclass(frozen=True)
class SimConfig:
    """Scale of the SYNTHETIC transaction graph."""

    n_accounts: int = 100_000
    n_background_tx: int = 2_000_000
    n_days: int = 180
    n_regions: int = 400
    instances_per_typology: int = 80
    seed: int = SEED


SMALL = SimConfig(n_accounts=4_000, n_background_tx=40_000, n_regions=40, instances_per_typology=12)


@dataclass(frozen=True)
class DetectorConfig:
    """Label-free typology detector thresholds (documented in the README)."""

    min_amount: float = 1_000.0  # only material transfers participate in flow motifs
    flow_window_s: float = 72 * HOUR  # max hold time between an incoming and the onward transfer
    flow_ratio_lo: float = 0.80  # onward amount / incoming amount
    flow_ratio_hi: float = 1.00
    star_window_s: float = 72 * HOUR
    star_min_counterparties: int = 5
    star_min_total: float = 20_000.0
    star_pass_through: float = 0.60  # share of the star's total that leaves (fan-in) / arrived (fan-out)
    star_follow_s: float = 7 * DAY
    sg_min_intermediates: int = 3
    chain_min_hops: int = 4
    cycle_max_hops: int = 7
    max_paths_per_edge: int = 50  # guard against combinatorial blow-up around busy hubs


@dataclass(frozen=True)
class ModelConfig:
    test_size: float = 0.30
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    n_bootstrap: int = 300
    betweenness_sources: int = 64
    n_alerts: int = 10
    target_recall: float = 0.80
    seed: int = SEED
    extra: dict = field(default_factory=dict)
