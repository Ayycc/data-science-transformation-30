"""Config loading: every cost, service target and constraint comes from config/config.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "config" / "config.yaml"

REQUIRED_COSTS = ("holding_rate_annual", "ordering_cost_per_line", "shortage_goodwill_frac_price", "seasonal_salvage_frac_cost")


def load_config(path: str | Path | None = None) -> dict:
    cfg = yaml.safe_load(Path(path or DEFAULT).read_text())
    missing = [k for k in REQUIRED_COSTS if k not in cfg.get("costs", {})]
    if missing:
        raise KeyError(f"config is missing cost parameters: {missing}")
    return cfg
