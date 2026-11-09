"""What-if backend for the Streamlit app: rebuilds the plan-year context from committed artefacts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import promo as promo_mod
from .calibrate import Calibration
from .config import HISTORICAL_ASSIGNMENT, Decisions
from .demand import Posterior
from .scenarios import PlanContext, risk_summary, run_many
from .twin import StartState

MODEL_DIR = Path(__file__).resolve().parents[2] / "results" / "model"


def load_context(model_dir: Path = MODEL_DIR) -> tuple[PlanContext, dict]:
    meta = json.loads((model_dir / "plan_context.json").read_text())
    with np.load(model_dir / "posterior.npz") as z:
        post = Posterior.from_npz(z)
    cals = {}
    for k, (p, q) in meta["calendars"].items():
        cals[k] = (promo_mod.weekly_to_daily(np.array(p)), promo_mod.weekly_to_daily(np.array(q)))
    ctx = PlanContext(
        cal=Calibration.from_dict(meta["calibration"]),
        posterior=post,
        start=StartState.from_snapshot(meta["snapshot"]),
        day0=meta["day0"],
        horizon=meta["horizon"],
        calendars=cals,
        annual_units=np.zeros(1),
    )
    return ctx, meta


def decisions(meta: dict, open_east: bool, nearshore: float, store_z: float, promo: str) -> Decisions:
    assign = tuple(meta["east_assignment"]) if open_east else HISTORICAL_ASSIGNMENT
    return replace(Decisions(), open_dc_east=open_east, assignment=assign, nearshore_share=nearshore, store_z=store_z, promo_strategy=promo)


def evaluate(ctx: PlanContext, dec: Decisions, regime: str, reps: int, seed: int = 50_000) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the chosen decisions and the status quo on the same seeds (common random numbers)."""
    base = pd.DataFrame(run_many(ctx, [(Decisions(), regime, r, seed + r) for r in range(reps)]))
    mine = pd.DataFrame(run_many(ctx, [(dec, regime, r, seed + r) for r in range(reps)]))
    return mine, base, risk_summary(mine, base)
