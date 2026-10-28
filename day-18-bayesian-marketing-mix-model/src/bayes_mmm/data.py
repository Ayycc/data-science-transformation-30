"""SYNTHETIC weekly sales panel with known media effects, plus a last-click attribution report.

All quantities are synthetic. Units: spend and sales in $k per region-week; population in millions.
The generator works per capita (per million people) so regions of different size share one
response curve family, which is what the hierarchical model assumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CHANNELS, DATA_RAW, DATA_SAMPLE, L_MAX, N_REGIONS, N_WEEKS, SEED, START_DATE

# Ground-truth media parameters (per channel, in CHANNELS order).
TRUE_ALPHA = np.array([0.60, 0.15, 0.35, 0.25])  # geometric adstock decay
TRUE_LAM = np.array([2.0, 3.0, 1.5, 2.5])  # logistic saturation steepness on x / channel_scale
TARGET_ROAS = np.array([1.6, 1.1, 2.2, 1.4])  # national ROAS the generator is calibrated to
REGION_BETA_SD = 0.20  # log-scale spread of channel effects across regions
NOISE_SD_PC = 2.5  # $k per million people per week
PRICE_EFFECT_PC = -150.0  # $k per million per unit of (price index - 1): 10% discount -> +15
COMP_EFFECT_PC = -4.0  # $k per million per 1 sd of competitor activity
HOLIDAY_EFFECT_PC = 12.0


@dataclass
class Panel:
    """Arrays indexed (week, region[, channel])."""

    dates: pd.DatetimeIndex
    regions: list[str]
    population: np.ndarray  # (R,)
    spend: np.ndarray  # (T, R, C) $k
    sales: np.ndarray  # (T, R) $k
    price: np.ndarray  # (T, R) index, 1 = list price
    competitor: np.ndarray  # (T, R) standardised activity
    holiday: np.ndarray  # (T,) 0/1
    truth: dict = field(default_factory=dict)

    @property
    def n_weeks(self) -> int:
        return len(self.dates)

    def to_frame(self) -> pd.DataFrame:
        t, r = self.sales.shape
        df = pd.DataFrame(
            {
                "week": np.repeat(self.dates.strftime("%Y-%m-%d"), r),
                "region": np.tile(self.regions, t),
                "population_m": np.tile(self.population, t).round(4),
                "sales_k": self.sales.reshape(-1).round(3),
                "price_index": self.price.reshape(-1).round(4),
                "competitor_z": self.competitor.reshape(-1).round(4),
                "holiday": np.repeat(self.holiday, r).astype(int),
            }
        )
        for k, c in enumerate(CHANNELS):
            df[f"spend_{c}_k"] = self.spend[:, :, k].reshape(-1).round(3)
        return df

    @classmethod
    def from_frame(cls, df: pd.DataFrame, truth: dict | None = None) -> Panel:
        dates = pd.DatetimeIndex(sorted(df["week"].unique()))
        regions = list(dict.fromkeys(df["region"]))
        df = df.set_index(["week", "region"]).loc[[(d.strftime("%Y-%m-%d"), r) for d in dates for r in regions]]
        t, r = len(dates), len(regions)
        pop = df["population_m"].to_numpy().reshape(t, r)[0]
        spend = np.stack([df[f"spend_{c}_k"].to_numpy().reshape(t, r) for c in CHANNELS], axis=-1)
        return cls(
            dates=dates,
            regions=regions,
            population=pop,
            spend=spend,
            sales=df["sales_k"].to_numpy().reshape(t, r),
            price=df["price_index"].to_numpy().reshape(t, r),
            competitor=df["competitor_z"].to_numpy().reshape(t, r),
            holiday=df["holiday"].to_numpy().reshape(t, r)[:, 0].astype(float),
            truth=truth or {},
        )


def geometric_adstock_np(x: np.ndarray, alpha: np.ndarray, l_max: int = L_MAX, normalize: bool = True) -> np.ndarray:
    """Geometric adstock along axis 0. x: (T, ..., C); alpha broadcast over the last axis."""
    alpha = np.asarray(alpha, dtype=float)
    w = alpha[None, :] ** np.arange(l_max)[:, None]  # (L, C)
    if normalize:
        w = w / w.sum(axis=0, keepdims=True)
    out = np.zeros_like(x, dtype=float)
    for lag in range(l_max):
        shifted = np.zeros_like(x, dtype=float)
        shifted[lag:] = x[: x.shape[0] - lag]
        out += w[lag] * shifted
    return out


def logistic_saturation_np(x: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """(1 - exp(-lam x)) / (1 + exp(-lam x)), the same form as pymc_marketing.logistic_saturation."""
    e = np.exp(-np.asarray(lam) * x)
    return (1.0 - e) / (1.0 + e)


def fourier_features(dates: pd.DatetimeIndex, order: int) -> np.ndarray:
    """Yearly Fourier terms (T, 2*order)."""
    doy = dates.dayofyear.to_numpy() / 365.25
    cols = []
    for k in range(1, order + 1):
        cols += [np.sin(2 * np.pi * k * doy), np.cos(2 * np.pi * k * doy)]
    return np.column_stack(cols)


def _spend_patterns(rng: np.random.Generator, t: int, r: int, season: np.ndarray, q4: np.ndarray) -> np.ndarray:
    """Per-capita weekly spend ($k per million people), (T, R, C)."""
    pc = np.zeros((t, r, len(CHANNELS)))
    # TV: 4-week flights, ~55% of blocks on air, national calendar with regional weight noise.
    blocks = rng.random((int(np.ceil(t / 4)), r)) < 0.55
    on = np.repeat(blocks, 4, axis=0)[:t]
    pc[:, :, 0] = on * 6.0 * rng.lognormal(0.0, 0.25, (t, r))
    # Search: follows demand (seasonality), the classic confounder.
    pc[:, :, 1] = 3.0 * (1.0 + 0.6 * season[:, None]) * rng.lognormal(0.0, 0.25, (t, r))
    # Social: always on with occasional bursts.
    burst = rng.random((t, r)) < 0.10
    pc[:, :, 2] = 2.0 * rng.lognormal(0.0, 0.35, (t, r)) * np.where(burst, 2.5, 1.0)
    # Retail media: heavier in Q4.
    pc[:, :, 3] = 2.5 * (1.0 + 0.5 * q4[:, None]) * rng.lognormal(0.0, 0.30, (t, r))
    return pc


def simulate(n_weeks: int = N_WEEKS, n_regions: int = N_REGIONS, seed: int = SEED) -> Panel:
    """Generate the SYNTHETIC panel and its ground truth."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=n_weeks, freq="7D")
    regions = [f"R{i:02d}" for i in range(1, n_regions + 1)]
    t, r, c = n_weeks, n_regions, len(CHANNELS)

    population = np.round(rng.uniform(0.6, 3.0, r), 3)
    doy = dates.dayofyear.to_numpy() / 365.25
    season = np.cos(2 * np.pi * (doy - 0.92))  # peak in early December
    q4 = np.asarray(dates.month >= 10, dtype=float)
    holiday = np.asarray((dates.month == 12) & (dates.day >= 18), dtype=float)

    base_r = 100.0 * np.exp(rng.normal(0.0, 0.08, r))
    trend = 6.0 * np.arange(t) / (t - 1)
    seas = 8.0 * season + 3.0 * np.sin(4 * np.pi * doy)

    promo = rng.random((t, r)) < 0.12
    price = 1.0 + rng.normal(0.0, 0.01, (t, r)) - promo * rng.uniform(0.05, 0.15, (t, r))
    comp = np.zeros((t, r))
    eps = rng.normal(0.0, 1.0, (t, r))
    for i in range(1, t):
        comp[i] = 0.8 * comp[i - 1] + 0.6 * eps[i]
    comp = (comp - comp.mean()) / comp.std()

    spend_pc = _spend_patterns(rng, t, r, season, q4)
    channel_scale = spend_pc.reshape(-1, c).max(axis=0)
    sat = logistic_saturation_np(geometric_adstock_np(spend_pc / channel_scale, TRUE_ALPHA), TRUE_LAM)  # (T,R,C)

    region_mult = np.exp(rng.normal(0.0, REGION_BETA_SD, (r, c)))
    # Calibrate the channel-level beta so the national ROAS is close to TARGET_ROAS.
    unit = (sat * region_mult[None] * population[None, :, None]).sum(axis=(0, 1))
    spend_tot = (spend_pc * population[None, :, None]).sum(axis=(0, 1))
    beta_c = TARGET_ROAS * spend_tot / unit
    beta_rc = beta_c[None, :] * region_mult  # $k per million per week at full saturation
    contrib_pc = sat * beta_rc[None]  # (T,R,C)

    base_pc = base_r[None, :] + trend[:, None] + seas[:, None] + HOLIDAY_EFFECT_PC * holiday[:, None]
    controls_pc = PRICE_EFFECT_PC * (price - 1.0) + COMP_EFFECT_PC * comp
    sales_pc = base_pc + controls_pc + contrib_pc.sum(axis=-1) + rng.normal(0.0, NOISE_SD_PC, (t, r))

    spend = spend_pc * population[None, :, None]
    sales = sales_pc * population[None, :]
    contrib = contrib_pc * population[None, :, None]
    true_roas = contrib.sum(axis=(0, 1)) / spend.sum(axis=(0, 1))
    baseline_total = ((base_pc + controls_pc) * population[None, :]).sum()

    truth = {
        "label": "SYNTHETIC ground truth",
        "seed": seed,
        "channels": list(CHANNELS),
        "alpha": TRUE_ALPHA.tolist(),
        "lam": TRUE_LAM.tolist(),
        "channel_scale_pc": channel_scale.tolist(),
        "beta_pc": beta_rc.tolist(),  # (R, C) per-capita $k at full saturation
        "beta_channel_pc": beta_c.tolist(),
        "roas": true_roas.tolist(),
        "contribution_total_k": contrib.sum(axis=(0, 1)).tolist(),
        "baseline_total_k": float(baseline_total),
        "sales_total_k": float(sales.sum()),
        "price_effect_pc": PRICE_EFFECT_PC,
        "competitor_effect_pc": COMP_EFFECT_PC,
        "holiday_effect_pc": HOLIDAY_EFFECT_PC,
        "noise_sd_pc": NOISE_SD_PC,
        "base_pc": base_r.tolist(),
    }
    panel = Panel(dates, regions, population, spend, sales, price, comp, holiday, truth)
    truth["last_click"] = last_click_report(panel, contrib)
    return panel


def last_click_report(panel: Panel, contrib: np.ndarray) -> dict:
    """SYNTHETIC platform-style last-click attribution.

    Assumptions (illustrative, documented in the README): TV has almost no click path and keeps 10% of
    its true incremental sales; social keeps 50% and retail media 80%. Search, the last click before
    purchase, collects 70% of what the other channels lose plus 5% of baseline (brand-search) sales.
    """
    own = contrib.sum(axis=(0, 1))
    keep = np.array([0.10, 1.0, 0.50, 0.80])
    credited = own * keep
    lost = (own * (1 - keep)).sum()
    baseline = panel.truth.get("baseline_total_k", panel.sales.sum() - own.sum())
    credited[1] += 0.70 * lost + 0.05 * baseline
    spend_tot = panel.spend.sum(axis=(0, 1))
    return {"credited_sales_k": credited.tolist(), "roas": (credited / spend_tot).tolist()}


def save(panel: Panel, raw_dir: Path = DATA_RAW, sample_dir: Path = DATA_SAMPLE) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    df = panel.to_frame()
    paths = {
        "raw": raw_dir / "panel.csv.gz",
        "truth": raw_dir / "truth.json",
        "sample": sample_dir / "panel_sample.csv.gz",
        "sample_truth": sample_dir / "truth.json",
    }
    df.to_csv(paths["raw"], index=False)
    df.to_csv(paths["sample"], index=False)  # the full panel is ~150 KB gzipped, so the sample is the full panel
    for key in ("truth", "sample_truth"):
        paths[key].write_text(json.dumps(panel.truth, indent=2))
    return paths


def load(path: Path, truth_path: Path | None = None) -> Panel:
    truth = json.loads(truth_path.read_text()) if truth_path and truth_path.exists() else {}
    return Panel.from_frame(pd.read_csv(path), truth)


def get_panel(seed: int = SEED) -> tuple[Panel, str]:
    """Data path: this project is SYNTHETIC by design (parameter recovery needs ground truth).

    No network call is made: the seeded generator always runs (it takes well under a second) and
    reproduces data/sample/panel_sample.csv.gz exactly. Returns (panel, path_used).
    """
    panel = simulate(seed=seed)
    return panel, "synthetic_generator"
