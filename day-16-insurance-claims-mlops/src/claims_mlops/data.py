"""Data acquisition: public freMTPL2 via OpenML if reachable, else a seeded SYNTHETIC generator.

Both paths produce the same monthly policy table with columns
``month, PolicyID, Exposure, ClaimNb, ClaimAmount, PurePremium`` plus the features in ``config.FEATURES``.
"""

from __future__ import annotations

import io
import json
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

from claims_mlops import config as C

OPENML_FREQ = "https://api.openml.org/data/v1/download/20649148"  # freMTPL2freq (OpenML id 41214)
OPENML_SEV = "https://api.openml.org/data/v1/download/20649149"  # freMTPL2sev (OpenML id 41215)


@dataclass
class Acquired:
    policies: pd.DataFrame
    provenance: dict


# --------------------------------------------------------------------------------------
# SYNTHETIC generator
# --------------------------------------------------------------------------------------
def _true_frequency(df: pd.DataFrame) -> np.ndarray:
    """Annual claim frequency (SYNTHETIC ground truth, freMTPL2-like shape)."""
    age = df["DrivAge"].to_numpy(float)
    young = np.exp(-(age - 18) / 6.0)  # steep decay after 18
    old = np.clip(age - 70, 0, None) / 20.0
    lin = (
        np.log(0.075)
        + 1.10 * young
        + 0.35 * old
        + 0.012 * (df["BonusMalus"].to_numpy(float) - 50)
        + 0.10 * np.log(df["Density"].to_numpy(float) / 100)
        + 0.03 * (df["VehPower"].to_numpy(float) - 6)
        + 0.08 * (df["VehGas"].to_numpy() == "Diesel")
        - 0.25 * (df["VehBrand"].to_numpy() == "B12")
        + 0.10 * np.isin(df["Region"].to_numpy(), ["R82", "R93"])
        # non-additive structure a GLM without interactions cannot express
        + 1.00 * ((age < 26) & (df["VehPower"].to_numpy() >= 9))
        + 0.60 * ((df["BonusMalus"].to_numpy() > 90) & (df["Density"].to_numpy() > 2_000))
        - 0.50 * ((age > 55) & (df["VehGas"].to_numpy() == "Diesel") & (df["Density"].to_numpy() < 200))
    )
    return np.exp(lin)


def _true_severity(df: pd.DataFrame, month: np.ndarray, inflation: float) -> np.ndarray:
    """Mean cost per claim in EUR (SYNTHETIC ground truth), with monthly claims inflation."""
    vehage = df["VehAge"].to_numpy(float)
    lin = (
        np.log(1_700)
        + 0.06 * (df["VehPower"].to_numpy(float) - 6)
        - 0.25 * np.log1p(vehage) / np.log(21)
        + 0.08 * np.searchsorted(C.AREAS, df["Area"].to_numpy())
        + 0.15 * (df["DrivAge"].to_numpy(float) < 25)
        + 0.12 * np.isin(df["VehBrand"].to_numpy(), ["B10", "B11"])
        + 0.50 * ((vehage < 3) & (df["VehPower"].to_numpy() >= 9))
    )
    return np.exp(lin) * (1 + inflation) ** (month - 1)


def _features(rng: np.random.Generator, n: int, shifted: bool) -> pd.DataFrame:
    """Policy covariates. ``shifted`` applies the injected covariate shift (new young-driver urban product)."""
    p_young = 0.30 if shifted else 0.06
    young = rng.random(n) < p_young
    drivage = np.where(young, rng.integers(18, 27, n), np.clip(np.round(22 + rng.gamma(4.0, 6.5, n)), 18, 95))
    density_mu = np.log(250) + (0.9 if shifted else 0.0)
    density = np.clip(np.round(np.exp(rng.normal(density_mu, 1.6, n))), 1, 27_000)
    area_idx = np.clip(np.digitize(np.log(density), [3.0, 4.5, 6.0, 7.5, 9.0]) + rng.integers(-1, 2, n), 0, 5)
    bm_base = 50 + rng.gamma(0.6, 12, n) + np.where(drivage < 25, rng.uniform(20, 50, n), 0)
    bonus = np.clip(np.round(bm_base), 50, 230)
    return pd.DataFrame(
        {
            "VehPower": rng.choice(np.arange(4, 16), n, p=_power_p()),
            "VehAge": np.clip(np.round(rng.gamma(1.6, 4.5, n)), 0, 40).astype(int),
            "DrivAge": drivage.astype(int),
            "BonusMalus": bonus.astype(int),
            "Density": density.astype(int),
            "Area": np.array(C.AREAS)[area_idx],
            "VehBrand": rng.choice(C.BRANDS, n, p=_brand_p()),
            "VehGas": rng.choice(C.GAS, n, p=[0.5, 0.5]),
            "Region": rng.choice(C.REGIONS, n, p=_region_p()),
        }
    )


def _power_p() -> np.ndarray:
    w = np.array([16, 21, 21, 18, 9, 5, 3, 2, 1.5, 1.2, 1.0, 0.8])
    return w / w.sum()


def _brand_p() -> np.ndarray:
    w = np.array([24, 22, 8, 4, 5, 4, 3, 2, 21, 2, 5.0])
    return w / w.sum()


def _region_p() -> np.ndarray:
    w = np.array([10, 18, 4, 6, 7, 5, 8, 6, 12, 4.0])
    return w / w.sum()


def generate_synthetic(cfg: C.SimConfig | None = None) -> pd.DataFrame:
    """SYNTHETIC monthly policy cohorts with compound Poisson-gamma (Tweedie) claim cost."""
    cfg = cfg or C.SimConfig()
    rng = np.random.default_rng(cfg.seed)
    frames = []
    for m in range(1, cfg.n_months + 1):
        n = cfg.policies_per_month
        df = _features(rng, n, shifted=m >= cfg.shift_start_month)
        df.insert(0, "month", m)
        exposure = np.where(rng.random(n) < 0.45, 1.0, rng.uniform(0.05, 1.0, n)).round(3)
        df.insert(1, "Exposure", exposure)
        lam = _true_frequency(df) * exposure
        nb = rng.poisson(lam)
        mu_sev = _true_severity(df, np.full(n, m), cfg.inflation_per_month)
        shape = 2.0
        # sum of nb iid Gamma(shape, mu/shape) = Gamma(nb*shape, mu/shape)
        amount = np.where(nb > 0, rng.gamma(np.maximum(nb, 1) * shape, mu_sev / shape), 0.0)
        df["ClaimNb"] = nb
        df["ClaimAmount"] = amount.round(2)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.insert(1, "PolicyID", np.arange(1, len(out) + 1))
    out[C.TARGET] = out["ClaimAmount"] / out["Exposure"]
    return out


# --------------------------------------------------------------------------------------
# Public freMTPL2 path (used only if OpenML is reachable)
# --------------------------------------------------------------------------------------
def _fetch_arff(url: str, timeout: float) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed https URL)
        text = resp.read().decode("utf-8", errors="replace")
    lines = text.splitlines()
    cols, start = [], 0
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if low.startswith("@attribute"):
            cols.append(line.split()[1].strip("'\""))
        if low.startswith("@data"):
            start = i + 1
            break
    body = "\n".join(lines[start:])
    return pd.read_csv(io.StringIO(body), header=None, names=cols, quotechar="'", na_values="?")


def monthly_from_public(freq: pd.DataFrame, sev: pd.DataFrame, cfg: C.SimConfig | None = None) -> pd.DataFrame:
    """Turn freMTPL2 (no time axis) into monthly batches with the same injected shift as the simulator.

    Policies are assigned to months at random; from ``shift_start_month`` batches are re-sampled to
    over-represent drivers under 27 and denser communes, and claim amounts are inflated monthly.
    """
    cfg = cfg or C.SimConfig()
    rng = np.random.default_rng(cfg.seed)
    amt = sev.groupby("IDpol")["ClaimAmount"].sum()
    df = freq.copy()
    df["ClaimAmount"] = df["IDpol"].map(amt).fillna(0.0)
    df.loc[df["ClaimNb"] == 0, "ClaimAmount"] = 0.0
    df = df.rename(columns={"IDpol": "PolicyID"})
    for c in C.CAT_FEATURES:
        df[c] = df[c].astype(str).str.strip("'\"")
    df["Exposure"] = df["Exposure"].clip(0.01, 1.0)
    n_per = min(cfg.policies_per_month, len(df) // cfg.n_months)
    frames = []
    w_shift = np.where(df["DrivAge"] < 27, 5.0, 1.0) * np.where(df["Density"] > 1000, 2.0, 1.0)
    w_shift = w_shift / w_shift.sum()
    for m in range(1, cfg.n_months + 1):
        p = w_shift if m >= cfg.shift_start_month else None
        idx = rng.choice(len(df), n_per, replace=False, p=p)
        part = df.iloc[idx].copy()
        part["ClaimAmount"] = part["ClaimAmount"] * (1 + cfg.inflation_per_month) ** (m - 1)
        part.insert(0, "month", m)
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    out[C.TARGET] = out["ClaimAmount"] / out["Exposure"]
    keep = ["month", "PolicyID", "Exposure", *C.FEATURES, "ClaimNb", "ClaimAmount", C.TARGET]
    return out[keep]


def acquire(force_synthetic: bool = False, cfg: C.SimConfig | None = None, timeout: float = 15.0) -> Acquired:
    """Try OpenML freMTPL2; on any failure fall back to the SYNTHETIC generator."""
    cfg = cfg or C.SimConfig()
    prov: dict = {"public_source": "OpenML freMTPL2freq (41214) + freMTPL2sev (41215)", "attempted": not force_synthetic}
    if not force_synthetic:
        try:
            freq = _fetch_arff(OPENML_FREQ, timeout)
            sev = _fetch_arff(OPENML_SEV, timeout)
            df = monthly_from_public(freq, sev, cfg)
            prov.update(path="public", synthetic=False, rows=len(df))
            return Acquired(df, prov)
        except Exception as exc:  # network blocked, 403, parse error -> fall back
            prov["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    df = generate_synthetic(cfg)
    prov.update(path="synthetic", synthetic=True, rows=len(df), generator="claims_mlops.data.generate_synthetic", seed=cfg.seed)
    return Acquired(df, prov)


def load_or_create(force_synthetic: bool = False, cfg: C.SimConfig | None = None) -> Acquired:
    """Cache the acquired table under data/raw/ (git-ignored)."""
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    path, meta = C.DATA_RAW / "policies.parquet", C.DATA_RAW / "provenance.json"
    cfg = cfg or C.SimConfig()
    if path.exists() and meta.exists():
        prov = json.loads(meta.read_text())
        if prov.get("sim") == cfg.__dict__ and (not force_synthetic or prov.get("synthetic")):
            return Acquired(pd.read_parquet(path), prov)
    acq = acquire(force_synthetic=force_synthetic, cfg=cfg)
    acq.provenance["sim"] = cfg.__dict__
    acq.policies.to_parquet(path, index=False)
    meta.write_text(json.dumps(acq.provenance, indent=2))
    return acq


def write_sample(df: pd.DataFrame, n: int = 4_000, seed: int = C.SEED) -> None:
    """Stratified (by month) sample committed to data/sample/ for offline tests (<= 1 MB)."""
    C.DATA_SAMPLE.mkdir(parents=True, exist_ok=True)
    per = max(1, n // df["month"].nunique())
    s = df.groupby("month", group_keys=False).sample(n=per, random_state=seed)
    s.to_csv(C.DATA_SAMPLE / "policies_sample.csv", index=False, float_format="%.4f")


def split_months(df: pd.DataFrame, months: range | list[int]) -> pd.DataFrame:
    return df[df["month"].isin(list(months))].reset_index(drop=True)
