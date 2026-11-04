"""Census-like microdata: UCI Adult loader (public path) and a seeded SYNTHETIC population generator (fallback).

Both paths return the same schema, so every downstream step is identical. Domain bounds below are *public*
(they come from the questionnaire design, not from the data), which matters for differential privacy:
clipping ranges and histogram bins must never be read off the private records.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

UCI_URLS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "https://archive.ics.uci.edu/static/public/2/adult.zip",
)
ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

WORKCLASS = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Local-gov", "State-gov", "Federal-gov", "Without-pay"]
EDUCATION = [
    "Preschool",
    "1st-4th",
    "5th-6th",
    "7th-8th",
    "9th",
    "10th",
    "11th",
    "12th",
    "HS-grad",
    "Some-college",
    "Assoc-voc",
    "Assoc-acdm",
    "Bachelors",
    "Masters",
    "Prof-school",
    "Doctorate",
]
# population shares of each education level in the 1994 CPS extract (public published tabulation, rounded)
EDUCATION_SHARE = [0.0016, 0.005, 0.010, 0.020, 0.016, 0.029, 0.036, 0.013, 0.322, 0.224, 0.042, 0.033, 0.164, 0.053, 0.018, 0.013]
MARITAL = ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
OCCUPATION = [
    "Prof-specialty",
    "Exec-managerial",
    "Adm-clerical",
    "Sales",
    "Craft-repair",
    "Other-service",
    "Machine-op-inspct",
    "Transport-moving",
    "Handlers-cleaners",
    "Farming-fishing",
    "Tech-support",
    "Protective-serv",
    "Priv-house-serv",
    "Armed-Forces",
]
RACE = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
SEX = ["Male", "Female"]
REGION = ["United-States", "Latin-America", "Asia", "Europe", "Other"]
INCOME = ["<=50K", ">50K"]

CATEGORICAL: dict[str, list[str]] = {
    "workclass": WORKCLASS,
    "education": EDUCATION,
    "marital_status": MARITAL,
    "occupation": OCCUPATION,
    "race": RACE,
    "sex": SEX,
    "native_region": REGION,
    "income": INCOME,
}
# public bounds (questionnaire ranges / top-codes), used for clipping and binning
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {"age": (17, 90), "hours_per_week": (1, 99), "capital_gain": (0, 99999)}
COLUMNS = [
    "age",
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "race",
    "sex",
    "capital_gain",
    "hours_per_week",
    "native_region",
    "income",
]
TARGET = "income"
RARE_GROUP = ("race", "Amer-Indian-Eskimo")
QUASI_IDENTIFIERS = ["age", "sex", "race", "native_region", "marital_status", "education"]

_COUNTRY_REGION = {
    "United-States": "United-States",
    "Outlying-US(Guam-USVI-etc)": "United-States",
    "Mexico": "Latin-America",
    "Puerto-Rico": "Latin-America",
    "El-Salvador": "Latin-America",
    "Cuba": "Latin-America",
    "Jamaica": "Latin-America",
    "Dominican-Republic": "Latin-America",
    "Guatemala": "Latin-America",
    "Columbia": "Latin-America",
    "Haiti": "Latin-America",
    "Nicaragua": "Latin-America",
    "Peru": "Latin-America",
    "Ecuador": "Latin-America",
    "Honduras": "Latin-America",
    "Trinadad&Tobago": "Latin-America",
    "Philippines": "Asia",
    "India": "Asia",
    "China": "Asia",
    "Japan": "Asia",
    "Vietnam": "Asia",
    "Taiwan": "Asia",
    "Iran": "Asia",
    "Thailand": "Asia",
    "Cambodia": "Asia",
    "Laos": "Asia",
    "Hong": "Asia",
    "South": "Asia",
    "Germany": "Europe",
    "England": "Europe",
    "Italy": "Europe",
    "Poland": "Europe",
    "Portugal": "Europe",
    "France": "Europe",
    "Greece": "Europe",
    "Ireland": "Europe",
    "Yugoslavia": "Europe",
    "Hungary": "Europe",
    "Scotland": "Europe",
    "Holand-Netherlands": "Europe",
}


@dataclass
class Dataset:
    df: pd.DataFrame
    path_used: str
    attempts: list[str] = field(default_factory=list)


@dataclass
class PopulationConfig:
    n: int = 32561  # same size as the UCI Adult training file
    seed: int = 20261104


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _sample_rows(rng: np.random.Generator, weights: np.ndarray) -> np.ndarray:
    """Sample one category per row from an (n, k) matrix of unnormalised weights."""
    p = weights / weights.sum(1, keepdims=True)
    u = rng.random(len(p))[:, None]
    return np.minimum((p.cumsum(1) < u).sum(1), p.shape[1] - 1)


def generate_population(cfg: PopulationConfig | None = None) -> pd.DataFrame:
    """SYNTHETIC census-like population with realistic dependencies and rare subgroups (seeded)."""
    cfg = cfg or PopulationConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n
    race = rng.choice(len(RACE), size=n, p=np.array([0.854, 0.096, 0.031, 0.0098, 0.0092]) / 1.0)
    male = rng.random(n) < 0.67
    # region depends on race
    reg_p = np.array(
        [
            [0.925, 0.045, 0.004, 0.022, 0.004],  # White
            [0.930, 0.050, 0.000, 0.002, 0.018],  # Black
            [0.330, 0.010, 0.640, 0.000, 0.020],  # Asian-Pac-Islander
            [0.985, 0.015, 0.000, 0.000, 0.000],  # Amer-Indian-Eskimo
            [0.450, 0.500, 0.010, 0.000, 0.040],  # Other
        ]
    )
    region = _sample_rows(rng, reg_p[race])
    age = np.clip(np.round(17 + rng.gamma(2.3, 9.0, size=n)), 17, 90).astype(int)

    # education: latent normal shifted by group, mapped through the published education shares
    z = rng.normal(size=n)
    z += np.where(race == 2, 0.40, 0.0) + np.where(race == 3, -0.35, 0.0) + np.where(race == 1, -0.15, 0.0)
    z += np.where(region == 1, -0.75, 0.0) + np.where(age > 65, -0.35, 0.0)
    cuts = np.quantile(rng.normal(size=200_000), np.cumsum(EDUCATION_SHARE)[:-1] / sum(EDUCATION_SHARE))
    edu = np.searchsorted(cuts, z)
    young = age < 22
    edu = np.where(young, np.minimum(edu, 9), edu)  # nobody under 22 holds a degree beyond Some-college
    edu_num = edu + 1

    # marital status depends on age and sex
    a = age.astype(float)
    w = np.zeros((n, len(MARITAL)))
    p_never = _sigmoid(1.2 - 0.2 * (a - 20))
    w[:, 0] = (1 - p_never) * np.where(male, 0.66, 0.40)
    w[:, 1] = p_never + 0.03
    w[:, 2] = (1 - p_never) * np.where(male, 0.13, 0.27)
    w[:, 3] = (1 - p_never) * np.where(male, 0.025, 0.06) * np.where(race == 1, 2.5, 1.0)
    w[:, 4] = (1 - p_never) * _sigmoid((a - 68) / 6) * np.where(male, 0.25, 0.9)
    w[:, 5] = (1 - p_never) * 0.018
    w[:, 6] = (1 - p_never) * 0.0012
    marital = _sample_rows(rng, w)
    married = marital == 0

    # occupation depends on education and sex
    occ_base = np.array([0.127, 0.125, 0.116, 0.112, 0.126, 0.101, 0.061, 0.049, 0.042, 0.031, 0.028, 0.020, 0.005, 0.0004])
    occ_edu = np.array([0.65, 0.40, 0.05, 0.10, -0.15, -0.45, -0.45, -0.35, -0.50, -0.40, 0.25, 0.05, -0.80, 0.0])
    occ_male = np.array([0.0, 0.35, -1.10, 0.0, 1.60, -0.50, 0.40, 1.50, 1.10, 1.30, 0.0, 1.00, -2.50, 1.5])
    e = (edu_num[:, None] - 10) / 2.0
    w = occ_base[None, :] * np.exp(occ_edu[None, :] * e + occ_male[None, :] * (male[:, None] - 0.5))
    occ = _sample_rows(rng, w)

    # workclass depends on occupation
    wc_p = np.tile(np.array([0.78, 0.07, 0.03, 0.06, 0.035, 0.0246, 0.0004]), (len(OCCUPATION), 1))
    wc_p[0] = [0.55, 0.08, 0.03, 0.17, 0.11, 0.0596, 0.0004]
    wc_p[1] = [0.66, 0.10, 0.12, 0.05, 0.035, 0.0346, 0.0004]
    wc_p[9] = [0.45, 0.42, 0.05, 0.03, 0.02, 0.0296, 0.0004]
    wc_p[11] = [0.30, 0.01, 0.01, 0.48, 0.12, 0.0796, 0.0004]
    wc_p[13] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    workclass = _sample_rows(rng, wc_p[occ])

    # hours per week: a spike at exactly 40, otherwise depends on sex, occupation and age
    hours = 40 + 4.0 * male + np.where(occ == 1, 5, 0) + np.where(occ == 9, 8, 0) - np.where(a < 22, 14, 0) - np.where(a > 65, 15, 0)
    hours = hours + rng.normal(0, 10, size=n)
    hours = np.where(rng.random(n) < 0.46, 40, hours)
    hours = np.clip(np.round(hours), 1, 99).astype(int)

    # income (>50K) from a logistic model; intercept fixed so that ~24% are positive
    occ_eff = np.array([0.6, 0.85, -0.2, 0.15, 0.0, -1.0, -0.4, -0.2, -0.8, -0.6, 0.4, 0.3, -1.5, 0.0])
    race_eff = np.array([0.0, -0.30, 0.05, -0.45, -0.40])
    lin = (
        0.38 * (edu_num - 10)
        + 2.0 * married
        + 0.40 * male
        + 0.035 * (hours - 40)
        + 0.06 * (a - 38)
        - 0.0016 * (a - 45) ** 2
        + occ_eff[occ]
        + race_eff[race]
        + np.where(workclass == 2, 0.6, 0.0)
    )
    lo, hi = -10.0, 5.0
    for _ in range(60):  # bisection on the intercept (deterministic)
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _sigmoid(lin + mid).mean() < 0.24 else (lo, mid)
    income = (rng.random(n) < _sigmoid(lin + lo)).astype(int)

    # capital gains: zero-inflated log-normal, top-coded at 99999 like the CPS
    p_gain = _sigmoid(-3.3 + 0.25 * (edu_num - 10) + 1.4 * income + 0.03 * (a - 40))
    has_gain = rng.random(n) < p_gain
    amount = np.round(np.exp(rng.normal(8.4, 0.95, size=n)))
    amount = np.where(rng.random(n) < 0.06, 99999, amount)
    gain = np.where(has_gain, np.clip(amount, 114, 99999), 0).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "workclass": np.array(WORKCLASS)[workclass],
            "education": np.array(EDUCATION)[edu],
            "marital_status": np.array(MARITAL)[marital],
            "occupation": np.array(OCCUPATION)[occ],
            "race": np.array(RACE)[race],
            "sex": np.where(male, "Male", "Female"),
            "capital_gain": gain,
            "hours_per_week": hours,
            "native_region": np.array(REGION)[region],
            "income": np.array(INCOME)[income],
        }
    )
    return df[COLUMNS]


def parse_adult(text: str) -> pd.DataFrame:
    """Parse the raw ``adult.data`` CSV into the shared schema (drops rows with missing values)."""
    raw = pd.read_csv(io.StringIO(text), header=None, names=ADULT_COLUMNS, skipinitialspace=True, na_values="?")
    raw = raw.dropna(subset=["workclass", "occupation", "native_country"])
    raw = raw[raw["workclass"] != "Never-worked"].copy()
    raw["native_region"] = raw["native_country"].map(_COUNTRY_REGION).fillna("Other")
    raw["income"] = raw["income"].str.replace(".", "", regex=False).str.strip()
    df = raw[COLUMNS].reset_index(drop=True)
    for c, (lo, hi) in NUMERIC_BOUNDS.items():
        df[c] = df[c].clip(lo, hi).astype(int)
    for c, levels in CATEGORICAL.items():
        df = df[df[c].isin(levels)]
    return df.reset_index(drop=True)


def load_dataset(mode: str = "auto", cfg: PopulationConfig | None = None, timeout: float = 10.0) -> Dataset:
    """``auto``: try UCI Adult, fall back to the SYNTHETIC generator. ``synthetic`` / ``uci`` force a path."""
    attempts: list[str] = []
    if mode in ("auto", "uci"):
        for url in UCI_URLS[:1]:
            try:
                with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - fixed public URL
                    text = r.read().decode("utf-8")
                df = parse_adult(text)
                attempts.append(f"{url}: ok ({len(df)} rows)")
                return Dataset(df=df, path_used="public_uci_adult", attempts=attempts)
            except Exception as exc:  # network blocked, DNS failure, HTTP error...
                attempts.append(f"{url}: {type(exc).__name__}: {exc}"[:240])
        if mode == "uci":
            raise RuntimeError("UCI Adult unreachable: " + "; ".join(attempts))
    return Dataset(df=generate_population(cfg), path_used="synthetic_generator", attempts=attempts)


def split(df: pd.DataFrame, seed: int, holdout_frac: float = 1 / 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random split into the confidential file (used by every release) and a holdout never shown to them."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_hold = int(round(len(df) * holdout_frac))
    return df.iloc[idx[n_hold:]].reset_index(drop=True), df.iloc[idx[:n_hold]].reset_index(drop=True)
