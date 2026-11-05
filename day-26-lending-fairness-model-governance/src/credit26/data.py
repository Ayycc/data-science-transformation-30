"""Data: public sources first (HMDA, then German Credit), else a seeded SYNTHETIC application generator.

Every dataset is returned as a :class:`CreditData` with a :class:`DatasetSpec` that tells the rest of the
pipeline which columns are model features, which are protected attributes (never model inputs), which
features are suspected proxies, which are actionable for counterfactuals and how each maps to an
adverse-action reason statement.

Label convention everywhere: ``bad = 1`` is the unfavourable outcome (default, or denial for HMDA);
the favourable decision is ``approve = 1``.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HMDA_URL = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=DE&years=2022&actions_taken=1,3&loan_purposes=1"
GERMAN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"


@dataclass
class DatasetSpec:
    numeric: list[str]
    categorical: list[str]
    protected: str  # main protected attribute (used for audit and mitigation, never as a model input)
    reference_group: str
    secondary_protected: list[str] = field(default_factory=list)
    proxy_candidates: list[str] = field(default_factory=list)
    monotone: dict[str, int] = field(default_factory=dict)  # +1: higher value -> higher risk
    actionable: dict[str, tuple[int, float]] = field(default_factory=dict)  # feature -> (direction, bound)
    reason_text: dict[str, str] = field(default_factory=dict)
    time_col: str | None = None
    has_unbiased_label: bool = False

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical


@dataclass
class CreditData:
    df: pd.DataFrame
    spec: DatasetSpec
    path_used: str
    attempts: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------------------------- synthetic

SYNTH_REASON_TEXT = {
    "credit_utilization": "Proportion of revolving balances to credit limits is too high",
    "debt_to_income": "Debt obligations are too high relative to income",
    "num_delinquencies": "Number of delinquent or past-due accounts",
    "num_inquiries_6m": "Too many recent credit inquiries",
    "credit_history_months": "Length of credit history is insufficient",
    "annual_income": "Income insufficient for the amount of credit requested",
    "employment_years": "Length of employment",
    "savings_balance": "Insufficient liquid assets / savings",
    "loan_amount": "Amount of credit requested is too high",
    "months_at_address": "Length of residence at current address",
    "age": "Age of applicant (permitted only as a positive factor under ECOA)",
    "zip_cluster": "Geographic area of residence (prohibited basis risk: proxy)",
}

SYNTH_SPEC = DatasetSpec(
    numeric=[
        "credit_utilization",
        "debt_to_income",
        "num_delinquencies",
        "num_inquiries_6m",
        "credit_history_months",
        "annual_income",
        "employment_years",
        "savings_balance",
        "loan_amount",
        "months_at_address",
    ],
    categorical=["zip_cluster"],
    protected="group",
    reference_group="A",
    secondary_protected=["sex", "age_band"],
    proxy_candidates=["zip_cluster", "months_at_address", "annual_income", "credit_history_months"],
    monotone={
        "credit_utilization": 1,
        "debt_to_income": 1,
        "num_delinquencies": 1,
        "num_inquiries_6m": 1,
        "credit_history_months": -1,
        "annual_income": -1,
        "employment_years": -1,
        "savings_balance": -1,
        "loan_amount": 1,
        "months_at_address": -1,
    },
    actionable={
        # direction -1: may only decrease, down to bound; +1: may only increase, up to bound x current (or bound absolute)
        "credit_utilization": (-1, 0.0),
        "debt_to_income": (-1, 0.05),
        "num_inquiries_6m": (-1, 0.0),
        "loan_amount": (-1, 0.5),  # relative: at most halve the requested amount
        "savings_balance": (1, 2.0),  # relative: at most double savings
    },
    reason_text=SYNTH_REASON_TEXT,
    time_col="month",
    has_unbiased_label=True,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_applications(n: int = 40_000, seed: int = 20261105, n_months: int = 24) -> pd.DataFrame:
    """SYNTHETIC loan applications with structural group gaps, a residential-segregation proxy and label bias.

    * ``group`` (A majority / B / C) is the protected attribute (a stand-in for race or ethnicity).
    * ``zip_cluster`` is assigned with strong group-dependent probabilities (residential segregation) so it is a proxy.
    * ``bad_unbiased`` comes from a risk model with **no** group or geography term.
    * ``bad`` (the recorded label used for training) adds historical bias: the same uniform draw is compared
      against a probability with an extra logit for groups B/C and for historically redlined clusters,
      so every unbiased bad is also a recorded bad, plus extra spurious bads for disadvantaged applicants.
    * Later months drift (utilisation, inquiries and DTI rise) to make the out-of-time PSI meaningful.
    """
    rng = np.random.default_rng(seed)
    month = rng.integers(1, n_months + 1, n)
    drift = (month - 1) / max(n_months - 1, 1)
    group = rng.choice(np.array(["A", "B", "C"]), size=n, p=[0.62, 0.23, 0.15])
    sex = rng.choice(np.array(["M", "F"]), size=n, p=[0.54, 0.46])
    age = np.clip(np.round(rng.normal(41, 12, n)), 18, 80)

    # residential segregation: clusters 0-9 majority-A, 10-14 B, 15-19 C
    pz = {
        "A": np.r_[np.full(10, 0.085), np.full(5, 0.01), np.full(5, 0.02)],
        "B": np.r_[np.full(10, 0.02), np.full(5, 0.14), np.full(5, 0.02)],
        "C": np.r_[np.full(10, 0.02), np.full(5, 0.02), np.full(5, 0.14)],
    }
    zip_cluster = np.empty(n, dtype=int)
    for g, p in pz.items():
        m = group == g
        zip_cluster[m] = rng.choice(20, size=m.sum(), p=p / p.sum())
    redlined = zip_cluster >= 10

    gB, gC, fem = (group == "B"), (group == "C"), (sex == "F")
    log_income = 10.95 + 0.012 * (age - 40) - 0.0004 * (age - 40) ** 2 - 0.18 * gB - 0.10 * gC - 0.08 * fem + 0.04 * drift + rng.normal(0, 0.42, n)
    income = np.round(np.exp(log_income), -2)
    hist = np.clip((age - 18) * 12 * rng.uniform(0.35, 1.0, n) * np.where(gB | gC, 0.82, 1.0), 6, None).round()
    util = np.clip(rng.beta(2.0, 3.2, n) + 0.04 * gB + 0.03 * gC + 0.07 * drift, 0, 1)
    dti = np.clip(rng.normal(0.31, 0.11, n) + 0.025 * gB + 0.015 * gC + 0.03 * drift, 0.02, 0.85)
    delinq = rng.poisson(np.exp(-1.6 + 1.8 * util - 0.003 * (hist - 120) / 12))
    inq = rng.poisson(1.0 + 1.0 * drift + 0.4 * util)
    emp = np.minimum(np.round(rng.exponential(6.5, n), 1), age - 17)
    savings = np.round(np.exp(np.log(income) - 2.6 + rng.normal(0, 1.0, n) - 0.25 * (gB | gC)), -1)
    loan = np.round(income * np.exp(rng.normal(np.log(0.33), 0.35, n)), -2)
    months_addr = np.round(rng.exponential(np.where(redlined, 30, 55), n)) + 1

    logit = (
        -2.55
        + 2.4 * (util - 0.4)
        + 3.2 * (dti - 0.32)
        + 0.42 * delinq
        + 0.14 * inq
        - 0.0035 * (hist - 150)
        - 0.55 * (np.log(income) - 10.9)
        - 0.03 * emp
        + 1.6 * (loan / income - 0.33)
        - 0.12 * np.log1p(savings / 1000)
    )
    u = rng.random(n)
    p_true = _sigmoid(logit)
    bias = 0.70 * gB + 0.45 * gC + 0.30 * redlined
    p_rec = _sigmoid(logit + bias)
    df = pd.DataFrame(
        {
            "application_id": np.arange(1, n + 1),
            "month": month,
            "group": group,
            "sex": sex,
            "age": age.astype(int),
            "zip_cluster": zip_cluster,
            "credit_utilization": util.round(3),
            "debt_to_income": dti.round(3),
            "num_delinquencies": delinq,
            "num_inquiries_6m": inq,
            "credit_history_months": hist.astype(int),
            "annual_income": income,
            "employment_years": emp,
            "savings_balance": savings,
            "loan_amount": loan,
            "months_at_address": months_addr.astype(int),
            "p_true": p_true.round(5),
            "bad_unbiased": (u < p_true).astype(int),
            "bad": (u < p_rec).astype(int),
        }
    )
    df["age_band"] = np.where(df["age"] >= 62, "62+", "<62")
    return df


# --------------------------------------------------------------------------------------------- public sources

HMDA_SPEC = DatasetSpec(
    numeric=["loan_amount", "income", "debt_to_income", "loan_to_value_ratio", "property_value", "loan_term"],
    categorical=[],
    protected="derived_race",
    reference_group="White",
    secondary_protected=["derived_sex"],
    proxy_candidates=["income", "property_value", "loan_amount"],
    monotone={"loan_amount": 1, "income": -1, "debt_to_income": 1, "loan_to_value_ratio": 1, "property_value": 0, "loan_term": 0},
    actionable={"loan_amount": (-1, 0.5), "debt_to_income": (-1, 0.05)},
    reason_text={
        "loan_amount": "Amount of credit requested",
        "income": "Income insufficient for the amount of credit requested",
        "debt_to_income": "Debt-to-income ratio",
        "loan_to_value_ratio": "Value or type of collateral not sufficient",
        "property_value": "Value of collateral",
        "loan_term": "Loan term",
    },
)

_DTI_MAP = {"<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0, "50%-60%": 55.0, ">60%": 65.0, "Exempt": np.nan}


def parse_hmda(text: str) -> pd.DataFrame:
    """HMDA Data Browser CSV -> one row per originated (1) or denied (3) application; ``bad`` = denied."""
    raw = pd.read_csv(io.StringIO(text), low_memory=False)
    raw = raw[raw["action_taken"].isin([1, 3])].copy()
    dti = raw["debt_to_income_ratio"].astype(str).map(lambda s: _DTI_MAP.get(s, pd.to_numeric(s, errors="coerce")))
    out = pd.DataFrame(
        {
            "loan_amount": pd.to_numeric(raw["loan_amount"], errors="coerce"),
            "income": pd.to_numeric(raw["income"], errors="coerce"),
            "debt_to_income": pd.to_numeric(dti, errors="coerce") / 100.0,
            "loan_to_value_ratio": pd.to_numeric(raw["loan_to_value_ratio"], errors="coerce"),
            "property_value": pd.to_numeric(raw["property_value"], errors="coerce"),
            "loan_term": pd.to_numeric(raw["loan_term"], errors="coerce"),
            "derived_race": raw["derived_race"].astype(str),
            "derived_sex": raw["derived_sex"].astype(str),
            "bad": (raw["action_taken"] == 3).astype(int),
        }
    )
    keep = out["derived_race"].isin(["White", "Black or African American", "Asian"]) & out["derived_sex"].isin(["Male", "Female"])
    out = out[keep].dropna().reset_index(drop=True)
    out["application_id"] = np.arange(1, len(out) + 1)
    return out


GERMAN_COLS = [
    "checking", "duration", "history", "purpose", "amount", "savings", "employment", "installment_rate", "personal_status",
    "debtors", "residence", "property", "age", "other_plans", "housing", "existing_credits", "job", "liable", "telephone",
    "foreign", "class",
]  # fmt: skip

GERMAN_SPEC = DatasetSpec(
    numeric=["duration", "amount", "installment_rate", "residence", "existing_credits", "liable"],
    categorical=["checking", "history", "purpose", "savings", "employment", "property", "other_plans", "housing", "job"],
    protected="sex",
    reference_group="male",
    secondary_protected=["age_band"],
    proxy_candidates=["housing", "employment", "residence"],
    monotone={"duration": 1, "amount": 1, "installment_rate": 1},
    actionable={"amount": (-1, 0.5), "duration": (-1, 6.0)},
    reason_text={c: c.replace("_", " ").capitalize() for c in GERMAN_COLS},
)


def parse_german(text: str) -> pd.DataFrame:
    """UCI Statlog German Credit (space separated, coded) -> frame; ``bad`` = class 2, ``sex`` from personal_status."""
    raw = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None, names=GERMAN_COLS)
    raw["bad"] = (raw["class"] == 2).astype(int)
    raw["sex"] = np.where(raw["personal_status"].isin(["A92", "A95"]), "female", "male")
    raw["age_band"] = np.where(raw["age"] < 25, "<25", "25+")
    raw["application_id"] = np.arange(1, len(raw) + 1)
    return raw.drop(columns=["class", "personal_status"])


def _fetch(url: str, timeout: float = 20.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (fixed public URL)
        return r.read().decode("utf-8", errors="replace")


def load_data(mode: str = "auto", n: int = 40_000, seed: int = 20261105, raw_dir: Path | None = None) -> CreditData:
    """``auto``: try HMDA, then German Credit, else the SYNTHETIC generator. ``synthetic``/``hmda``/``german`` force a path."""
    attempts: list[dict] = []
    if mode in ("auto", "hmda"):
        try:
            df = parse_hmda(_fetch(HMDA_URL))
            if len(df) < 500:
                raise ValueError(f"only {len(df)} usable rows")
            attempts.append({"source": "hmda", "url": HMDA_URL, "ok": True, "rows": len(df)})
            _save_raw(raw_dir, "hmda.csv", df)
            return CreditData(df, HMDA_SPEC, "hmda_public", attempts)
        except Exception as e:  # noqa: BLE001
            attempts.append({"source": "hmda", "url": HMDA_URL, "ok": False, "error": f"{type(e).__name__}: {e}"[:200]})
            if mode == "hmda":
                raise
    if mode in ("auto", "german"):
        try:
            df = parse_german(_fetch(GERMAN_URL))
            attempts.append({"source": "german_credit", "url": GERMAN_URL, "ok": True, "rows": len(df)})
            _save_raw(raw_dir, "german.csv", df)
            return CreditData(df, GERMAN_SPEC, "german_credit_public", attempts)
        except Exception as e:  # noqa: BLE001
            attempts.append({"source": "german_credit", "url": GERMAN_URL, "ok": False, "error": f"{type(e).__name__}: {e}"[:200]})
            if mode == "german":
                raise
    df = generate_applications(n=n, seed=seed)
    attempts.append({"source": "synthetic_generator", "ok": True, "rows": len(df), "seed": seed})
    return CreditData(df, SYNTH_SPEC, "synthetic_generator", attempts)


def _save_raw(raw_dir: Path | None, name: str, df: pd.DataFrame) -> None:
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_dir / name, index=False)


@dataclass
class Splits:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    scheme: str


def split_data(data: CreditData, seed: int = 0) -> Splits:
    """Time-based split when the data has a month column (train 1-14, validation 15-18, out-of-time test 19-24), else random 60/20/20."""
    df = data.df
    t = data.spec.time_col
    if t and t in df:
        tr, va = df[df[t] <= 14], df[(df[t] > 14) & (df[t] <= 18)]
        te = df[df[t] > 18]
        scheme = "time: train months 1-14, validation 15-18, out-of-time test 19-24"
    else:
        idx = np.random.default_rng(seed).permutation(len(df))
        a, b = int(0.6 * len(df)), int(0.8 * len(df))
        tr, va, te = df.iloc[idx[:a]], df.iloc[idx[a:b]], df.iloc[idx[b:]]
        scheme = "random 60/20/20"
    return Splits(tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True), scheme)
