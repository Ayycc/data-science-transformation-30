"""Utility and disclosure-risk evaluation of a released microdata file (synthetic or SDC-protected)."""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from .data import COLUMNS, QUASI_IDENTIFIERS, RARE_GROUP, TARGET
from .encoding import discretize, domain_sizes, model_frame, nearest_gower


# ----------------------------------------------------------------- utility
def marginal_fidelity(real: pd.DataFrame, syn: pd.DataFrame) -> dict:
    """Total variation distance (TVD) of every 1-way and 2-way marginal on the public bins (0 = identical)."""
    a, b = discretize(real), discretize(syn)
    sizes = [domain_sizes()[c] for c in COLUMNS]

    def tvd(cols: tuple[int, ...]) -> float:
        shape = tuple(sizes[c] for c in cols)
        fa = np.bincount(np.ravel_multi_index(tuple(a[:, c] for c in cols), shape), minlength=int(np.prod(shape))) / len(a)
        fb = np.bincount(np.ravel_multi_index(tuple(b[:, c] for c in cols), shape), minlength=int(np.prod(shape))) / len(b)
        return float(0.5 * np.abs(fa - fb).sum())

    one = {COLUMNS[j]: tvd((j,)) for j in range(len(COLUMNS))}
    two = [tvd(p) for p in combinations(range(len(COLUMNS)), 2)]
    return {
        "tvd_1way_mean": float(np.mean(list(one.values()))),
        "tvd_1way": one,
        "tvd_2way_mean": float(np.mean(two)),
        "tvd_2way_max": float(np.max(two)),
    }


def fit_classifier(df: pd.DataFrame, seed: int) -> HistGradientBoostingClassifier:
    X, y = model_frame(df)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, categorical_features="from_dtype", random_state=seed).fit(X, y)


def tstr(train_release: pd.DataFrame, test: pd.DataFrame, seed: int) -> dict:
    """Train on the released file, test on real held-out records; overall and for the rare subgroup."""
    X, y = model_frame(test)
    rare = (test[RARE_GROUP[0]] == RARE_GROUP[1]).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = fit_classifier(train_release, seed).predict_proba(X)[:, 1]
    pred = (p >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "auc_rare": float(roc_auc_score(y[rare], p[rare])),
        "accuracy_rare": float(accuracy_score(y[rare], pred[rare])),
        "auc_other": float(roc_auc_score(y[~rare], p[~rare])),
        "n_test_rare": int(rare.sum()),
    }


def rare_subgroup_fidelity(real: pd.DataFrame, syn: pd.DataFrame) -> dict:
    """How well the release keeps the rare subgroup: its size, its income rate and its conditional marginals."""
    col, val = RARE_GROUP
    r_rare, s_rare = real[real[col] == val], syn[syn[col] == val]
    r_maj, s_maj = real[real[col] == "White"], syn[syn[col] == "White"]

    def cond_tvd(r: pd.DataFrame, s: pd.DataFrame) -> float:
        if len(s) == 0:
            return 1.0
        a, b = discretize(r), discretize(s)
        sizes = [domain_sizes()[c] for c in COLUMNS]
        out = []
        for j, c in enumerate(COLUMNS):
            if c == col:
                continue
            fa = np.bincount(a[:, j], minlength=sizes[j]) / len(a)
            fb = np.bincount(b[:, j], minlength=sizes[j]) / len(b)
            out.append(0.5 * np.abs(fa - fb).sum())
        return float(np.mean(out))

    return {
        "real_share": float(len(r_rare) / len(real)),
        "release_share": float(len(s_rare) / max(len(syn), 1)),
        "real_positive_rate": float((r_rare[TARGET] == ">50K").mean()),
        "release_positive_rate": float((s_rare[TARGET] == ">50K").mean()) if len(s_rare) else float("nan"),
        "conditional_tvd_rare": cond_tvd(r_rare, s_rare),
        "conditional_tvd_majority": cond_tvd(r_maj, s_maj),
    }


# ----------------------------------------------------------------- risk
def dcr_membership_inference(train: pd.DataFrame, holdout: pd.DataFrame, release: pd.DataFrame, n: int, seed: int, transform=None) -> dict:
    """Distance-to-closest-record membership attack.

    The attacker scores a candidate as a member when its nearest released record is close (Gower distance).
    AUC 0.5 means members and non-members look the same to the attacker. ``transform`` (for SDC releases) maps
    a candidate into the release's generalised space before measuring distance.
    """
    rng = np.random.default_rng(seed)
    n = min(n, len(train), len(holdout))
    mem = train.iloc[rng.choice(len(train), n, replace=False)]
    non = holdout.iloc[rng.choice(len(holdout), n, replace=False)]
    if transform is not None:
        mem, non = transform(mem), transform(non)
    ref = release.iloc[rng.choice(len(release), min(len(release), 12000), replace=False)]
    d_mem, d_non = nearest_gower(mem, ref), nearest_gower(non, ref)
    score = -np.concatenate([d_mem, d_non])
    lab = np.concatenate([np.ones(n), np.zeros(n)])
    return {
        "auc": float(roc_auc_score(lab, score)),
        "exact_match_rate_members": float((d_mem == 0).mean()),
        "exact_match_rate_non_members": float((d_non == 0).mean()),
        "median_dcr_members": float(np.median(d_mem)),
        "median_dcr_non_members": float(np.median(d_non)),
        "n_each": int(n),
    }


def qi_uniqueness(train: pd.DataFrame, holdout: pd.DataFrame, release: pd.DataFrame) -> dict:
    """Share of released records whose quasi-identifier combination is *unique* in the confidential file.

    Compared with the same share for fresh holdout records (what chance alone produces), an excess means the
    release copies rare real combinations an intruder could match against a register.
    """
    key = lambda d: d[QUASI_IDENTIFIERS].astype(str).agg("|".join, axis=1)  # noqa: E731
    k_train = key(train)
    counts = k_train.value_counts()
    uniques = set(counts.index[counts == 1])
    rel = key(release).isin(uniques).mean()
    base = key(holdout).isin(uniques).mean()
    return {"release_rows_matching_train_unique": float(rel), "holdout_rows_matching_train_unique": float(base)}


def anonymeter_risks(train: pd.DataFrame, holdout: pd.DataFrame, release: pd.DataFrame, n_attacks: int, seed: int) -> dict:
    """Singling-out, linkability and attribute-inference risk (Giomi et al. 2023, Anonymeter).

    Risk = (attack success on the confidential file - success on the control file) / (1 - control success):
    0 means the release gives an attacker nothing beyond what a fresh sample of the population gives.
    The inference attack uses 4x more targets than the other two because its control success rate is high
    (most people have no capital gain), which makes its normalised risk noisy.
    """
    from anonymeter.evaluators import InferenceEvaluator, LinkabilityEvaluator, SinglingOutEvaluator

    np.random.seed(seed)  # linkability and inference draw targets with numpy's global RNG; singling-out takes a seed
    ori = train.sample(n=min(len(train), 10000), random_state=seed).reset_index(drop=True)
    ctrl = holdout.sample(n=min(len(holdout), 5000), random_state=seed).reset_index(drop=True)
    syn = release.sample(n=min(len(release), 10000), random_state=seed).reset_index(drop=True)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        so = SinglingOutEvaluator(ori=ori, syn=syn, control=ctrl, n_attacks=n_attacks, n_cols=4, max_attempts=20 * n_attacks, seed=seed)
        so.evaluate(mode="multivariate")
        r = so.risk()
        out["singling_out"] = {"risk": float(r.value), "ci": [float(r.ci[0]), float(r.ci[1])], "attack_rate": float(so.results().attack_rate.value)}
        aux = (["age", "sex", "race", "native_region", "marital_status"], ["education", "occupation", "workclass", "hours_per_week", "capital_gain"])
        lk = LinkabilityEvaluator(ori=ori, syn=syn, control=ctrl, n_attacks=n_attacks, aux_cols=aux, n_neighbors=10)
        lk.evaluate(n_jobs=1)
        r = lk.risk(n_neighbors=10)
        out["linkability"] = {
            "risk": float(r.value),
            "ci": [float(r.ci[0]), float(r.ci[1])],
            "attack_rate": float(lk.results(n_neighbors=10).attack_rate.value),
        }
        aux_inf = [c for c in COLUMNS if c not in ("income", "capital_gain")]
        inf = InferenceEvaluator(
            ori=ori, syn=syn, control=ctrl, aux_cols=aux_inf, secret="capital_gain", regression=False, n_attacks=min(4 * n_attacks, len(ctrl))
        )
        inf.evaluate(n_jobs=1)
        r = inf.risk()
        out["inference_capital_gain"] = {
            "risk": float(r.value),
            "ci": [float(r.ci[0]), float(r.ci[1])],
            "attack_rate": float(inf.results().attack_rate.value),
        }
    out["max_risk"] = max(v["risk"] for v in out.values() if isinstance(v, dict))
    out["max_risk_ci_high"] = max(v["ci"][1] for v in out.values() if isinstance(v, dict))
    return out
