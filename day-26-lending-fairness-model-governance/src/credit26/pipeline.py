"""End-to-end run: data -> scorecard vs GBM -> fairness audit -> proxy analysis -> LDA search -> mitigation -> explanations -> documents."""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from . import explain, fairness, mitigation
from .data import CreditData, DatasetSpec, load_data, split_data
from .models import GBMConfig, TrainedGBM, cutoff_for_approval, performance, psi, psi_band, psi_categorical, train_gbm
from .scorecard import Scorecard


@dataclass
class RunConfig:
    data_mode: str = "auto"
    n: int = 40_000
    seed: int = 20261105
    approval_rate: float = 0.70
    n_boot: int = 1000
    auc_tolerance: float = 0.02
    rw_alphas: list[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    eg_eps: list[float] = field(default_factory=lambda: [0.10, 0.05, 0.03, 0.02, 0.01])
    eg_max_iter: int = 30
    cf_max: int | None = None  # None = every declined test applicant
    cf_K: int = 12
    mono_check_n: int = 200


def small_config() -> RunConfig:
    return RunConfig(data_mode="synthetic", n=6000, n_boot=100, rw_alphas=[1.0], eg_eps=[0.05], eg_max_iter=8, cf_max=40, cf_K=6, mono_check_n=30)


def _r(x, k=4):
    if isinstance(x, dict):
        return {a: _r(b, k) for a, b in x.items()}
    if isinstance(x, list | tuple):
        return [_r(v, k) for v in x]
    if isinstance(x, float | np.floating):
        return round(float(x), k)
    if isinstance(x, np.integer):
        return int(x)
    return x


@dataclass
class Evaluated:
    name: str
    family: str
    p_bad_test: np.ndarray | None
    approve_test: np.ndarray
    approve_valid: np.ndarray
    p_bad_valid: np.ndarray | None
    cutoff: float | None
    uses_protected_at_decision: bool = False
    explainable_single_model: bool = False
    model: TrainedGBM | None = None
    features: list[str] = field(default_factory=list)


def evaluate(e: Evaluated, valid: pd.DataFrame, test: pd.DataFrame, spec: DatasetSpec, cfg: RunConfig, full_audit: bool = True) -> dict:
    yt, yv = test["bad"].to_numpy(), valid["bad"].to_numpy()
    out: dict = {"family": e.family, "uses_protected_attribute_at_decision": e.uses_protected_at_decision, "features": e.features}
    if e.p_bad_test is not None:
        out["test"] = performance(yt, e.p_bad_test, e.approve_test)
        out["valid_auc"] = float(roc_auc_score(yv, e.p_bad_valid))
    else:
        out["test"] = {"approval_rate": float(e.approve_test.mean()), "bad_rate_approved": float(yt[e.approve_test == 1].mean())}
        out["valid_auc"] = None
    out["test"]["balanced_accuracy"] = float(balanced_accuracy_score(1 - yt, e.approve_test))
    if spec.has_unbiased_label:
        yu = test["bad_unbiased"].to_numpy()
        if e.p_bad_test is not None:
            out["test"]["auc_vs_unbiased_label"] = float(roc_auc_score(yu, e.p_bad_test))
        out["test"]["balanced_accuracy_vs_unbiased_label"] = float(balanced_accuracy_score(1 - yu, e.approve_test))
        out["test"]["unbiased_bad_rate_approved"] = float(yu[e.approve_test == 1].mean())
    g = test[spec.protected].astype(str).to_numpy()
    out["fairness"] = fairness.audit(e.approve_test, 1 - yt, g, spec.reference_group, n_boot=cfg.n_boot if full_audit else 200, seed=cfg.seed)
    vg = valid[spec.protected].astype(str).to_numpy()
    out["valid_air_min"] = fairness.fairness_point(e.approve_valid, 1 - yv, vg, spec.reference_group)["air_min"]
    sec = {}
    for s in spec.secondary_protected:
        levels = test[s].astype(str)
        ref = levels.value_counts().idxmax()
        pt = fairness.fairness_point(e.approve_test, 1 - yt, levels.to_numpy(), ref)
        sec[s] = {"reference": ref, "air": pt["air"], "approval_rate": pt["approval_rate"]}
    out["fairness_secondary"] = sec
    return out


def _gbm_eval(name: str, family: str, m: TrainedGBM, valid: pd.DataFrame, test: pd.DataFrame, rate: float) -> Evaluated:
    pv, pt = m.p_bad(valid), m.p_bad(test)
    cut = cutoff_for_approval(pv, rate)
    return Evaluated(name, family, pt, (pt <= cut).astype(int), (pv <= cut).astype(int), pv, cut, False, True, m, m.features)


def monotonicity_check(m: TrainedGBM, df: pd.DataFrame, spec: DatasetSpec, n: int, seed: int) -> dict[str, float]:
    """Share of sampled applicants whose p_bad moves only in the business-expected direction along a 15-point grid of each feature."""
    s = df.sample(n=min(n, len(df)), random_state=seed)
    out = {}
    for f, d in spec.monotone.items():
        if f not in m.features or d == 0:
            continue
        grid = np.quantile(df[f].to_numpy(dtype=float), np.linspace(0.01, 0.99, 15))
        big = s.loc[s.index.repeat(len(grid))].copy()
        big[f] = np.tile(grid, len(s))
        p = m.p_bad(big).reshape(len(s), len(grid))
        diffs = np.diff(p, axis=1) * d
        out[f] = float(np.mean(np.all(diffs >= -1e-12, axis=1)))
    return out


def run(cfg: RunConfig, out_dir: Path, data_dir: Path, sample_dir: Path | None = None, log: Callable[[str], None] = print) -> dict:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    def lap(k: str) -> None:
        timings[k] = round(time.perf_counter() - t0 - sum(timings.values()), 2)

    # ---------------------------------------------------------------- data
    data: CreditData = load_data(cfg.data_mode, n=cfg.n, seed=cfg.seed, raw_dir=data_dir / "raw")
    spec = data.spec
    log(f"data path: {data.path_used} ({len(data.df):,} rows); attempts: {[(a['source'], a['ok']) for a in data.attempts]}")
    sp = split_data(data, seed=cfg.seed)
    train, valid, test = sp.train, sp.valid, sp.test
    if sample_dir is not None:
        sample_dir.mkdir(parents=True, exist_ok=True)
        tag = "SYNTHETIC" if data.path_used == "synthetic_generator" else "public"
        data.df.head(3000).to_csv(sample_dir / f"applications_{tag}_sample.csv", index=False)
    df = data.df
    dsum = {
        "path_used": data.path_used,
        "attempts": data.attempts,
        "synthetic": data.path_used == "synthetic_generator",
        "rows": len(df),
        "split_scheme": sp.scheme,
        "rows_train_valid_test": [len(train), len(valid), len(test)],
        "protected_attribute": spec.protected,
        "reference_group": spec.reference_group,
        "group_share": df[spec.protected].astype(str).value_counts(normalize=True).sort_index().to_dict(),
        "bad_rate": float(df["bad"].mean()),
        "bad_rate_by_group": df.groupby(spec.protected)["bad"].mean().to_dict(),
    }
    if spec.has_unbiased_label:
        dsum["bad_rate_unbiased"] = float(df["bad_unbiased"].mean())
        dsum["bad_rate_unbiased_by_group"] = df.groupby(spec.protected)["bad_unbiased"].mean().to_dict()
        dsum["label_bias_share_of_recorded_bads"] = float(1 - df["bad_unbiased"].sum() / df["bad"].sum())
    lap("data")

    # ---------------------------------------------------------------- baseline scorecard vs GBM
    rate = cfg.approval_rate
    sc = Scorecard(spec.numeric, spec.categorical).fit(train, train["bad"].to_numpy())
    pv, pt = sc.predict_proba_bad(valid), sc.predict_proba_bad(test)
    cut = cutoff_for_approval(pv, rate)
    evs: dict[str, Evaluated] = {
        "scorecard_woe_lr": Evaluated(
            "scorecard_woe_lr", "baseline", pt, (pt <= cut).astype(int), (pv <= cut).astype(int), pv, cut, features=spec.features
        )
    }
    gbm = train_gbm("gbm", train, spec.features, spec, GBMConfig(monotone=False, seed=cfg.seed % 1000))
    gbm_mono = train_gbm("gbm_monotone", train, spec.features, spec, GBMConfig(monotone=True, seed=cfg.seed % 1000))
    evs["gbm"] = _gbm_eval("gbm", "challenger", gbm, valid, test, rate)
    evs["gbm_monotone"] = _gbm_eval("gbm_monotone", "candidate", gbm_mono, valid, test, rate)
    log("trained scorecard, GBM, monotone GBM")

    stability = {}
    for k, m_sc in [("scorecard_woe_lr", sc.predict_proba_bad(train)), ("gbm", gbm.p_bad(train)), ("gbm_monotone", gbm_mono.p_bad(train))]:
        stability[k] = {"score_psi_train_vs_valid": psi(m_sc, evs[k].p_bad_valid), "score_psi_train_vs_test": psi(m_sc, evs[k].p_bad_test)}
    feat_psi = {
        f: (psi_categorical(train[f], test[f]) if f in spec.categorical else psi(train[f].to_numpy(float), test[f].to_numpy(float)))
        for f in spec.features
    }
    stability["feature_psi_train_vs_test"] = feat_psi
    stability["feature_psi_band"] = {f: psi_band(v) for f, v in feat_psi.items()}
    mono_chk = {
        "gbm": monotonicity_check(gbm, test, spec, cfg.mono_check_n, cfg.seed),
        "gbm_monotone": monotonicity_check(gbm_mono, test, spec, cfg.mono_check_n, cfg.seed),
    }
    scorecard_info = {
        "iv": sc.binner.iv,
        "coef": dict(zip(sc.features_, sc.lr.coef_[0].tolist())),
        "pdo": sc.pdo,
        "base_score": sc.base_score,
        "base_odds": sc.base_odds,
    }
    lap("models")

    # ---------------------------------------------------------------- proxy analysis
    prox = {
        "single_feature_auc": fairness.proxy_strength(train, spec.features, spec.categorical, spec.protected, spec.reference_group, cfg.seed % 1000),
        "joint_auc_all_features": fairness.joint_proxy_auc(
            train, spec.features, spec.categorical, spec.protected, spec.reference_group, cfg.seed % 1000
        ),
    }
    no_proxy = [f for f in spec.features if f not in spec.categorical] if spec.categorical else spec.features
    prox["joint_auc_without_categorical_proxies"] = fairness.joint_proxy_auc(
        train, no_proxy, spec.categorical, spec.protected, spec.reference_group, cfg.seed % 1000
    )
    gt = (test[spec.protected].astype(str) != spec.reference_group).astype(int)
    prox["candidate_score_group_separation_auc"] = float(roc_auc_score(gt, evs["gbm_monotone"].p_bad_test))
    sv_c, _ = explain.shap_values(gbm_mono, test)
    mean_abs = np.abs(sv_c).mean(axis=0)
    prox["candidate_mean_abs_shap"] = dict(zip(gbm_mono.features, mean_abs.tolist()))
    prox["candidate_shap_share"] = dict(zip(gbm_mono.features, (mean_abs / mean_abs.sum()).tolist()))
    lap("proxy")
    log("proxy analysis done")

    # ---------------------------------------------------------------- less-discriminatory-alternative search (stretch)
    feature_sets = {"all": spec.features}
    if spec.categorical:
        feature_sets["no_geography"] = no_proxy
    strong = [f for f, a in sorted(prox["single_feature_auc"].items(), key=lambda kv: -kv[1]) if f not in spec.categorical and a >= 0.60]
    for f in strong[:2]:
        feature_sets[f"no_geography_no_{f}"] = [x for x in no_proxy if x != f]
    lda = {}
    for fs_name, feats in feature_sets.items():
        for leaves in (7, 15):
            nm = f"lda_{fs_name}_mono_l{leaves}"
            m = train_gbm(nm, train, feats, spec, GBMConfig(monotone=True, num_leaves=leaves, seed=cfg.seed % 1000))
            e = _gbm_eval(nm, "lda_candidate", m, valid, test, rate)
            evs[nm] = e
            lda[nm] = {
                "feature_set": fs_name,
                "num_leaves": leaves,
                "valid_auc": float(roc_auc_score(valid["bad"], e.p_bad_valid)),
                "valid_air_min": None,
            }
    ref_auc = float(roc_auc_score(valid["bad"], evs["gbm_monotone"].p_bad_valid))
    for nm in lda:
        yv = valid["bad"].to_numpy()
        lda[nm]["valid_air_min"] = fairness.fairness_point(
            evs[nm].approve_valid, 1 - yv, valid[spec.protected].astype(str).to_numpy(), spec.reference_group
        )["air_min"]
    within = {k: v for k, v in lda.items() if v["valid_auc"] >= ref_auc - cfg.auc_tolerance}
    lda_pick = max(within, key=lambda k: (within[k]["valid_air_min"], within[k]["valid_auc"])) if within else "gbm_monotone"
    base_feats = evs[lda_pick].features
    log(f"LDA search: {len(lda)} candidates, selected {lda_pick}")
    lap("lda")

    # ---------------------------------------------------------------- mitigation on the LDA-selected feature set
    base_leaves = lda[lda_pick]["num_leaves"] if lda_pick in lda else 15
    for a in cfg.rw_alphas:
        w = mitigation.reweighing_weights(train[spec.protected].astype(str).to_numpy(), train["bad"].to_numpy(), a)
        nm = f"reweighing_a{a:g}"
        m = train_gbm(nm, train, base_feats, spec, GBMConfig(monotone=True, num_leaves=base_leaves, seed=cfg.seed % 1000), sample_weight=w)
        evs[nm] = _gbm_eval(nm, "reweighing", m, valid, test, rate)
    base_model = evs[lda_pick].model
    for cons, nm in [("demographic_parity", "threshold_opt_dp"), ("equalized_odds", "threshold_opt_eo")]:
        to = mitigation.threshold_optimizer(base_model.p_bad, valid, spec.protected, cons, seed=cfg.seed % 1000)
        evs[nm] = Evaluated(
            nm,
            "threshold_optimizer",
            None,
            mitigation.to_approve(to, test, spec.protected),
            mitigation.to_approve(to, valid, spec.protected),
            None,
            None,
            True,
            False,
            None,
            base_feats,
        )
    for eps in cfg.eg_eps:
        nm = f"exp_gradient_dp{eps:g}"
        c = evs[lda_pick].cutoff
        eg = mitigation.exponentiated_gradient(
            train, base_feats, spec.protected, eps, seed=cfg.seed % 1000, max_iter=cfg.eg_max_iter, fp_cost=(1 - c) / c
        )
        ptest, pval = mitigation.eg_p_bad(eg, test, base_feats), mitigation.eg_p_bad(eg, valid, base_feats)
        ap_t = np.asarray(eg.predict(mitigation.numeric_matrix(test, base_feats), random_state=cfg.seed % 1000)).astype(int)
        ap_v = np.asarray(eg.predict(mitigation.numeric_matrix(valid, base_feats), random_state=cfg.seed % 1000)).astype(int)
        evs[nm] = Evaluated(nm, "exponentiated_gradient", ptest, ap_t, ap_v, pval, None, False, False, None, base_feats)
        log(f"exponentiated gradient eps={eps:g}: {len(eg.predictors_)} predictors")
    lap("mitigation")

    results = {k: evaluate(e, valid, test, spec, cfg) for k, e in evs.items()}
    for k in lda:
        results[k]["lda"] = lda[k]
    log("evaluated all models")
    lap("evaluation")

    # ---------------------------------------------------------------- recommended model (selection on validation only)
    eligible = {
        k: r
        for k, r in results.items()
        if evs[k].explainable_single_model and r["valid_auc"] is not None and r["valid_auc"] >= ref_auc - cfg.auc_tolerance
    }
    passing = {k: r for k, r in eligible.items() if r["valid_air_min"] >= fairness.AIR_THRESHOLD}
    if passing:
        rec = max(passing, key=lambda k: passing[k]["valid_auc"])
        rule = "highest validation AUC among single, explainable, monotone GBMs within the AUC tolerance whose validation AIR >= 0.80"
    else:
        rec = max(eligible, key=lambda k: eligible[k]["valid_air_min"])
        rule = "no eligible model reached validation AIR >= 0.80; picked the highest validation AIR within the AUC tolerance"
    rm = evs[rec].model
    log(f"recommended: {rec}")

    # ---------------------------------------------------------------- explanations for the recommended model
    sv, base_val = explain.shap_values(rm, test)
    raw = rm.model.predict(rm.X(test), raw_score=True)
    additivity = float(np.max(np.abs(sv.sum(axis=1) + base_val - raw)))
    dec_mask = evs[rec].approve_test == 0
    declined = test[dec_mask].reset_index(drop=True)
    names, vals = explain.reason_codes(sv[dec_mask], rm.features, k=4)
    rframe = explain.reason_frame(declined, names, vals, spec, rm.features)
    rframe.to_csv(out_dir / "reason_codes_declined_test.csv", index=False)
    n_pos = (sv[dec_mask] > 0).sum(axis=1)
    reason_freq = pd.Series(names[:, 0]).value_counts().to_dict()
    reason_any = pd.Series(names.ravel()).value_counts().to_dict()
    sc_dec = (evs["scorecard_woe_lr"].approve_test == 0)[dec_mask]
    sc_names = sc.reason_codes(declined, k=4)
    both = sc_dec
    top1_agree = float(np.mean(sc_names[both, 0] == names[both, 0])) if both.any() else float("nan")
    overlap = float(np.mean([len(set(a) & set(b)) / 4 for a, b in zip(sc_names[both], names[both])])) if both.any() else float("nan")
    expl = {
        "model": rec,
        "declined_test_applicants": int(dec_mask.sum()),
        "declined_with_4_reason_codes": int((rframe[[f"reason_{j}_code" for j in range(1, 5)]].notna().all(axis=1)).sum()),
        "share_declined_with_4_positive_contributions": float(np.mean(n_pos >= 4)),
        "shap_additivity_max_abs_error": additivity,
        "shap_base_value_logodds": base_val,
        "top_reason_frequency": {k: int(v) for k, v in reason_freq.items()},
        "any_of_top4_frequency": {k: int(v) for k, v in reason_any.items()},
        "mean_abs_shap": dict(zip(rm.features, np.abs(sv).mean(axis=0).tolist())),
        "scorecard_vs_shap_declined_by_both": int(both.sum()),
        "scorecard_vs_shap_top1_agreement": top1_agree,
        "scorecard_vs_shap_top4_overlap": overlap,
        "reason_code_table": {explain.code_table(spec, rm.features)[f]: spec.reason_text.get(f, f) for f in rm.features},
    }
    lap("reason_codes")
    q = train[list(spec.actionable)].quantile([0.25, 0.75])
    scale = {f: float(max(q.loc[0.75, f] - q.loc[0.25, f], 1e-6)) for f in spec.actionable}
    cf_in = declined if cfg.cf_max is None else declined.head(cfg.cf_max)
    cf = explain.counterfactuals(rm, cf_in, evs[rec].cutoff, spec, scale, K=cfg.cf_K)
    cf.to_csv(out_dir / "counterfactuals_declined_test.csv", index=False)
    fnd = cf[cf["found"]]
    feats_used = pd.Series([c.split(":")[0] for s in fnd["changes"] for c in s.split("; ")]).value_counts().to_dict() if len(fnd) else {}
    expl["counterfactual"] = {
        "applicants_searched": len(cf),
        "found_share": float(cf["found"].mean()),
        "single_feature_share_of_found": float((fnd["n_changed"] == 1).mean()) if len(fnd) else float("nan"),
        "median_cost_iqr_units": float(fnd["cost"].median()) if len(fnd) else float("nan"),
        "feature_used_count": {k: int(v) for k, v in feats_used.items()},
        "actionable_features": {f: list(v) for f, v in spec.actionable.items()},
        "grid_steps": cfg.cf_K,
        "iqr_scale": scale,
    }
    explain_examples = []
    for i in range(min(3, len(declined))):
        row = declined.iloc[i]
        c = cf[cf["application_id"] == row["application_id"]].iloc[0]
        explain_examples.append(
            {
                "application_id": int(row["application_id"]),
                "p_bad": float(evs[rec].p_bad_test[dec_mask][i]),
                "reasons": [
                    (rframe.iloc[i][f"reason_{j}_code"], spec.reason_text.get(names[i, j - 1], names[i, j - 1]), float(vals[i, j - 1]))
                    for j in range(1, 5)
                ],
                "inputs": {f: (float(row[f]) if f not in spec.categorical else str(row[f])) for f in rm.features},
                "counterfactual": c["changes"] if c["found"] else None,
            }
        )
    expl["examples"] = explain_examples
    lap("counterfactuals")
    log(
        f"explanations: {expl['declined_with_4_reason_codes']}/{expl['declined_test_applicants']} declined with 4 reasons; CF found {expl['counterfactual']['found_share']:.1%}"
    )

    # ---------------------------------------------------------------- monitoring evidence for the recommended model
    rec_stab = {
        "score_psi_train_vs_test": psi(rm.p_bad(train), evs[rec].p_bad_test),
        "score_psi_band": psi_band(psi(rm.p_bad(train), evs[rec].p_bad_test)),
    }
    monthly = []
    if spec.time_col:
        for mth, idx in test.groupby(spec.time_col).groups.items():
            idx = np.asarray(list(idx))
            ap = evs[rec].approve_test[idx]
            ap_c = evs["gbm_monotone"].approve_test[idx]
            g = test.loc[idx, spec.protected].astype(str).to_numpy()
            yb = 1 - test.loc[idx, "bad"].to_numpy()
            monthly.append(
                {
                    "month": int(mth),
                    "n": len(idx),
                    "approval_rate": float(ap.mean()),
                    "air_min_recommended": fairness.fairness_point(ap, yb, g, spec.reference_group)["air_min"],
                    "air_min_candidate": fairness.fairness_point(ap_c, yb, g, spec.reference_group)["air_min"],
                    "score_psi_vs_train": psi(rm.p_bad(train), evs[rec].p_bad_test[idx]),
                }
            )
    rec_stab["monthly"] = monthly
    lap("monitoring")

    metrics = {
        "config": asdict(cfg),
        "environment": {"python": platform.python_version()},
        "data": dsum,
        "models": results,
        "stability": stability,
        "monotonicity_check": mono_chk,
        "scorecard": scorecard_info,
        "proxy_analysis": prox,
        "lda_search": {"reference_valid_auc": ref_auc, "auc_tolerance": cfg.auc_tolerance, "candidates": lda, "selected": lda_pick},
        "recommended": {"model": rec, "rule": rule, "cutoff_p_bad": evs[rec].cutoff, "features": rm.features, "stability": rec_stab},
        "explanations": expl,
    }
    metrics = _r(metrics, 4)
    timings["total_seconds"] = round(time.perf_counter() - t0, 1)
    metrics["timings"] = timings
    from . import figures, governance

    figures.make_all(metrics, evs, test, spec, out_dir / "figures")
    governance.write_all(metrics, out_dir)
    metrics["timings"]["total_seconds"] = round(time.perf_counter() - t0, 1)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics
