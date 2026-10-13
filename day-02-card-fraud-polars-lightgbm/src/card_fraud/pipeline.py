"""End-to-end run: data -> features (Polars, benchmarked vs pandas) -> tuned LightGBM -> cost threshold -> SHAP."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import polars as pl
import shap
import sklearn
from sklearn.metrics import brier_score_loss, precision_recall_curve

from card_fraud import evaluate as ev
from card_fraud import explain, plots
from card_fraud import features_pandas as fpd
from card_fraud import features_polars as fpl
from card_fraud.config import DATA_DIR, FEATURES, SEED, Scale, load_cost_matrix
from card_fraud.model import downsample_negatives, fit_isotonic, fit_lgbm, fit_logreg, time_split, to_xy, tune_lgbm


def _log(msg: str) -> None:
    print(msg, flush=True)


def benchmark_features(raw: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Time the Polars lazy pipeline and the pandas baseline on the same in-memory data; check they agree."""
    raw_pd = raw.to_pandas()  # conversion is excluded from both timings
    t = time.perf_counter()
    feats = fpl.build_features(raw)
    t_pl = time.perf_counter() - t
    t = time.perf_counter()
    feats_pd = fpd.build_features(raw_pd)
    t_pd = time.perf_counter() - t
    max_rel = 0.0
    assert np.array_equal(feats["txn_id"].to_numpy(), feats_pd["txn_id"].to_numpy())
    for c in FEATURES:
        a = feats[c].cast(pl.Float64).to_numpy()
        b = feats_pd[c].astype("float64").to_numpy()
        if not np.array_equal(np.isnan(a), np.isnan(b)):
            raise AssertionError(f"null pattern differs for {c}")
        ok = ~np.isnan(a)
        if ok.any():
            max_rel = max(max_rel, float(np.max(np.abs(a[ok] - b[ok]) / np.maximum(1.0, np.abs(b[ok])))))
    del feats_pd, raw_pd
    return feats, {
        "rows": raw.height,
        "features": len(FEATURES),
        "polars_seconds": round(t_pl, 3),
        "pandas_seconds": round(t_pd, 3),
        "speedup": round(t_pd / t_pl, 2),
        "max_relative_difference": max_rel,
        "polars_version": pl.__version__,
        "pandas_version": pd.__version__,
    }


def run(
    raw: pl.DataFrame,
    manifest: dict,
    scale: Scale,
    results: Path,
    model_dir: Path = DATA_DIR / "models",
    gen_seconds: float = 0.0,
) -> dict:
    results.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    stage: dict[str, float] = {"generate_data": round(gen_seconds, 1)}
    cost = load_cost_matrix()

    # ---- features + benchmark ---------------------------------------------------------------------
    t = time.perf_counter()
    feats, bench = benchmark_features(raw)
    stage["features_and_benchmark"] = round(time.perf_counter() - t, 1)
    _log(f"[features] polars {bench['polars_seconds']}s vs pandas {bench['pandas_seconds']}s ({bench['speedup']}x)")
    plots.benchmark(bench, results / "benchmark_polars_vs_pandas.png")

    # ---- split -------------------------------------------------------------------------------------
    parts = time_split(feats)
    tr_ds = downsample_negatives(parts["train"], scale.neg_sample_rate)
    x_tr, y_tr = to_xy(tr_ds)
    x_va, y_va = to_xy(parts["val"])
    x_te, y_te = to_xy(parts["test"])
    amt_va, amt_te = parts["val"]["amount"].to_numpy(), parts["test"]["amount"].to_numpy()
    split_info = {
        name: {
            "from": str(df["ts"].min()),
            "to": str(df["ts"].max()),
            "rows": df.height,
            "fraud": int(df["is_fraud"].sum()),
            "fraud_rate": float(df["is_fraud"].mean()),
        }
        for name, df in parts.items()
    }
    split_info["train_after_negative_downsampling"] = {
        "rows": tr_ds.height,
        "fraud": int(y_tr.sum()),
        "neg_sample_rate": scale.neg_sample_rate,
    }
    assert parts["train"]["ts"].max() < parts["val"]["ts"].min() <= parts["val"]["ts"].max() < parts["test"]["ts"].min()

    # ---- tune + fit LightGBM ----------------------------------------------------------------------
    t = time.perf_counter()
    best, study = tune_lgbm(x_tr, y_tr, x_va, y_va, n_trials=scale.n_trials)
    stage["optuna_tuning"] = round(time.perf_counter() - t, 1)
    _log(f"[tune] {scale.n_trials} trials, best val PR-AUC {study.best_value:.4f} in {stage['optuna_tuning']}s")
    t = time.perf_counter()
    booster = fit_lgbm(x_tr, y_tr, x_va, y_va, best)
    stage["final_fit"] = round(time.perf_counter() - t, 1)
    s_va = booster.predict(x_va, num_iteration=booster.best_iteration)
    s_te = booster.predict(x_te, num_iteration=booster.best_iteration)
    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / "model_lgbm.txt"), num_iteration=booster.best_iteration)

    # ---- challenger -------------------------------------------------------------------------------
    t = time.perf_counter()
    lr = fit_logreg(x_tr, y_tr)
    stage["logreg_fit"] = round(time.perf_counter() - t, 1)
    l_va, l_te = lr.predict_proba(x_va)[:, 1], lr.predict_proba(x_te)[:, 1]

    rank = {
        "lightgbm": {"val": ev.ranking_metrics(y_va, s_va), "test": ev.ranking_metrics(y_te, s_te)},
        "logistic_regression": {"val": ev.ranking_metrics(y_va, l_va), "test": ev.ranking_metrics(y_te, l_te)},
    }
    _log(
        f"[test] PR-AUC LightGBM {rank['lightgbm']['test']['pr_auc']:.4f}; "
        f"LR {rank['logistic_regression']['test']['pr_auc']:.4f}"
    )
    plots.pr_curves(y_te, {"LightGBM (tuned)": s_te, "Logistic regression": l_te}, results / "pr_curve_test.png")

    # ---- cost-sensitive threshold (chosen on validation, applied to test) -------------------------
    sel = ev.select_threshold(y_va, s_va, amt_va, cost)
    thr = sel["threshold"]
    cc_te = ev.cost_curve(y_te, s_te, amt_te, cost)
    i_te = int(np.searchsorted(-cc_te["threshold"], -thr, side="right") - 1)
    p, r, th = precision_recall_curve(y_va, s_va)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(1e-12, p[:-1] + r[:-1])
    thr_f1 = float(th[int(np.argmax(f1))])
    lr_sel = ev.select_threshold(y_va, l_va, amt_va, cost)
    policies = {
        "approve_all": ev.apply_threshold(y_te, s_te, amt_te, cost, np.inf),
        "lightgbm_cost_optimal": ev.apply_threshold(y_te, s_te, amt_te, cost, thr),
        "lightgbm_threshold_0.5": ev.apply_threshold(y_te, s_te, amt_te, cost, 0.5),
        "lightgbm_max_f1": ev.apply_threshold(y_te, s_te, amt_te, cost, thr_f1),
        "lightgbm_cost_optimal_x0.5_alerts": None,
        "lightgbm_cost_optimal_x2_alerts": None,
        "logreg_cost_optimal": ev.apply_threshold(y_te, l_te, amt_te, cost, lr_sel["threshold"]),
    }
    # sensitivity: thresholds that alert half / double as many validation transactions
    va_sorted = np.sort(s_va)[::-1]
    for key, mult in (("lightgbm_cost_optimal_x0.5_alerts", 0.5), ("lightgbm_cost_optimal_x2_alerts", 2.0)):
        k = int(max(1, round(sel["n_alerts"] * mult)))
        policies[key] = ev.apply_threshold(y_te, s_te, amt_te, cost, float(va_sorted[k - 1]))
    for v in policies.values():
        v["savings_vs_approve_all"] = policies["approve_all"]["total_cost"] - v["total_cost"]
    threshold_info = {
        "selected_on": "validation (May 2025)",
        "threshold": thr,
        "val_total_cost_at_threshold": sel["total_cost"],
        "val_alerts": sel["n_alerts"],
        "val_total_cost_approve_all": float(sel["curve"]["total_cost"][0]),
        "max_f1_threshold_val": thr_f1,
        "logreg_threshold": lr_sel["threshold"],
        "test_cost_minimum_in_hindsight": float(cc_te["total_cost"].min()),
        "test_cost_at_selected": float(cc_te["total_cost"][i_te]),
    }
    _log(
        f"[cost] thr={thr:.4f}: test cost ${policies['lightgbm_cost_optimal']['total_cost']:,.0f} vs approve-all "
        f"${policies['approve_all']['total_cost']:,.0f}"
    )
    plots.cost_curve(
        {
            "validation (May)": {**sel["curve"], "n": len(y_va)},
            "test (June)": {**cc_te, "n": len(y_te)},
        },
        {
            "validation (May)": int(np.argmin(sel["curve"]["total_cost"])),
            "test (June)": i_te,
        },
        results / "expected_cost_curve.png",
    )

    # recall by attack type at the chosen threshold
    te = parts["test"].with_columns(score=pl.Series(s_te), alert=pl.Series(s_te >= thr))
    by_attack = (
        te.filter(pl.col("is_fraud") == 1)
        .group_by("attack_type")
        .agg(
            transactions=pl.len(),
            caught=pl.col("alert").sum(),
            fraud_usd=pl.col("amount").sum(),
            caught_usd=(pl.col("amount") * pl.col("alert")).sum(),
        )
        .with_columns(
            recall=pl.col("caught") / pl.col("transactions"), usd_recall=pl.col("caught_usd") / pl.col("fraud_usd")
        )
        .sort("attack_type")
    )
    by_attack.write_csv(results / "recall_by_attack_type.csv", float_precision=4)

    # ---- calibration (stretch) --------------------------------------------------------------------
    iso = fit_isotonic(s_va, y_va)
    c_te = iso.predict(s_te)
    calib = {
        "method": "isotonic regression fitted on validation scores, evaluated on test",
        "brier_raw": float(brier_score_loss(y_te, s_te)),
        "brier_isotonic": float(brier_score_loss(y_te, c_te)),
        "ece_raw": ev.expected_calibration_error(y_te, s_te),
        "ece_isotonic": ev.expected_calibration_error(y_te, c_te),
        "mean_score_raw": float(s_te.mean()),
        "mean_score_isotonic": float(c_te.mean()),
        "observed_fraud_rate": float(y_te.mean()),
    }
    plots.reliability(
        y_te, {"raw LightGBM score": s_te, "isotonic-calibrated": c_te}, results / "calibration_reliability.png"
    )

    # ---- SHAP ---------------------------------------------------------------------------------------
    t = time.perf_counter()
    rng = np.random.default_rng(SEED)
    pos = np.flatnonzero(y_te == 1)
    neg = rng.choice(np.flatnonzero(y_te == 0), size=min(len(pos) * 2, int((y_te == 0).sum())), replace=False)
    idx = np.sort(np.concatenate([pos, neg]))
    sv, base = explain.shap_values(booster, x_te[idx])
    # sanity: shap's TreeExplainer must agree with LightGBM's own contributions
    contrib = booster.predict(x_te[idx[:200]], num_iteration=booster.best_iteration, pred_contrib=True)
    shap_vs_native = float(np.max(np.abs(contrib[:, :-1] - sv[:200])))
    gi = explain.global_importance(sv)
    gi.write_csv(results / "shap_global_importance.csv", float_precision=5)
    plots.shap_beeswarm(sv, x_te[idx], list(FEATURES), results / "shap_summary_beeswarm.png")

    alert_idx = np.flatnonzero(s_te >= thr)
    sv_alert, _ = explain.shap_values(booster, x_te[alert_idx])
    codes = explain.reason_codes(sv_alert, x_te[alert_idx], k=3)
    te_alerts = parts["test"][alert_idx.tolist()]
    rc = pl.DataFrame(
        {
            "txn_id": te_alerts["txn_id"],
            "ts": te_alerts["ts"],
            "card_id": te_alerts["card_id"],
            "amount": te_alerts["amount"],
            "score": s_te[alert_idx].round(5),
            "is_fraud": te_alerts["is_fraud"],
            "attack_type": te_alerts["attack_type"],
            **{f"reason_{k + 1}": [c[k]["text"] for c in codes] for k in range(3)},
            **{f"reason_{k + 1}_feature": [c[k]["feature"] for c in codes] for k in range(3)},
        }
    ).sort("score", descending=True)
    rc.write_csv(results / "alert_reason_codes.csv")
    top1 = (
        rc.group_by("attack_type", "reason_1_feature")
        .agg(n=pl.len())
        .with_columns(share=pl.col("n") / pl.col("n").sum().over("attack_type"))
        .sort(["attack_type", "n"], descending=[False, True])
    )
    top1.write_csv(results / "reason_codes_by_attack_type.csv", float_precision=4)
    examples = {}
    for attack in ("card_testing", "account_takeover", "geo_impossible"):
        hit = np.flatnonzero(te_alerts["attack_type"].to_numpy() == attack)
        if len(hit) == 0:
            continue
        j = int(hit[np.argmax(s_te[alert_idx][hit])])
        examples[attack] = {
            "txn_id": int(te_alerts["txn_id"][j]),
            "score": float(s_te[alert_idx][j]),
            "reasons": [c["text"] for c in codes[j]],
        }
        plots.shap_waterfall(
            sv_alert[j],
            base,
            x_te[alert_idx][j],
            list(FEATURES),
            f"Alert txn {examples[attack]['txn_id']} ({attack}), score {examples[attack]['score']:.3f}",
            results / f"shap_local_{attack}.png",
        )
    stage["shap"] = round(time.perf_counter() - t, 1)

    trials = [
        {"number": tr.number, "value": tr.value, **tr.params, "best_iteration": tr.user_attrs.get("best_iteration")}
        for tr in study.trials
    ]
    pl.DataFrame(trials).write_csv(results / "optuna_trials.csv", float_precision=5)

    stage["pipeline_total_excluding_generation"] = round(time.perf_counter() - t_all, 1)
    metrics = {
        "data": {"synthetic": True, **manifest},
        "feature_benchmark": bench,
        "split": split_info,
        "tuning": {
            "sampler": "TPE (seed 42)",
            "n_trials": scale.n_trials,
            "objective": "validation PR-AUC (average precision)",
            "best_value": float(study.best_value),
            "best_params": best,
            "best_iteration": int(booster.best_iteration),
        },
        "ranking": rank,
        "cost_matrix": cost,
        "threshold": threshold_info,
        "policies_test": policies,
        "recall_by_attack_type_test": by_attack.to_dicts(),
        "calibration": calib,
        "explainability": {
            "shap_sample": {"fraud": int(len(pos)), "legit": int(len(neg))},
            "shap_vs_lightgbm_pred_contrib_max_abs_diff": shap_vs_native,
            "top_features": gi.head(8).to_dicts(),
            "alerts_explained": int(len(alert_idx)),
            "examples": examples,
        },
        "runtime_seconds": stage,
        "environment": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "optuna": optuna.__version__,
            "shap": shap.__version__,
            "scikit_learn": sklearn.__version__,
            "cpu_count": __import__("os").cpu_count(),
        },
    }
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    write_report(metrics, results / "report.md")
    return metrics


def _usd(x: float) -> str:
    return f"${x:,.0f}"


def write_report(m: dict, path: Path) -> None:
    """Human-readable cost table generated from metrics (so the README numbers are traceable)."""
    lines = [
        "# Day 02 results report (SYNTHETIC data)",
        "",
        f"Generated by `make run` from `results/metrics.json`. Test month: {m['split']['test']['from'][:10]} to "
        f"{m['split']['test']['to'][:10]}, {m['split']['test']['rows']:,} transactions, "
        f"{m['split']['test']['fraud']:,} fraudulent.",
        "",
        "## Decision policies on the test month (cost matrix: `configs/cost_matrix.json`)",
        "",
        "| Policy | Threshold | Alerts | Recall | Precision | FPR | Fraud $ caught | Total cost "
        "| Saving vs approve-all |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, p in m["policies_test"].items():
        thr = "none" if not np.isfinite(p["threshold"]) else f"{p['threshold']:.4f}"
        lines.append(
            f"| {name} | {thr} | {p['alerts']:,} | {p['recall']:.1%} | {p['precision']:.1%} | {p['fpr']:.3%} | "
            f"{p['fraud_usd_recall']:.1%} | {_usd(p['total_cost'])} | {_usd(p['savings_vs_approve_all'])} |"
        )
    pc = m["policies_test"]["lightgbm_cost_optimal"]
    lines += [
        "",
        "## Cost breakdown at the selected threshold",
        "",
        f"- Missed fraud (FN): {pc['fn']:,} transactions, {_usd(pc['false_negative_cost'])}",
        f"- False declines (FP): {pc['fp']:,} transactions, {_usd(pc['false_positive_cost'])}",
        f"- Correct declines (TP handling): {pc['tp']:,} transactions, {_usd(pc['true_positive_cost'])}",
        f"- Total: {_usd(pc['total_cost'])}",
        "",
        "The best threshold in hindsight on the test month would cost "
        f"{_usd(m['threshold']['test_cost_minimum_in_hindsight'])}; "
        f"the validation-chosen threshold costs {_usd(m['threshold']['test_cost_at_selected'])}.",
        "",
    ]
    path.write_text("\n".join(lines))
