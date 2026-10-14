"""End-to-end run: validate -> normalise to OMOP -> SQL cohort -> models -> evaluation -> results/."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import plots
from .config import FOLLOWUP_DAYS, SEED, TRAIN_END
from .db import build_cohort, connect, load_omop
from .evaluate import (
    binary_metrics,
    calibration_bins,
    cluster_bootstrap,
    decision_curve,
    kaplan_meier,
    subgroup_table,
)
from .generate import read_lines
from .model import MODELS, TARGET, logistic_coefficients, make_lace, oof_predictions, prepare
from .normalise import normalise as to_omop
from .normalise import validate_all

THRESHOLDS = np.round(np.arange(0.02, 0.51, 0.01), 2)


def cohort_fingerprint(feats: pd.DataFrame) -> str:
    """SHA-256 over (visit id, outcome) of the cohort: equal fingerprints = identical cohort."""
    key = feats[["visit_occurrence_id", TARGET]].astype(int).to_csv(index=False).encode()
    return hashlib.sha256(key).hexdigest()


def build(data_dir: Path, db_path: Path | str = ":memory:") -> dict:
    """Validate, normalise and build the cohort. Returns intermediate objects for run() and tests."""
    t0 = time.perf_counter()
    lines = read_lines(data_dir)
    valid, report = validate_all(lines)
    t_val = time.perf_counter() - t0
    frames = to_omop(valid)
    t_norm = time.perf_counter() - t0 - t_val
    con = connect(db_path)
    load_omop(con, frames)
    attrition, feats = build_cohort(con)
    keys = con.execute(
        """SELECT v.visit_occurrence_id, p.person_source_value || '|' || v.visit_source_value AS key
           FROM visit_occurrence v JOIN person p USING (person_id)"""
    ).df()
    con.close()
    return {
        "report": report,
        "frames": frames,
        "attrition": attrition,
        "features": feats,
        "keys": keys,
        "timings": {"validate_s": t_val, "normalise_s": t_norm, "sql_s": time.perf_counter() - t0 - t_val - t_norm},
    }


def _validation_outputs(report, manifest: dict, results: Path) -> dict:
    injected = manifest.get("corrupted_bundles", {})
    rej = pd.DataFrame(report.rejected, columns=["source_file", "bundle_id", "n_errors", "first_error", "errors"])
    rej.insert(2, "injected_defect", rej["bundle_id"].map(injected).fillna("none (not injected)"))
    rej.sort_values(["injected_defect", "bundle_id"]).to_csv(results / "validation_rejections.csv", index=False)
    rejected_ids = set(rej["bundle_id"])
    caught = sum(b in rejected_ids for b in injected)
    by_type = {}
    for kind in sorted(set(injected.values())):
        ids = [b for b, k in injected.items() if k == kind]
        by_type[kind] = {"injected": len(ids), "rejected": sum(b in rejected_ids for b in ids)}
    return {
        **report.summary(),
        "injected_defects": len(injected),
        "injected_defects_rejected": caught,
        "rejected_without_injected_defect": int(sum(b not in injected for b in rejected_ids)),
        "by_defect_type": by_type,
    }


def run(data_dir: Path, results: Path, manifest: dict, truth: pd.DataFrame | None = None, n_boot: int = 500) -> dict:
    t_start = time.perf_counter()
    results.mkdir(parents=True, exist_ok=True)
    b = build(data_dir)
    validation = _validation_outputs(b["report"], manifest, results)
    print(f"[validate] {validation['bundles_valid']:,}/{validation['bundles_read']:,} bundles valid", flush=True)
    attrition = b["attrition"]
    attrition.to_csv(results / "cohort_attrition.csv", index=False)
    feats = b["features"]
    print(f"[cohort] {len(feats):,} index admissions, {feats[TARGET].mean():.3%} readmitted", flush=True)

    # ---- temporal split ----
    train_mask = feats["discharge_date"] <= pd.Timestamp(TRAIN_END)
    tr, te = feats[train_mask].reset_index(drop=True), feats[~train_mask].reset_index(drop=True)
    Xtr, Xte = prepare(tr), prepare(te)
    ytr, yte = tr[TARGET].to_numpy(int), te[TARGET].to_numpy(int)
    gtr = tr["person_id"].to_numpy()

    # ---- model selection on training-period out-of-fold predictions ----
    oof = {name: oof_predictions(name, Xtr, ytr, gtr, seed=SEED) for name in MODELS}
    oof_metrics = {name: binary_metrics(ytr, p) for name, p in oof.items()}
    primary = min(oof_metrics, key=lambda k: oof_metrics[k]["log_loss"])
    print(f"[select] primary model by OOF log loss: {primary}", flush=True)

    # ---- final fits on the whole training period ----
    fitted, preds = {}, {}
    for name, factory in MODELS.items():
        m = factory(SEED)
        m.fit(Xtr, ytr)
        fitted[name] = m
        preds[name] = m.predict_proba(Xte)[:, 1]
    lace = make_lace().fit(tr[["lace"]].to_numpy(float), ytr)
    preds["lace"] = lace.predict_proba(te[["lace"]].to_numpy(float))[:, 1]

    test_metrics = {name: binary_metrics(yte, p) for name, p in preds.items()}
    gte = te["person_id"].to_numpy()
    for name, p in preds.items():
        test_metrics[name]["ci95"] = cluster_bootstrap(yte, p, gte, n_boot=n_boot, seed=SEED)
    test_metrics["lace"]["raw_score_auroc"] = float(roc_auc_score(yte, te["lace"]))

    # ---- calibration / DCA / subgroups / survival ----
    bins = {name: calibration_bins(yte, p) for name, p in preds.items()}
    pd.concat([b_.assign(model=n) for n, b_ in bins.items()]).to_csv(results / "calibration_bins.csv", index=False)
    dca = decision_curve(yte, preds, THRESHOLDS)
    dca.to_csv(results / "decision_curve.csv", index=False)
    te_eval = te.assign(p=preds[primary])
    sub = subgroup_table(te_eval, TARGET, "p")
    sub.to_csv(results / "subgroup_performance.csv", index=False)

    q = pd.qcut(pd.Series(preds[primary]).rank(method="first"), 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    t_evt = te["days_to_unplanned"].fillna(FOLLOWUP_DAYS).to_numpy(int)
    km = {
        str(lev): kaplan_meier(t_evt[q == lev], yte[q == lev].astype(bool), FOLLOWUP_DAYS) for lev in q.cat.categories
    }
    pd.concat([c.assign(quartile=k) for k, c in km.items()]).to_csv(results / "km_by_risk_quartile.csv", index=False)
    km_summary = {
        k: {
            "n": int(c["at_risk"].iloc[0]),
            "mean_predicted": float(preds[primary][(q == k).to_numpy()].mean()),
            "cum_incidence_day7": float(c.loc[c["day"] == 7, "cumulative_incidence"].iloc[0]),
            "cum_incidence_day30": float(c["cumulative_incidence"].iloc[-1]),
        }
        for k, c in km.items()
    }
    readmitted = te.loc[te[TARGET], "days_to_unplanned"]
    timing = {
        "median_days_to_readmission": float(readmitted.median()),
        "share_within_7_days": float((readmitted <= 7).mean()),
    }

    coefs = logistic_coefficients(fitted["logistic_l2"])
    coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index).to_csv(
        results / "logistic_coefficients.csv", index=False
    )

    # ---- decision-curve summary at clinically used thresholds ----
    dca_summary = {}
    for pt in (0.10, 0.15, 0.20, 0.30):
        row = dca.loc[np.isclose(dca["threshold"], pt)].iloc[0]
        p = preds[primary]
        flagged = p >= pt
        dca_summary[f"{pt:.2f}"] = {
            "net_benefit": {k: float(row[k]) for k in [*preds, "treat_all"]},
            "primary_flagged_share": float(flagged.mean()),
            "primary_sensitivity": float(flagged[yte == 1].mean()),
            "primary_ppv": float(yte[flagged].mean()) if flagged.any() else float("nan"),
        }
    useful = dca[primary] > np.maximum(dca["treat_all"], 0) + 1e-12
    dca_coverage = {
        "thresholds_evaluated": [float(THRESHOLDS[0]), float(THRESHOLDS[-1])],
        "n_thresholds": int(len(THRESHOLDS)),
        "n_where_primary_beats_treat_all_and_none": int(useful.sum()),
        "thresholds_where_it_does_not": [float(t) for t in dca.loc[~useful, "threshold"]],
    }

    oracle = None
    if truth is not None:
        t = te[["visit_occurrence_id"]].merge(b["keys"], on="visit_occurrence_id").merge(truth, on="key", how="left")
        if t["p_true"].notna().all():
            oracle = {
                "auroc_generator_hazard": float(roc_auc_score(yte, t["p_true"])),
                "note": "AUROC of the generator's own readmission probability on the test cohort (ceiling).",
            }

    # ---- figures ----
    plots.calibration_plot(bins, preds, results / "calibration.png", "Calibration on the 2024 test period")
    plots.decision_curve_plot(dca, list(preds), results / "decision_curve.png", "Decision curve (2024 test period)")
    plots.roc_plot(yte, preds, {k: v["auroc"] for k, v in test_metrics.items()}, results / "roc.png")
    plots.subgroup_plot(sub, results / "subgroup_performance.png", f"Subgroups, {plots.LABELS[primary]}")
    plots.km_plot(km, results / "km_by_risk_quartile.png")

    omop_counts = {k: int(len(v)) for k, v in b["frames"].items()}
    metrics = {
        "synthetic_data": True,
        "data_path_used": manifest.get("data_path_used", "synthetic_generator"),
        "scale": manifest.get("scale"),
        "n_patients_generated": manifest.get("n_patients"),
        "validation": validation,
        "omop_row_counts": omop_counts,
        "cohort": {
            "attrition": attrition.to_dict(orient="records"),
            "n_index_admissions": int(len(feats)),
            "n_patients": int(feats["person_id"].nunique()),
            "readmission_rate": float(feats[TARGET].mean()),
            "planned_readmissions_within_30d_not_counted": int(feats["n_planned_within_horizon"].sum()),
            "fingerprint_sha256": cohort_fingerprint(feats),
            "train": {"n": int(len(tr)), "events": int(ytr.sum()), "discharges": f"<= {TRAIN_END.isoformat()}"},
            "test": {"n": int(len(te)), "events": int(yte.sum()), "discharges": f"> {TRAIN_END.isoformat()}"},
        },
        "primary_model": primary,
        "train_oof": oof_metrics,
        "test": test_metrics,
        "calibration_slope_acceptance_0.8_1.2": {
            k: bool(0.8 <= test_metrics[k]["calibration_slope"] <= 1.2) for k in test_metrics
        },
        "decision_curve": {"at_thresholds": dca_summary, "primary_vs_default_strategies": dca_coverage},
        "survival": {"km_by_risk_quartile": km_summary, **timing},
        "oracle": oracle,
        "hyperparameters": {"logistic_l2_C": float(coefs["C"].iloc[0])},
        "runtime_seconds": {k: round(v, 1) for k, v in b["timings"].items()},
    }
    metrics["runtime_seconds"]["pipeline_total"] = round(time.perf_counter() - t_start, 1)
    write_report(metrics, sub, results / "report.md")
    return metrics


def _f(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_report(m: dict, sub: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Day 03 results (SYNTHETIC data)",
        "",
        "## Cohort attrition",
        "",
        "| Step | n | Unplanned 30-day readmits |",
    ]
    lines.append("|---|---|---|")
    for r in m["cohort"]["attrition"]:
        lines.append(f"| {r['step']} | {r['n']:,} | {int(r['n_unplanned_readmit_30d'] or 0):,} |")
    lines += ["", "## Test-period performance", ""]
    lines.append("| Model | AUROC (95% CI) | Brier | Scaled Brier | Calibration slope (95% CI) | CITL | O/E |")
    lines.append("|---|---|---|---|---|---|---|")
    for k, v in m["test"].items():
        ci = v["ci95"]
        lines.append(
            f"| {k} | {_f(v['auroc'])} ({_f(ci['auroc'][0])}-{_f(ci['auroc'][1])}) | {_f(v['brier'], 4)} | "
            f"{_f(v['scaled_brier'])} | {_f(v['calibration_slope'])} ({_f(ci['calibration_slope'][0])}-"
            f"{_f(ci['calibration_slope'][1])}) | {_f(v['calibration_intercept'])} | {_f(v['o_e_ratio'])} |"
        )
    lines += ["", f"Primary model (lowest training out-of-fold log loss): **{m['primary_model']}**", ""]
    lines += [
        "## Subgroups (primary model, test)",
        "",
        "| Dimension | Level | n | Events | Obs | Pred | O/E | AUROC | Brier | Slope |",
    ]
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in sub.itertuples():
        lines.append(
            f"| {r.dimension} | {r.level} | {r.n:,} | {r.events:,} | {_f(r.observed_rate)} | {_f(r.mean_predicted)} | "
            f"{_f(r.o_e_ratio, 2)} | {_f(r.auroc)} | {_f(r.brier, 4)} | {_f(r.calibration_slope, 2)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def save_metrics(metrics: dict, results: Path) -> None:
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
