"""End-to-end MLOps lifecycle: data -> tracked training -> gated registry promotion -> API serving ->
monthly Evidently monitoring -> retraining trigger -> gated re-promotion -> shadow/canary comparison."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from claims_mlops import config as C
from claims_mlops import data as D
from claims_mlops import models as M
from claims_mlops import monitoring as Mon
from claims_mlops import plots
from claims_mlops import registry as R
from claims_mlops.api import create_app
from claims_mlops.modelcard import write_model_card

API_CHUNK = 5_000


@dataclass
class Paths:
    results: Path = C.RESULTS
    mlruns: Path = C.MLRUNS
    reports_all: Path = C.ROOT / "data" / "reports"
    export: Path = C.MODEL_EXPORT

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def reports(self) -> Path:
        return self.results / "reports"


@dataclass
class Serving:
    """The live deployment: champion behind the FastAPI app (in-process TestClient), optional shadow."""

    version: str
    model: object
    shadow_version: str | None = None
    shadow: object = None
    client: TestClient = field(init=False)

    def __post_init__(self) -> None:
        self.client = TestClient(
            create_app(self.model, f"v{self.version}", self.shadow, f"v{self.shadow_version}" if self.shadow_version else None)
        )
        self.client.__enter__()
        assert self.client.get("/ready").status_code == 200

    def close(self) -> None:
        self.client.__exit__(None, None, None)

    def score(self, batch: pd.DataFrame) -> np.ndarray:
        """Production batch scoring through the HTTP contract (/score/batch)."""
        out = []
        cols = [*C.FEATURES, C.WEIGHT]
        for start in range(0, len(batch), API_CHUNK):
            part = batch.iloc[start : start + API_CHUNK]
            recs = part[cols].rename(columns={C.WEIGHT: "Exposure"}).to_dict(orient="records")
            for r, pid in zip(recs, part["PolicyID"], strict=True):
                r["policy_id"] = str(pid)
            resp = self.client.post("/score/batch", json={"policies": recs})
            resp.raise_for_status()
            out.extend(s["pure_premium"] for s in resp.json()["scores"])
        return np.asarray(out)


def _train_candidate(family: str, train: pd.DataFrame, cfg_seed: int = C.SEED, weak: bool = False):
    if family == "glm":
        feats = ["DrivAge", "Area"] if weak else None
        params = {"alpha": 1e-3, "n_bins": 8, "features": ",".join(feats or C.FEATURES)}
        return M.fit(M.build_glm(alpha=1e-3, features=feats), train), params
    pipe, cat_idx = M.build_gbm(seed=cfg_seed)
    gbm = pipe.named_steps["gbm"]
    params = {
        k: gbm.get_params()[k]
        for k in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree", "reg_lambda")
    }
    return M.fit(pipe, train, cat_idx), params


def _lifecycle_step(
    client, family: str, train: pd.DataFrame, valid: pd.DataFrame, gate_policy: C.GatePolicy, tags: dict, weak: bool = False
) -> dict:
    t0 = time.time()
    model, params = _train_candidate(family, train, weak=weak)
    cand = R.log_candidate(model, family, params, train, valid, tags=tags)
    champ_ver, champ_metrics = R.evaluate_champion(client, valid)
    decision = R.gate(cand, champ_metrics, champ_ver, gate_policy)
    action = R.apply_decision(client, decision)
    return {
        "family": family + ("-weak" if weak else ""),
        "action": action,
        "seconds": round(time.time() - t0, 2),
        **decision.to_dict(),
        "model": model,
    }


def run(force_synthetic: bool = False, cfg: C.RunConfig | None = None, paths: Paths | None = None, fresh_registry: bool = True) -> dict:
    cfg = cfg or C.RunConfig()
    paths = paths or Paths()
    t_start = time.time()
    for p in (paths.figures, paths.reports):
        p.mkdir(parents=True, exist_ok=True)
    if fresh_registry and paths.mlruns.exists():
        shutil.rmtree(paths.mlruns)
    if paths.reports_all.exists():
        shutil.rmtree(paths.reports_all)

    # 1. data -------------------------------------------------------------------------------
    acq = D.load_or_create(force_synthetic=force_synthetic, cfg=cfg.sim)
    df = acq.policies
    D.write_sample(df)
    n0 = cfg.initial_train_months
    train0 = D.split_months(df, range(1, n0 - 1))
    valid0 = D.split_months(df, range(n0 - 1, n0 + 1))

    # 2. initial tracked training + automatic gated promotion --------------------------------
    client = R.setup(paths.mlruns)
    lifecycle = []
    for family, weak, note in [
        ("glm", False, "bootstrap champion"),
        ("gbm", False, "challenger"),
        ("glm", True, "negative control: under-specified challenger"),
    ]:
        step = _lifecycle_step(client, family, train0, valid0, cfg.gate, {"stage": "initial", "note": note}, weak=weak)
        step["note"] = note
        lifecycle.append(step)
    initial_ver = R.champion_version(client)
    initial_model = R.load_version(initial_ver)
    glm_model = lifecycle[0]["model"]

    # 3. production months -------------------------------------------------------------------
    serving = Serving(initial_ver, initial_model)
    ref = valid0.copy()
    ref[Mon.PRED] = initial_model.predict(ref)
    ref_dev = M.evaluate(initial_model, valid0)["tweedie_deviance"]
    ref0 = ref
    months_since_promo = 0
    monthly, triggers, canary = [], [], []
    last_month = cfg.sim.n_months
    for m in range(n0 + 1, last_month + 1):
        batch = D.split_months(df, [m])
        t0 = time.time()
        pred = serving.score(batch)
        api_s = time.time() - t0
        scored = batch.copy()
        scored[Mon.PRED] = pred
        perf = M.evaluate(serving.model, batch)  # labels assumed available at month end (see Limitations)
        static = M.evaluate(initial_model, batch)
        glm_perf = M.evaluate(glm_model, batch)
        html = paths.reports_all / f"drift_month_{m:02d}.html" if m in cfg.html_reports else None
        drift = Mon.drift_report(ref, scored, m, html_path=html)
        drift_fixed = Mon.drift_report(ref0, scored, m)  # detector evaluation vs the fixed initial reference
        if html is not None:
            shutil.copy(html, paths.reports / html.name)
        if serving.shadow is not None:
            summ = serving.client.get("/shadow/summary").json()
            canary.append(
                {
                    "month": m,
                    "champion_version": serving.version,
                    "shadow_version": serving.shadow_version,
                    "champion_deviance": perf["tweedie_deviance"],
                    "shadow_deviance": M.evaluate(serving.shadow, batch)["tweedie_deviance"],
                    "champion_ae": perf["ae_ratio"],
                    "shadow_ae": M.evaluate(serving.shadow, batch)["ae_ratio"],
                    "api_shadow_rows": summ["n"],
                    "api_shadow_mean_champion": summ["mean_champion"],
                    "api_shadow_mean_shadow": summ["mean_shadow"],
                    "api_mean_ratio_shadow_to_champion": summ["mean_ratio_shadow_to_champion"],
                    "api_rank_corr": summ["rank_corr"],
                }
            )
            serving.client.app.state.model_state.shadow_log.clear()
        decision = Mon.retrain_trigger(m, drift, perf, ref_dev, months_since_promo, cfg.trigger)
        row = {
            "month": m,
            "champion_version": serving.version,
            "n_policies": len(batch),
            "api_seconds": round(api_s, 3),
            "api_rows_per_s": round(len(batch) / api_s, 1),
            "api_max_abs_diff_vs_direct": float(np.max(np.abs(pred - serving.model.predict(batch)))),
            "deviance": perf["tweedie_deviance"],
            "deviance_ratio": perf["tweedie_deviance"] / ref_dev,
            "ae_ratio": perf["ae_ratio"],
            "gini": perf["gini"],
            "static_deviance": static["tweedie_deviance"],
            "static_ae_ratio": static["ae_ratio"],
            "glm_deviance": glm_perf["tweedie_deviance"],
            "drift_share": drift.drift_share,
            "drifted_columns": ";".join(drift.drifted_columns),
            **{f"drift_{k}": v for k, v in drift.column_scores.items()},
            "fixed_ref_drift_share": drift_fixed.drift_share,
            "fixed_ref_drifted_columns": ";".join(drift_fixed.drifted_columns),
            **{f"fixed_drift_{k}": v for k, v in drift_fixed.column_scores.items()},
            "trigger": decision.fire,
            "trigger_reasons": " | ".join(decision.reasons),
            "suppressed_by_cooldown": decision.suppressed_by_cooldown,
            "shift_injected": m >= cfg.sim.shift_start_month,
        }
        months_since_promo += 1
        if decision.fire and m < last_month:
            # retrain on the trailing window; validate on a held-out 30% of the newest month
            newest = batch.sample(frac=1.0, random_state=C.SEED + m)
            n_hold = int(0.3 * len(newest))
            hold, rest = newest.iloc[:n_hold], newest.iloc[n_hold:]
            window = D.split_months(df, range(m - cfg.trigger.train_window_months + 1, m))
            train = pd.concat([window, rest], ignore_index=True)
            step = _lifecycle_step(
                client, "gbm", train, hold, cfg.gate, {"stage": "retrain", "trigger": f"month {m}: " + " | ".join(decision.reasons)}
            )
            step.update(month=m, note=f"retrain triggered at month {m}")
            lifecycle.append(step)
            triggers.append(
                {"month": m, "reasons": decision.reasons, "candidate_version": step["candidate_version"], "action": step["action"]}
            )
            row["retrain_action"] = step["action"]
            if step["action"] == "promoted":
                old_ver, old_model = serving.version, serving.model
                serving.close()
                serving = Serving(step["candidate_version"], step["model"], shadow_version=old_ver, shadow=old_model)
                ref = batch.copy()
                ref[Mon.PRED] = serving.model.predict(ref)
                ref_dev = step["candidate_metrics"]["tweedie_deviance"]
                months_since_promo = 0
        monthly.append(row)
    serving.close()

    # 4. export champion, API contract smoke on the exported artefact ------------------------
    exported = R.export_champion(paths.export)
    champ_ver = R.champion_version(client)
    (exported / "VERSION").write_text(f"v{champ_ver}\n")
    api_smoke = _api_smoke(exported)

    # 5. write results -----------------------------------------------------------------------
    mon = pd.DataFrame(monthly)
    mon.to_csv(paths.results / "monthly_monitoring.csv", index=False, float_format="%.6g")
    reg = pd.DataFrame(R.registry_table(client))
    reg.drop(columns=["run_id"]).to_csv(paths.results / "registry.csv", index=False, float_format="%.6g")
    decisions = [{k: v for k, v in s.items() if k != "model"} for s in lifecycle]
    (paths.results / "gate_decisions.json").write_text(json.dumps(decisions, indent=2, default=float))
    plots.monitoring_timeline(mon, cfg, paths.figures / "monitoring_timeline.png")
    plots.drift_heatmap(mon, paths.figures / "drift_heatmap.png")
    plots.lift_chart(
        D.split_months(df, range(n0 - 1, n0 + 1)),
        {"GLM v1": glm_model, f"GBM v{initial_ver}": initial_model},
        paths.figures / "lift_chart_validation.png",
    )
    card = write_model_card(client, champ_ver, paths.results / "model_card.md", gate_policy=cfg.gate)
    metrics = _summarise(acq, cfg, lifecycle, mon, triggers, canary, reg, api_smoke, champ_ver, initial_ver, time.time() - t_start, card)
    (paths.results / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    from claims_mlops.report import write_report

    write_report(metrics, mon, reg, paths.results / "report.md")
    return metrics


def _api_smoke(exported: Path) -> dict:
    """Load the exported champion exactly as the container would (CLAIMS_MODEL_URI) and call each endpoint."""
    import os

    os.environ["CLAIMS_MODEL_URI"] = str(exported)
    try:
        app = create_app()
        with TestClient(app) as c:
            ready = c.get("/ready")
            one = c.post(
                "/score",
                json={
                    "VehPower": 7,
                    "VehAge": 3,
                    "DrivAge": 42,
                    "BonusMalus": 50,
                    "Density": 1200,
                    "Area": "D",
                    "VehBrand": "B12",
                    "VehGas": "Diesel",
                    "Region": "R11",
                    "Exposure": 0.5,
                },
            )
            bad = c.post(
                "/score",
                json={
                    "VehPower": 7,
                    "VehAge": 3,
                    "DrivAge": 12,
                    "BonusMalus": 50,
                    "Density": 1200,
                    "Area": "Z",
                    "VehBrand": "B12",
                    "VehGas": "Diesel",
                    "Region": "R11",
                },
            )
            return {
                "ready_status": ready.status_code,
                "ready_version": ready.json().get("model_version"),
                "score_status": one.status_code,
                "example_pure_premium": one.json().get("pure_premium"),
                "example_expected_cost": one.json().get("expected_cost"),
                "invalid_request_status": bad.status_code,
                "invalid_request_errors": len(bad.json().get("detail", [])),
            }
    finally:
        os.environ.pop("CLAIMS_MODEL_URI", None)


def _summarise(acq, cfg, lifecycle, mon, triggers, canary, reg, api_smoke, champ_ver, initial_ver, seconds, card) -> dict:
    shift = cfg.sim.shift_start_month
    pre, post = mon[~mon.shift_injected], mon[mon.shift_injected]
    injected = ["DrivAge", "Density"]

    def _flags(frame: pd.DataFrame, col: str) -> dict:
        return {c: int(frame[col].fillna("").str.split(";").map(lambda xs, c=c: c in xs).sum()) for c in C.FEATURES}

    flagged_post, flagged_pre = _flags(post, "fixed_ref_drifted_columns"), _flags(pre, "fixed_ref_drifted_columns")
    first_trigger = triggers[0]["month"] if triggers else None
    after = mon[mon.month > first_trigger] if first_trigger else mon.iloc[0:0]
    return {
        "data": {k: v for k, v in acq.provenance.items() if k != "sim"},
        "scale": {
            "months": cfg.sim.n_months,
            "policies_per_month": cfg.sim.policies_per_month,
            "rows": int(len(acq.policies)),
            "claim_rate": float((acq.policies.ClaimNb > 0).mean()),
            "shift_start_month": shift,
            "inflation_per_month": cfg.sim.inflation_per_month,
        },
        "initial_training": {
            "train_months": f"1-{cfg.initial_train_months - 2}",
            "valid_months": f"{cfg.initial_train_months - 1}-{cfg.initial_train_months}",
            "steps": [
                {
                    "family": s["family"],
                    "note": s["note"],
                    "version": s["candidate_version"],
                    "action": s["action"],
                    "val_tweedie_deviance": s["candidate_metrics"]["tweedie_deviance"],
                    "val_gini": s["candidate_metrics"]["gini"],
                    "val_ae_ratio": s["candidate_metrics"]["ae_ratio"],
                    "null_deviance": s["candidate_metrics"]["null_deviance"],
                    "deviance_gain_vs_reference": s["checks"]["deviance_gain"]["value"],
                    "gate_reference": s["checks"]["deviance_gain"]["vs"],
                    "checks_passed": {k: c["pass"] for k, c in s["checks"].items()},
                }
                for s in lifecycle
                if s.get("month") is None
            ],
            "initial_champion_version": initial_ver,
        },
        "monitoring": {
            "months": f"{int(mon.month.min())}-{int(mon.month.max())}",
            "pre_shift_months": int(len(pre)),
            "post_shift_months": int(len(post)),
            "injected_columns": injected,
            "fixed_reference": {
                "note": "every month vs the initial champion's validation window (detector evaluation)",
                "pre_shift_months_with_any_flag": int((pre.fixed_ref_drift_share > 0).sum()),
                "post_shift_months_flagging_all_injected": int(
                    post.fixed_ref_drifted_columns.fillna("").str.split(";").map(lambda xs: all(c in xs for c in injected)).sum()
                ),
                "flag_counts_post_shift": flagged_post,
                "flag_counts_pre_shift": flagged_pre,
                "max_pre_shift_score": float(pre[[f"fixed_drift_{c}" for c in C.FEATURES]].max().max()),
                "min_post_shift_score_injected": float(post[[f"fixed_drift_{c}" for c in injected]].min().min()) if len(post) else None,
            },
            "live_reference": {
                "note": "each month vs the deployed champion's own reference (reset on promotion); drives the trigger",
                "months_with_any_flag": [int(x) for x in mon.loc[mon.drift_share > 0, "month"]],
                "first_month_flagging_injected": int(
                    post[post.drifted_columns.fillna("").str.split(";").map(lambda xs: all(c in xs for c in injected))].month.min()
                )
                if len(post) and post.drifted_columns.fillna("").str.split(";").map(lambda xs: all(c in xs for c in injected)).any()
                else None,
                "drift_score_month_shift": {c: float(mon.loc[mon.month == shift, f"drift_{c}"].iloc[0]) for c in C.FEATURES}
                if (mon.month == shift).any()
                else None,
            },
            "api_rows_per_s_median": float(mon.api_rows_per_s.median()),
            "api_max_abs_diff_vs_direct": float(mon.api_max_abs_diff_vs_direct.max()),
        },
        "retraining": {
            "triggers": triggers,
            "n_triggers": len(triggers),
            "n_promotions": int(sum(t["action"] == "promoted" for t in triggers)),
            "retrain_steps": [
                {
                    "month": s["month"],
                    "version": s["candidate_version"],
                    "action": s["action"],
                    "candidate_deviance": s["candidate_metrics"]["tweedie_deviance"],
                    "champion_deviance": (s["champion_metrics"] or {}).get("tweedie_deviance"),
                    "deviance_gain": s["checks"]["deviance_gain"]["value"],
                    "candidate_ae": s["candidate_metrics"]["ae_ratio"],
                    "champion_ae": (s["champion_metrics"] or {}).get("ae_ratio"),
                }
                for s in lifecycle
                if s.get("month") is not None
            ],
            "after_first_trigger": {
                "months": f"{int(after.month.min())}-{int(after.month.max())}" if len(after) else None,
                "deviance_retrained_mean": float(after.deviance.mean()) if len(after) else None,
                "deviance_static_mean": float(after.static_deviance.mean()) if len(after) else None,
                "deviance_reduction_vs_static": float(1 - after.deviance.mean() / after.static_deviance.mean()) if len(after) else None,
                "ae_retrained_mean": float(after.ae_ratio.mean()) if len(after) else None,
                "ae_static_mean": float(after.static_ae_ratio.mean()) if len(after) else None,
                "abs_ae_error_retrained": float((after.ae_ratio - 1).abs().mean()) if len(after) else None,
                "abs_ae_error_static": float((after.static_ae_ratio - 1).abs().mean()) if len(after) else None,
            },
        },
        "canary": canary,
        "final_champion_version": champ_ver,
        "registry": reg.drop(columns=["run_id"]).to_dict(orient="records"),
        "api_smoke_exported_model": api_smoke,
        "model_card": card,
        "runtime_seconds": round(seconds, 1),
    }
