"""End-to-end pipeline: generate -> detect -> group -> root cause -> evaluate -> results/."""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import detect, grouping, rca
from .evaluate import detection_metrics, match_groups
from .generate import SimConfig, Telemetry, generate, to_wide_frame
from .topology import CALLS, call_graph

DEPLOY_LOOKBACK, DEPLOY_LOOKAHEAD = 30, 5
PERSIST = 3  # minutes above threshold before a service alert fires
QUANTILE = 0.999  # per-service threshold = this quantile of training anomaly scores
RCA_METHODS = ["max_anomaly", "topology_frontier", "earliest_onset_pelt", "earliest_onset_binseg", "causal_pagerank", "ensemble"]


def run_detectors(tel: Telemetry, z: np.ndarray, seed: int) -> dict[str, detect.DetectorOutput]:
    tr = tel.config.train_days * 1440
    feats = detect.features(z)
    return {
        "univariate_threshold": detect.univariate(z, k=PERSIST),
        "iforest": detect.multivariate(feats, tr, "iforest", seed, quantile=QUANTILE, k=PERSIST),
        "ecod": detect.multivariate(feats, tr, "ecod", seed, quantile=QUANTILE, k=PERSIST),
    }


def deployed_near(deploys: pd.DataFrame, t: int) -> set[str]:
    m = (deploys["minute"] >= t - DEPLOY_LOOKBACK) & (deploys["minute"] <= t + DEPLOY_LOOKAHEAD)
    return set(deploys.loc[m, "service"])


def loo_likelihood_ratio(rows: list[dict], exclude: int) -> float:
    """P(recent deploy | root) / P(recent deploy | non-root candidate), Laplace-smoothed, leave-one-out."""
    a = n_r = b = n_o = 0
    for r in rows:
        if r["incident_id"] == exclude:
            continue
        for c in r["candidates"]:
            if c == r["root"]:
                n_r += 1
                a += c in r["deployed"]
            else:
                n_o += 1
                b += c in r["deployed"]
    return ((a + 1) / (n_r + 2)) / ((b + 1) / (n_o + 2))


def run(
    cfg: SimConfig | None = None, results_dir: Path | None = None, data_dir: Path | None = None, public: bool = True, figures: bool = True
) -> dict:
    t0 = time.time()
    cfg = cfg or SimConfig()
    g = call_graph()
    tel = generate(cfg)
    tr = cfg.train_days * 1440
    timings = {"generate_s": round(time.time() - t0, 2)}

    z = detect.seasonal_zscores(tel.values, tr)
    sig = rca.anomaly_signal(z)
    t1 = time.time()
    dets = run_detectors(tel, z, cfg.seed)
    timings["detect_s"] = round(time.time() - t1, 2)

    # Raw alert volume (status quo: one alert rule per service x metric), test period only
    raw = detect.raw_metric_alerts(z, tel.services, tel.metrics)
    raw = raw[raw["start"] >= tr]

    inc = tel.incidents
    det_metrics, grouped, ep_all = {}, {}, {}
    for name, d in dets.items():
        ep = detect.episodes(d.flags, d.scores, tel.services)
        ep = ep[ep["start"] >= tr].reset_index(drop=True)
        ep_all[name] = ep
        for gname, hops in (("topological", 2), ("temporal_only", math.inf)):
            gp = grouping.group_alerts(ep, g, window=10, max_hops=hops)
            summ = match_groups(grouping.summarise_groups(gp), inc)
            m = detection_metrics(summ, inc)
            m["n_alert_episodes"] = len(ep)
            m["incidents_merged_into_one_group"] = int(_merged_groups(summ, gp, inc))
            det_metrics[f"{name}/{gname}"] = m
            grouped[(name, gname)] = (gp, summ)
        # no grouping: every per-service episode is its own incident
        flat = pd.DataFrame(
            {"group": range(len(ep)), "start": ep["start"], "end": ep["end"], "n_alerts": 1, "services": ep["service"], "peak": ep["peak"]}
        )
        summ = match_groups(flat, inc)
        det_metrics[f"{name}/none"] = detection_metrics(summ, inc) | {"n_alert_episodes": len(ep)}

    # sensitivity of the ECOD alert rule to persistence k and threshold quantile (topological grouping)
    sens = []
    feats = detect.features(z)
    for q in (0.999, 0.9995):
        base = detect.multivariate(feats, tr, "ecod", cfg.seed, quantile=q, k=1)
        for k in (2, 3):
            fl = detect.persistence(base.scores > 1, k)
            ep = detect.episodes(fl, base.scores, tel.services)
            ep = ep[ep["start"] >= tr].reset_index(drop=True)
            sm = match_groups(grouping.summarise_groups(grouping.group_alerts(ep, g, window=10, max_hops=2)), inc)
            dm = detection_metrics(sm, inc)
            sens.append(
                dict(
                    quantile=q,
                    persistence=k,
                    precision=dm["precision"],
                    recall=dm["recall"],
                    delay_median_min=dm["delay_median_min"],
                    n_groups=dm["n_groups"],
                )
            )

    # choose the primary detector by F1 of the topologically grouped incidents
    primary = max(dets, key=lambda k: det_metrics[f"{k}/topological"]["f1"])
    gp, summ = grouped[(primary, "topological")]

    # ---- root cause on each detected injected incident (its largest matched group) ----
    t2 = time.time()
    rows = []
    for i in inc.itertuples(index=False):
        mg = summ[summ["incident_ids"].apply(lambda ids, k=i.incident_id: k in ids)]
        if mg.empty:
            continue
        best = mg.sort_values(["n_alerts", "start"], ascending=[False, True]).iloc[0]
        gsvcs = best["services"].split(";")
        res = rca.rank_incident(
            sig, dets[primary].scores, tel.services, g, gsvcs, int(best["start"]), int(best["end"]), seed=cfg.seed + int(i.incident_id)
        )
        true_on = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in i.true_onsets.split(";")}
        rows.append(
            dict(
                incident_id=int(i.incident_id),
                root=i.root,
                fault=i.fault,
                start=int(i.start),
                first_alert=int(best["start"]),
                group_services=gsvcs,
                candidates=res.candidates,
                root_in_candidates=i.root in res.candidates,
                deployed=deployed_near(tel.deploys, int(best["start"])),
                deploy_on_root=bool(i.deploy_on_root),
                decoy_deploy=bool(i.decoy_deploy),
                scores=res.scores,
                onsets=res.onsets,
                true_onsets=true_on,
                causal_edges=res.causal_edges,
            )
        )
    timings["rca_s"] = round(time.time() - t2, 2)

    n_inj = len(inc)
    n_det = len(rows)
    rc = {}
    for m in RCA_METHODS + ["deploy_only"]:
        for prior in (False, True):
            if m == "deploy_only" and not prior:
                continue
            h1 = h3 = 0
            for r in rows:
                base = r["scores"]["max_anomaly"] if m == "deploy_only" else r["scores"][m]
                if prior:
                    lr = loo_likelihood_ratio(rows, r["incident_id"]) if m != "deploy_only" else 1e6
                    base = rca.apply_deploy_prior(base, r["deployed"], lr)
                h1 += rca.topk_hit(base, r["root"], 1)
                h3 += rca.topk_hit(base, r["root"], 3)
            key = f"{m}+deploy_prior" if prior and m != "deploy_only" else m
            rc[key] = dict(
                top1_detected=round(h1 / n_det, 4),
                top3_detected=round(h3 / n_det, 4),
                top1_all_injected=round(h1 / n_inj, 4),
                top3_all_injected=round(h3 / n_inj, 4),
                hits_top1=h1,
                hits_top3=h3,
            )
    lrs = [loo_likelihood_ratio(rows, r["incident_id"]) for r in rows]

    # deploy-prior value split by whether the incident was deploy-caused
    split = {}
    for m in ("ensemble", "causal_pagerank"):
        for flag in (True, False):
            sub = [r for r in rows if r["deploy_on_root"] == flag]
            if not sub:
                continue
            a = sum(rca.topk_hit(r["scores"][m], r["root"], 1) for r in sub)
            b = sum(
                rca.topk_hit(rca.apply_deploy_prior(r["scores"][m], r["deployed"], loo_likelihood_ratio(rows, r["incident_id"])), r["root"], 1)
                for r in sub
            )
            split[f"{m}/{'deploy_caused' if flag else 'not_deploy_caused'}"] = dict(
                n=len(sub), top1_without_prior=round(a / len(sub), 4), top1_with_prior=round(b / len(sub), 4)
            )

    # change-point onset accuracy on the root and every affected service that had a detectable onset
    cp = {}
    for m in ("pelt", "binseg"):
        err_all, err_root, found, total = [], [], 0, 0
        for r in rows:
            for s, o in r["onsets"][m].items():
                if s not in r["true_onsets"]:
                    continue
                total += 1
                if o is None:
                    continue
                found += 1
                e = abs(o - r["true_onsets"][s])
                err_all.append(e)
                if s == r["root"]:
                    err_root.append(e)
        cp[m] = dict(
            affected_services_evaluated=total,
            onset_found_share=round(found / total, 4) if total else None,
            onset_mae_min=round(float(np.mean(err_all)), 2) if err_all else None,
            onset_median_abs_err_min=float(np.median(err_all)) if err_all else None,
            root_onset_median_abs_err_min=float(np.median(err_root)) if err_root else None,
        )

    # alert volume funnel (test period)
    gm = det_metrics[f"{primary}/topological"]
    funnel = dict(
        raw_metric_alerts=int(len(raw)),
        per_service_episodes=int(gm["n_alert_episodes"]),
        grouped_incidents_topological=int(gm["n_groups"]),
        grouped_incidents_temporal_only=int(det_metrics[f"{primary}/temporal_only"]["n_groups"]),
        injected_incidents=n_inj,
    )
    funnel["reduction_raw_to_incidents_pct"] = round(100 * (1 - funnel["grouped_incidents_topological"] / funnel["raw_metric_alerts"]), 2)
    funnel["reduction_episodes_to_incidents_pct"] = round(100 * (1 - funnel["grouped_incidents_topological"] / funnel["per_service_episodes"]), 2)

    public_res = None
    if public:
        from .public import run_public_check

        t3 = time.time()
        public_res = run_public_check((data_dir or Path("data")) / "nab", cfg.seed)
        timings["public_check_s"] = round(time.time() - t3, 2)

    root_ok = sum(r["root_in_candidates"] for r in rows)
    metrics = dict(
        data=dict(
            path_used="synthetic_generator",
            note="All platform telemetry is SYNTHETIC (seeded). The prompt specifies a simulated 20-service platform; public NAB data is used only as a detector sanity check.",
            public_sanity_check=(public_res or {}).get("path_used", "not_run"),
            config=asdict(cfg),
            n_services=len(tel.services),
            n_call_edges=len(CALLS),
            n_minutes=tel.n_minutes,
            metrics=tel.metrics,
            n_values=int(tel.values.size),
            injected_incidents=n_inj,
            faults=inc["fault"].value_counts().to_dict(),
            deploys=tel.deploys["kind"].value_counts().to_dict(),
            incidents_with_bad_deploy=int(inc["deploy_on_root"].sum()),
            incidents_with_decoy_deploy=int(inc["decoy_deploy"].sum()),
            traffic_surges=len(tel.surges),
            mean_affected_services=round(float(inc["n_affected"].mean()), 2),
        ),
        detection=dict(
            primary_detector=primary,
            selection="highest F1 of topologically grouped incidents (selected on the same test period)",
            persistence_min=PERSIST,
            threshold_quantile=QUANTILE,
            by_detector=det_metrics,
            ecod_sensitivity=sens,
        ),
        alert_volume=funnel,
        root_cause=dict(
            evaluated_incidents=n_det,
            root_in_candidate_set=root_ok,
            mean_candidates=round(float(np.mean([len(r["candidates"]) for r in rows])), 2),
            methods=rc,
            deploy_prior=dict(
                window=f"deploy in [first_alert-{DEPLOY_LOOKBACK}, first_alert+{DEPLOY_LOOKAHEAD}] min",
                likelihood_ratio_loo_median=round(float(np.median(lrs)), 3),
                likelihood_ratio_loo_range=[round(min(lrs), 3), round(max(lrs), 3)],
                by_incident_type=split,
            ),
        ),
        change_points=cp,
        public_nab=public_res,
        runtime=dict(timings | {"total_s": round(time.time() - t0, 2)}, python=platform.python_version()),
    )

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json))
        _write_tables(results_dir, rows, tel, primary)
        from .report import write_report, write_timelines

        write_report(results_dir, metrics)
        write_timelines(results_dir, rows, tel, n=12)
        if figures:
            from . import figures as fig

            fig.make_all(results_dir / "figures", tel, z, dets, primary, raw, funnel, rc, rows, g)
    if data_dir is not None:
        _write_sample(data_dir / "sample", tel)
    metrics["_rows"] = rows
    return metrics


def _merged_groups(summ: pd.DataFrame, gp: pd.DataFrame, inc: pd.DataFrame) -> int:
    """Number of groups whose alerts overlap two or more distinct injected incidents (over-grouping)."""
    n = 0
    for _gid, sub in gp.groupby("group"):
        hits = set()
        for e in sub.itertuples(index=False):
            for i in inc.itertuples(index=False):
                if i.start - 15 <= e.start <= i.end + 15 and e.service in i.affected.split(";"):
                    hits.add(i.incident_id)
        n += len(hits) > 1
    return n


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def _write_tables(results_dir: Path, rows: list[dict], tel: Telemetry, primary: str) -> None:
    out = []
    for r in rows:
        lr = loo_likelihood_ratio(rows, r["incident_id"])
        ens = r["scores"]["ensemble"]
        post = rca.apply_deploy_prior(ens, r["deployed"], lr)
        top = sorted(post, key=lambda c: (-post[c], c))[:3]
        out.append(
            dict(
                incident_id=r["incident_id"],
                true_root=r["root"],
                fault=r["fault"],
                delay_min=r["first_alert"] - r["start"],
                n_group_services=len(r["group_services"]),
                n_candidates=len(r["candidates"]),
                deploy_on_root=r["deploy_on_root"],
                decoy_deploy=r["decoy_deploy"],
                top1=top[0],
                top2=top[1] if len(top) > 1 else "",
                top3=top[2] if len(top) > 2 else "",
                hit_top1=top[0] == r["root"],
                hit_top3=r["root"] in top,
            )
        )
    pd.DataFrame(out).to_csv(results_dir / "incident_rca.csv", index=False)


def _write_sample(sample_dir: Path, tel: Telemetry) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    first = tel.incidents.iloc[0]
    a = max(0, int(first["start"]) - 180)
    to_wide_frame(tel, a, a + 360).round(4).to_csv(sample_dir / "telemetry_6h_SYNTHETIC.csv", index=False)
    tel.incidents.to_csv(sample_dir / "incidents_SYNTHETIC.csv", index=False)
    tel.deploys.to_csv(sample_dir / "deploys_SYNTHETIC.csv", index=False)
    pd.DataFrame(CALLS, columns=["caller", "callee"]).to_csv(sample_dir / "call_graph.csv", index=False)
