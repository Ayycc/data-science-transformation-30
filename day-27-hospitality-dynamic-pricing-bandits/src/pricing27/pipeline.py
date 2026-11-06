"""End-to-end run: logs -> elasticity estimation -> multi-seed policy study -> stretch goals -> results/."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures, report
from .estimation import TEST_MULTS, demand_models, estimate_elasticities, generate_logs
from .ope import evaluate
from .policies import Guardrails
from .simulator import BUCKETS, MarketConfig, make_world
from .study import BANDITS, POLICY_ORDER, StudyConfig, mean_ci, regret_exponent, run_seed

LOG_SEED = 20261106


@dataclass
class RunConfig:
    study: StudyConfig = field(default_factory=StudyConfig)
    reaction_seeds: int = 10
    reaction_strength: float = 0.5
    reaction_policies: tuple = ("static", "greedy_iv", "linucb", "thompson")
    workers: int = 4
    figures: bool = True


def small_config() -> RunConfig:
    return RunConfig(study=StudyConfig(n_seeds=2, n_nights=120), reaction_seeds=2, workers=1)


def _job(args):
    seed, models, cfg, scfg = args
    return seed, run_seed(seed, models, cfg, scfg)


def _run_many(seeds, models, cfg, scfg, workers) -> dict:
    jobs = [(s, models, cfg, scfg) for s in seeds]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return dict(ex.map(_job, jobs))
    return dict(map(_job, jobs))


def _summarise(runs: dict, names, scfg: StudyConfig, rng) -> dict:
    seeds = sorted(runs)
    out = {}
    base = np.array([runs[s]["static"]["revenue"] for s in seeds])
    for name in names:
        rs = [runs[s][name] for s in seeds]
        rev = np.array([r["revenue"] for r in rs])
        cum = np.mean([r["cum_regret"] for r in rs], axis=0)
        per_night = np.diff(np.concatenate([[0.0], cum]))
        years = [per_night[i : i + 365] for i in range(0, len(per_night), 365) if len(per_night[i : i + 365]) >= 60]
        out[name] = dict(
            revenue_mean=float(rev.mean()),
            revpar=mean_ci(np.array([r["revpar"] for r in rs]), rng),
            occupancy=float(np.mean([r["occupancy"] for r in rs])),
            adr=float(np.mean([r["adr"] for r in rs])),
            sellout_share=float(np.mean([r["sellout_share"] for r in rs])),
            uplift_vs_static=mean_ci(rev / base - 1.0, rng),
            seeds_beating_static=int(np.sum(rev > base)),
            cum_regret_final=float(cum[-1]),
            regret_per_night_by_year=[float(y.mean()) for y in years],
            regret_exponent_mean_curve=regret_exponent(cum),
            regret_exponent_per_seed=[regret_exponent(r["cum_regret"]) for r in rs],
            volatility_mean_abs_dlog=float(np.mean([r["volatility"] for r in rs])),
            max_rel_change=float(np.max([r["max_rel_change"] for r in rs])),
            min_price=float(np.min([r["min_price"] for r in rs])),
            max_price=float(np.max([r["max_price"] for r in rs])),
            guardrail_violations_posted=int(np.sum([r["guardrail_violations_posted"] for r in rs])),
            guardrail_interventions=int(np.sum([r["guardrail_interventions"] for r in rs])),
            decisions=int(sum(r["prices"].size for r in rs)),
        )
        out[name]["regret_sublinear"] = bool(
            out[name]["regret_exponent_mean_curve"] < 0.95 and out[name]["regret_per_night_by_year"][-1] < out[name]["regret_per_night_by_year"][0]
        )
    return out


def _ope(logs: pd.DataFrame, models, cfg: MarketConfig) -> dict:
    iv = models["iv_2sls"]
    mults = TEST_MULTS

    def greedy_myopic(ctx):
        p = ctx.static * mults
        return float(mults[np.argmax(p * iv.predict(ctx, p))])

    targets: dict[str, Callable] = {
        "static_rate (m=1.0)": lambda c: 1.0,
        "premium (m=1.1)": lambda c: 1.1,
        "discount (m=0.9)": lambda c: 0.9,
        "late-window premium (m=1.2 in 0-3d, else 1.0)": lambda c: 1.2 if c.bucket == 2 else 1.0,
        "greedy_iv myopic": greedy_myopic,
    }
    hidden_u = make_world(0, int(logs["night"].max()) + 1, LOG_SEED, cfg).u
    return evaluate(logs, targets, cfg, hidden_u)


def run(rc: RunConfig, out: Path, data_dir: Path, sample_dir: Path | None, log: Callable[[str], None] = print) -> dict:
    t0 = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    cfg = MarketConfig()
    scfg = rc.study
    metrics: dict = dict(
        data=dict(
            path_used="synthetic_simulator",
            note="SYNTHETIC. The prompt specifies a market simulator; no public dataset is used.",
            log_seed=LOG_SEED,
            capacity=cfg.capacity,
            booking_windows=list(BUCKETS),
        )
    )

    logs = generate_logs(cfg, seed=LOG_SEED)
    data_dir.mkdir(parents=True, exist_ok=True)
    logs.to_csv(data_dir / "booking_logs_SYNTHETIC.csv", index=False)
    if sample_dir is not None:
        sample_dir.mkdir(parents=True, exist_ok=True)
        logs.to_csv(sample_dir / "booking_logs_SYNTHETIC_sample.csv", index=False)
    nights = logs.groupby("night")["sold"].sum()
    metrics["logs"] = dict(
        rows=len(logs),
        nights=int(logs["night"].nunique()),
        randomised_test_cells=int(logs["price_test"].sum()),
        occupancy=float(nights.sum() / (cfg.capacity * len(nights))),
        sellout_share=float((nights >= cfg.capacity).mean()),
        corr_logprice_vs_logdemand_rm_cells=float(
            np.corrcoef(
                np.log(logs.loc[logs.price_test == 0, "price"] / logs.loc[logs.price_test == 0, "static_price"]),
                np.log(logs.loc[logs.price_test == 0, "demand_unconstrained"] + 0.5),
            )[0, 1]
        ),
    )
    log(f"logs: {len(logs)} cells, {metrics['logs']['randomised_test_cells']} randomised")

    est = estimate_elasticities(logs)
    models = demand_models(est)
    models["logs"] = logs
    metrics["elasticity"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("beta",)} for k, v in est.items()}
    for k in ("naive_ols", "randomised_test_ols", "iv_2sls"):
        th, se = est[k]["theta"], est[k]["se"]
        metrics["elasticity"][k]["ci95"] = [th - 1.96 * se, th + 1.96 * se]
        metrics["elasticity"][k]["covers_truth_mean"] = bool(th - 1.96 * se <= est["truth"]["theta_mean"] <= th + 1.96 * se)
    log(
        f"semi-elasticity naive {est['naive_ols']['theta']:.3f} | test {est['randomised_test_ols']['theta']:.3f} | IV {est['iv_2sls']['theta']:.3f} | truth {est['truth']['theta_mean']:.3f}"
    )

    seeds = list(range(scfg.seed0, scfg.seed0 + scfg.n_seeds))
    runs = _run_many(seeds, models, cfg, scfg, rc.workers)
    rng = np.random.default_rng(7)
    metrics["study"] = dict(config=dict(asdict(scfg), guard=asdict(scfg.guard)), policies=_summarise(runs, scfg.policies, scfg, rng))
    for name in scfg.policies:
        s = metrics["study"]["policies"][name]
        log(
            f"{name:13s} uplift {s['uplift_vs_static']['mean']:+.2%} regret exp {s['regret_exponent_mean_curve']:.2f} viol {s['guardrail_violations_posted']}"
        )

    # stretch: competitor reaction
    rcfg = replace(cfg, comp_reaction=rc.reaction_strength)
    r_scfg = replace(scfg, n_seeds=rc.reaction_seeds, policies=rc.reaction_policies)
    r_runs = _run_many(list(range(scfg.seed0, scfg.seed0 + rc.reaction_seeds)), models, rcfg, r_scfg, rc.workers)
    metrics["competitor_reaction"] = dict(
        strength=rc.reaction_strength, seeds=rc.reaction_seeds, policies=_summarise(r_runs, rc.reaction_policies, r_scfg, rng)
    )
    log(
        "competitor reaction: "
        + ", ".join(f"{n} {v['uplift_vs_static']['mean']:+.2%}" for n, v in metrics["competitor_reaction"]["policies"].items())
    )

    metrics["ope"] = _ope(logs, models, cfg)
    log("OPE done")

    P = metrics["study"]["policies"]
    metrics["acceptance"] = dict(
        bandits_sublinear_regret={b: P[b]["regret_sublinear"] for b in BANDITS if b in P},
        guardrail_violations_total=int(sum(v["guardrail_violations_posted"] for v in P.values())),
        uplift_ci_reported=True,
    )
    metrics["timings"] = dict(total_seconds=time.perf_counter() - t0)
    if rc.figures:
        figures.make_all(runs, metrics, logs, scfg, out / "figures")
    report.write(metrics, out / "report.md")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))
    pd.DataFrame(
        [
            dict(
                policy=n,
                uplift_mean=v["uplift_vs_static"]["mean"],
                uplift_lo=v["uplift_vs_static"]["t_ci"][0],
                uplift_hi=v["uplift_vs_static"]["t_ci"][1],
                revpar=v["revpar"]["mean"],
                occupancy=v["occupancy"],
                adr=v["adr"],
                regret=v["cum_regret_final"],
                regret_exponent=v["regret_exponent_mean_curve"],
                volatility=v["volatility_mean_abs_dlog"],
                violations=v["guardrail_violations_posted"],
            )
            for n, v in P.items()
        ]
    ).to_csv(out / "policy_summary.csv", index=False)
    return metrics


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Guardrails):
        return asdict(o)
    raise TypeError(type(o))


__all__ = ["RunConfig", "run", "small_config", "POLICY_ORDER"]
