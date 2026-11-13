"""End-to-end pipeline: data -> baselines -> two-tower + FAISS -> LightGBM re-ranker -> offline metrics -> OPE."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import kendalltau

from . import ann, metrics, ope
from .baselines import ALS, ItemKNN, popularity_scores
from .data import Dataset, WorldConfig, load_dataset, temporal_split
from .rerank import FEATURES, ReRanker, build_context, features, labels_for, mmr, rerank
from .two_tower import TwoTowerConfig, TwoTowerRecommender

ALS_GRID = [
    {"factors": f, "regularization": r, "alpha": a, "iterations": 15} for f in (16, 32, 64, 128) for r in (0.1, 1.0, 5.0) for a in (1.0, 2.0, 10.0)
]


@dataclass
class RunConfig:
    world: WorldConfig = field(default_factory=WorldConfig)
    tt: TwoTowerConfig = field(default_factory=TwoTowerConfig)
    ope: ope.OPEConfig = field(default_factory=ope.OPEConfig)
    n_candidates: int = 100
    als_grid: list = field(default_factory=lambda: list(ALS_GRID))
    mmr_lambdas: tuple = (1.0, 0.8, 0.6, 0.4)
    latency_requests: int = 2050
    bootstrap: int = 1000
    data_mode: str | None = None
    seed: int = 5


def small_config() -> RunConfig:
    """Tiny world for tests and smoke runs (seconds, not minutes)."""
    return RunConfig(
        world=WorldConfig(n_users=400, n_items=300, mean_events=30, max_events=80),
        tt=TwoTowerConfig(epochs=2, batch_size=256, hidden=64, dim=32),
        ope=ope.OPEConfig(slate_size=10, n_rounds=2000, n_replicates=4, sweep_rounds=(1000,), sweep_replicates=2),
        n_candidates=50,
        als_grid=[{"factors": 16, "regularization": 0.05, "alpha": 5.0, "iterations": 5}],
        mmr_lambdas=(1.0, 0.6),
        latency_requests=120,
        bootstrap=50,
        data_mode="synthetic",
    )


def _seen_lists(x) -> list[np.ndarray]:
    return [x.indices[x.indptr[u] : x.indptr[u + 1]] for u in range(x.shape[0])]


def _stage(ds: Dataset, fit_df: pd.DataFrame, cfg: RunConfig, als_params: dict | None, log, tt: TwoTowerRecommender | None = None) -> dict:
    """Fit every model on ``fit_df`` and return score matrices, candidates and features context."""
    n_u, n_i = ds.n_users, ds.n_items
    X = metrics.interaction_matrix(fit_df["user"].to_numpy(), fit_df["item"].to_numpy(), n_u, n_i)
    out: dict = {"X": X, "fit_df": fit_df}
    out["pop"] = np.broadcast_to(popularity_scores(X), (n_u, n_i))
    t0 = time.perf_counter()
    knn = ItemKNN(k=100).fit(X)
    out["knn"] = knn.score(X)
    log(f"  item-kNN fitted ({time.perf_counter() - t0:.1f}s)")
    out["als_models"] = {}
    grid = [als_params] if als_params else cfg.als_grid
    for p in grid:
        t0 = time.perf_counter()
        m = ALS(**p, seed=cfg.seed).fit(X)
        out["als_models"][json.dumps(p, sort_keys=True)] = m.score()
        log(f"  ALS {p} fitted ({time.perf_counter() - t0:.1f}s)")
    if tt is None:
        tt = TwoTowerRecommender(n_u, n_i, ds.item_genres, ds.item_bucket, cfg.tt).fit(fit_df)
        log(
            f"  two-tower trained: {tt.history['n_pairs']} pairs, {tt.history['train_seconds']:.1f}s, loss {tt.history['epoch_loss'][0]:.3f} -> {tt.history['epoch_loss'][-1]:.3f}"
        )
    else:
        log("  two-tower reused (no retraining): user tower re-encodes the longer histories")
    hist = tt.histories(fit_df)
    U, I = tt.user_embeddings(hist), tt.item_embeddings()
    out.update(tt=tt, hist=hist, U=U, I=I, tt_scores=U @ I.T)
    t0 = time.perf_counter()
    hnsw = ann.build_hnsw(I)
    out["hnsw"], out["flat"] = hnsw, ann.build_flat(I)
    out["hnsw_build_s"] = time.perf_counter() - t0
    seen = _seen_lists(X)
    overfetch = int(min(n_i - cfg.n_candidates, np.asarray(X.sum(1)).max()))
    out["cands"] = ann.search_filtered(hnsw, U, seen, cfg.n_candidates, overfetch)
    out["ctx"] = build_context(fit_df, n_u, n_i, ds.item_genres, U, I, out["knn"], knn.sim)
    return out


def offline_suite(ds: Dataset, cfg: RunConfig, log=print, refit_two_tower: bool = False) -> tuple[dict, dict]:
    """Stage A (fit on train, select on validation) then stage B (refit on train+val, score test).

    ``refit_two_tower=True`` also retrains the two-tower on train+val in stage B, as ALS is (see scripts/refit_check.py).
    """
    n_u, n_i = ds.n_users, ds.n_items
    split = temporal_split(ds.interactions)
    users = np.arange(n_u)
    val_truth = metrics.interaction_matrix(split.val["user"].to_numpy(), split.val["item"].to_numpy(), n_u, n_i)
    test_truth = metrics.interaction_matrix(split.test["user"].to_numpy(), split.test["item"].to_numpy(), n_u, n_i)
    res: dict = {
        "dataset": {
            "name": ds.name,
            "n_users": n_u,
            "n_items": n_i,
            "n_interactions": len(ds.interactions),
            "n_train": len(split.train),
            "n_val": len(split.val),
            "n_test": len(split.test),
            "density": len(ds.interactions) / (n_u * n_i),
            "split": "per-user temporal leave-last-out: last 5 events test, previous 5 validation",
        }
    }

    def ev(recs, truth, X):
        return metrics.evaluate_recs(recs, truth, np.asarray(X.sum(0)).ravel(), ds.item_genres, n_u)

    # ---------- stage A: fit on train, select ALS, train re-ranker on validation labels ----------
    log("stage A: fit on train, evaluate on validation")
    A = _stage(ds, split.train, cfg, None, log)
    als_val = {}
    for key, s in A["als_models"].items():
        als_val[key] = metrics.ndcg_at_k(metrics.top_k(s, A["X"], 10), val_truth, 10)
    best_key = max(als_val, key=als_val.get)
    best_als = json.loads(best_key)
    log(f"  best ALS on validation: {best_als} (NDCG@10 {als_val[best_key]:.4f})")
    XA = features(A["ctx"], users, A["cands"])
    XA_als = np.concatenate([XA, A["als_models"][best_key][users[:, None], A["cands"]][..., None]], axis=-1)
    yA = labels_for(A["cands"], val_truth, users)
    # choose the re-ranker objective on held-out validation users (80/20 user split), then refit on all users
    sel_rng = np.random.default_rng(cfg.seed + 2)
    inner = sel_rng.random(n_u) < 0.8
    held = ~inner
    obj_scores = {"two-tower order": metrics.ndcg_at_k(A["cands"][held], val_truth[held], 10)}
    for obj in ("binary", "lambdarank"):
        r = ReRanker(obj, seed=cfg.seed).fit(XA[inner], yA[inner])
        obj_scores[obj] = metrics.ndcg_at_k(rerank(A["cands"][held], r.score(XA[held])), val_truth[held], 10)
    best_obj = max(("binary", "lambdarank"), key=obj_scores.get)
    log(f"  re-ranker objective selection on held-out validation users: {obj_scores} -> {best_obj}")
    ranker = ReRanker(best_obj, seed=cfg.seed).fit(XA, yA)
    ranker_als = ReRanker(best_obj, FEATURES + ["als_score"], seed=cfg.seed).fit(XA_als, yA)
    val = {
        "Popularity": ev(metrics.top_k(A["pop"], A["X"], 50), val_truth, A["X"]),
        "Item-kNN": ev(metrics.top_k(A["knn"], A["X"], 50), val_truth, A["X"]),
        "ALS": ev(metrics.top_k(A["als_models"][best_key], A["X"], 50), val_truth, A["X"]),
        "Two-tower": ev(metrics.top_k(A["tt_scores"], A["X"], 50), val_truth, A["X"]),
    }
    res["validation"] = {
        "als_grid_ndcg@10": {k: float(v) for k, v in als_val.items()},
        "als_selected": best_als,
        "metrics": val,
        "reranker_train_lists": ranker.n_train_lists,
        "reranker_objective_ndcg@10_heldout_users": obj_scores,
        "reranker_objective": best_obj,
        "candidate_recall@100_val": metrics.recall_at_k(A["cands"], val_truth, cfg.n_candidates),
    }

    # ---------- stage B: refit on train+val, evaluate on test ----------
    log("stage B: refit on train+val, evaluate on test")
    fit_df = pd.concat([split.train, split.val], ignore_index=True)
    # The two-tower is NOT refit: its user tower is a function of the watch history, so new events are served by
    # re-encoding histories (as in production). This also keeps the re-ranker's tt_score feature on the scale it was trained on.
    B = _stage(ds, fit_df, cfg, best_als, log, tt=None if refit_two_tower else A["tt"])
    X = B["X"]
    als_scores = B["als_models"][json.dumps(best_als, sort_keys=True)]
    recs = {
        "Popularity": metrics.top_k(B["pop"], X, 50),
        "Item-kNN": metrics.top_k(B["knn"], X, 50),
        "ALS": metrics.top_k(als_scores, X, 50),
        "Two-tower (exact)": metrics.top_k(B["tt_scores"], X, 50),
        "Two-tower (HNSW)": B["cands"][:, :50],
    }
    XB = features(B["ctx"], users, B["cands"])
    rr_scores = ranker.score(XB)
    recs["Two-tower + re-ranker"] = rerank(B["cands"], rr_scores)[:, :50]
    XB_als = np.concatenate([XB, als_scores[users[:, None], B["cands"]][..., None]], axis=-1)
    recs["Two-tower + re-ranker (+ALS feature)"] = rerank(B["cands"], ranker_als.score(XB_als))[:, :50]
    test = {name: ev(r, test_truth, X) for name, r in recs.items()}
    res["test"] = test
    res["retrieval"] = {
        "candidate_recall@100": metrics.recall_at_k(B["cands"], test_truth, cfg.n_candidates),
        "n_candidates": cfg.n_candidates,
        "hnsw_recall@100_vs_exact": ann.ann_recall(B["hnsw"], B["flat"], B["U"][: min(n_u, 2000)], 100),
        "hnsw_build_seconds": B["hnsw_build_s"],
        "hnsw": {"M": 32, "efConstruction": 200, "efSearch": 128},
    }
    res["reranker"] = {
        "objective": best_obj,
        "feature_importance_gain": ranker.importance(),
        "feature_importance_gain_with_als": ranker_als.importance(),
        "n_train_lists": ranker.n_train_lists,
        "trained_on": "stage-A candidates (two-tower trained on train) with validation labels",
    }
    res["two_tower"] = {"config": asdict(cfg.tt), "training": A["tt"].history, "refit_for_test": refit_two_tower}

    # paired bootstrap of NDCG@10 differences vs ALS over users
    b = metrics.per_user_ndcg(recs["ALS"], test_truth, 10)
    for key, name in (
        ("ndcg10_rerank_minus_als", "Two-tower + re-ranker"),
        ("ndcg10_rerank_als_feature_minus_als", "Two-tower + re-ranker (+ALS feature)"),
    ):
        a = metrics.per_user_ndcg(recs[name], test_truth, 10)
        rng = np.random.default_rng(cfg.seed)
        diffs = np.array([(a - b)[rng.integers(0, n_u, n_u)].mean() for _ in range(cfg.bootstrap)])
        res[key] = {
            "mean": float((a - b).mean()),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "bootstrap": cfg.bootstrap,
            "relative_lift": float((a.mean() - b.mean()) / b.mean()),
        }

    # MMR diversity re-ranking (stretch goal)
    mm = {}
    for lam in cfg.mmr_lambdas:
        r = mmr(B["cands"], rr_scores, B["I"], k=10, lam=lam)
        m = ev(r, test_truth, X)
        mm[str(lam)] = {k: m[k] for k in ("ndcg@10", "recall@10", "ild@10", "coverage@10", "novelty@10", "head_share@10")}
    res["mmr"] = mm

    # popularity of what users actually watched next (reference for popularity-bias analysis)
    pop_fit = np.asarray(X.sum(0)).ravel()
    tr, tc = test_truth.nonzero()
    order = np.argsort(-pop_fit)
    head = np.zeros(n_i, dtype=bool)
    head[order[: int(round(0.1 * n_i))]] = True
    res["test_reference"] = {
        "arp": float(pop_fit[tc].mean()),
        "head_share": float(head[tc].mean()),
        "novelty": float((-np.log2((pop_fit[tc] + 1) / (n_u + 1))).mean()),
    }

    # latency: one request = user tower forward on its history + HNSW top-100
    rng = np.random.default_rng(cfg.seed + 1)
    lat_users = rng.integers(0, n_u, cfg.latency_requests)
    tt, hist = B["tt"], B["hist"]
    torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # one request = one thread for both the user tower and the FAISS search
    lat = ann.measure_latency(lambda u: tt.user_embedding_one(u, hist[u]), B["hnsw"], lat_users, k=100)
    lat_flat = ann.measure_latency(lambda u: tt.user_embedding_one(u, hist[u]), B["flat"], lat_users, k=100)
    torch.set_num_threads(torch_threads)
    res["latency"] = {
        "hnsw": {k: v for k, v in lat.items() if k != "samples_ms"},
        "exact_flat": {k: v for k, v in lat_flat.items() if k != "samples_ms"},
        "threads": 1,
    }

    art = {
        "B": B,
        "ranker": ranker,
        "als_scores": als_scores,
        "rr_scores": rr_scores,
        "recs": recs,
        "latency_samples": lat["samples_ms"],
        "test_truth": test_truth,
        "split": split,
    }
    return res, art


def ope_suite(ds: Dataset, art: dict, cfg: RunConfig, log=print) -> tuple[dict, dict]:
    oc = cfg.ope
    world = ds.world
    B, ranker = art["B"], art["ranker"]
    n_u, n_i = ds.n_users, ds.n_items
    users = np.arange(n_u)
    rng = np.random.default_rng(oc.seed)
    df = ds.interactions
    seen_all = np.zeros((n_u, n_i), dtype=bool)
    seen_all[df["user"].to_numpy(), df["item"].to_numpy()] = True
    pop = np.asarray(B["X"].sum(0)).ravel()
    C = ope.eligible_slates(seen_all, pop, oc.slate_size, rng)
    u = users[:, None]
    Xf = features(B["ctx"], users, C)
    score_sets = {
        "Popularity": np.log1p(pop[C]) + 1e-6 * rng.normal(size=C.shape),
        "ALS": art["als_scores"][u, C],
        "Two-tower": B["tt_scores"][u, C],
        "Two-tower + re-ranker": ranker.score(Xf),
    }
    pi0 = ope.logging_policy(score_sets["Popularity"], oc)
    policies = ope.target_policies(score_sets, oc, greedy=("Two-tower + re-ranker",))
    p_true = 1.0 / (1.0 + np.exp(-(oc.reward_c0 + oc.reward_c1 * world.affinity_now(users, C))))
    truth = ope.true_values(policies, p_true)
    truth_logging = float((pi0 * p_true).sum(1).mean())
    oracle = float(p_true.max(1).mean())
    reward_X = np.concatenate([Xf[..., [0, 1, 4, 5]], score_sets["ALS"][..., None], score_sets["Two-tower + re-ranker"][..., None]], axis=-1).astype(
        np.float32
    )

    log(f"OPE: {len(policies)} target policies, {oc.n_replicates} logs x {oc.n_rounds} rounds")
    t0 = time.perf_counter()
    main = ope.replicate(oc.n_rounds, oc.n_replicates, pi0, p_true, policies, reward_X, oc.seed)
    summary = ope.summarise(main, truth)
    log(f"  main replicates done ({time.perf_counter() - t0:.1f}s)")
    sweep = {str(oc.n_rounds): {name: {e: summary[name][e]["rmse"] for e in ope.ESTIMATORS} for name in policies}}
    for n in oc.sweep_rounds:
        t0 = time.perf_counter()
        r = ope.replicate(n, oc.sweep_replicates, pi0, p_true, policies, reward_X, oc.seed + 7 * n)
        s = ope.summarise(r, truth)
        sweep[str(n)] = {name: {e: s[name][e]["rmse"] for e in ope.ESTIMATORS} for name in policies}
        log(f"  sweep n={n} done ({time.perf_counter() - t0:.1f}s)")
    sweep = dict(sorted(sweep.items(), key=lambda kv: int(kv[0])))

    # which estimator is closest to the truth, per policy and overall
    names = list(policies)
    closest = {name: min(ope.ESTIMATORS, key=lambda e: summary[name][e]["rmse"]) for name in names}
    mean_rel_rmse = {e: float(np.mean([summary[name][e]["rel_rmse"] for name in names])) for e in ope.ESTIMATORS}
    tv = np.array([truth[n] for n in names])
    taus, picks = {}, {}
    for e in ope.ESTIMATORS:
        est = np.stack([main[n][e] for n in names], 1)  # rep x policies
        taus[e] = float(np.mean([kendalltau(row, tv)[0] for row in est]))
        picks[e] = float(np.mean(est.argmax(1) == tv.argmax()))
    first = {n: {e: float(main[n][e][0]) for e in ope.ESTIMATORS} for n in names}
    res = {
        "setting": {
            **asdict(oc),
            "n_users": n_u,
            "mean_true_play_rate_on_slates": float(p_true.mean()),
            "logging_policy": f"(1-{oc.epsilon}) * softmax(z(log popularity)/{oc.log_temperature}) + {oc.epsilon}/K",
            "min_propensity": float(pi0.min()),
            "reward_model": "LightGBM classifier, 2-fold cross-fitted, features: tt_score, log_pop, genre_affinity, knn_score, als_score, reranker_score",
        },
        "truth": {**truth, "Logging policy (popularity)": truth_logging, "Oracle (true best title)": oracle},
        "estimators": summary,
        "closest_estimator_by_rmse": closest,
        "mean_relative_rmse": mean_rel_rmse,
        "kendall_tau_vs_truth": taus,
        "share_of_logs_picking_true_best": picks,
        "rmse_vs_rounds": sweep,
        "single_log_example": first,
    }
    return res, {"main": main, "truth": truth, "names": names}


def write_sample(ds: Dataset, out: Path, n_users: int = 300) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    tag = "SYNTHETIC" if ds.world is not None else "MOVIELENS"
    df = ds.interactions[ds.interactions["user"] < n_users].copy()
    df["t"] = df["t"].round(5)
    p1 = out / f"interactions_{tag}_sample.csv"
    df.to_csv(p1, index=False)
    genres = ["|".join(str(g) for g in np.flatnonzero(row)) for row in ds.item_genres]
    p2 = out / f"items_{tag}.csv"
    pd.DataFrame({"item": np.arange(ds.n_items), "genres": genres, "bucket": ds.item_bucket}).to_csv(p2, index=False)
    return [str(p1), str(p2)]


def run(cfg: RunConfig, out_dir: Path, data_dir: Path, make_figures: bool = True, sample_dir: Path | None = None, log=print) -> dict:
    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    ds, info = load_dataset(cfg.data_mode, data_dir / "raw", cfg.world)
    log(f"data path: {info['path_used']} ({len(ds.interactions)} interactions); MovieLens attempt: {info.get('movielens_attempt', 'skipped')}")
    if sample_dir is not None:
        info["sample_files"] = [Path(p).name for p in write_sample(ds, sample_dir)]
    timings = {}
    t0 = time.perf_counter()
    offline, art = offline_suite(ds, cfg, log)
    timings["offline_seconds"] = time.perf_counter() - t0
    world_ds, world_art = ds, art
    if ds.world is None:  # MovieLens path: OPE needs ground truth, so it runs on the synthetic world
        from .data import generate_world

        world_ds = generate_world(cfg.world)
        _, world_art = offline_suite(world_ds, cfg, log)
    t0 = time.perf_counter()
    ope_res, ope_art = ope_suite(world_ds, world_art, cfg, log)
    timings["ope_seconds"] = time.perf_counter() - t0
    result = {
        "data": {
            **info,
            "label": "SYNTHETIC" if ds.world is not None else "public MovieLens-1M",
            "world_seed": cfg.world.seed,
            "ope_world": "synthetic_generator",
        },
        "offline": offline,
        "ope": ope_res,
        "timings": timings,
    }
    if make_figures:
        from . import figures, report

        figures.make_all(result, art, ope_art, out_dir / "figures")
        result["timings"]["total_seconds"] = time.perf_counter() - t_start
        (out_dir / "report.md").write_text(report.render(result))
    result["timings"]["total_seconds"] = time.perf_counter() - t_start
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return result
