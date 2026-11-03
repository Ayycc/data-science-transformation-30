"""Off-policy evaluation on the SYNTHETIC world, where the true online reward of every policy is known.

Setting (one homepage hero slot, a contextual bandit):

* each user has an eligible slate ``C_u`` of ``K`` unwatched titles (drawn proportionally to popularity^0.5,
  standing in for an upstream eligibility filter that is independent of the candidate models);
* the logging policy is today's popularity-based homepage with epsilon-uniform exploration, so every
  propensity is known and bounded below by ``epsilon / K``;
* reward = play, drawn from ``sigmoid(c0 + c1 * true_affinity)`` using the world's hidden current tastes.

Estimators: direct method (DM, cross-fitted LightGBM reward model), IPS, self-normalised IPS (SNIPS) and doubly robust
(DR). Each is replayed over many independent logs to measure bias, standard deviation and RMSE against the truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

ESTIMATORS = ["DM", "IPS", "SNIPS", "DR"]


@dataclass
class OPEConfig:
    slate_size: int = 30
    log_temperature: float = 0.5
    epsilon: float = 0.2
    target_temperature: float = 0.3
    reward_c0: float = -3.0
    reward_c1: float = 0.8
    n_rounds: int = 20000
    n_replicates: int = 200
    sweep_rounds: tuple = (2000, 5000, 50000)
    sweep_replicates: int = 30
    seed: int = 2024


def softmax_rows(z: np.ndarray, temperature: float) -> np.ndarray:
    z = z / temperature
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def zscore_rows(s: np.ndarray) -> np.ndarray:
    return (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True).clip(1e-9)


def eligible_slates(seen_mask: np.ndarray, item_pop: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """``k`` unwatched titles per user, sampled without replacement with weight popularity^0.5 (Gumbel top-k)."""
    w = np.log(np.sqrt(item_pop + 1.0))[None, :] + rng.gumbel(size=seen_mask.shape)
    w[seen_mask] = -np.inf
    return np.argpartition(-w, kth=k, axis=1)[:, :k]


def logging_policy(pop_scores: np.ndarray, cfg: OPEConfig) -> np.ndarray:
    k = pop_scores.shape[1]
    return (1 - cfg.epsilon) * softmax_rows(zscore_rows(pop_scores), cfg.log_temperature) + cfg.epsilon / k


def target_policies(score_sets: dict[str, np.ndarray], cfg: OPEConfig, greedy: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Softmax policies over each model's per-slate z-scored scores; names in ``greedy`` also get an argmax policy."""
    pols = {}
    for name, s in score_sets.items():
        pols[name] = softmax_rows(zscore_rows(s), cfg.target_temperature)
        if name in greedy:
            g = np.zeros_like(s)
            g[np.arange(len(s)), s.argmax(1)] = 1.0
            pols[f"{name} (greedy)"] = g
    k = next(iter(score_sets.values())).shape[1]
    pols["Uniform random"] = np.full_like(next(iter(score_sets.values())), 1.0 / k)
    return pols


def true_values(policies: dict[str, np.ndarray], p_true: np.ndarray) -> dict[str, float]:
    """Exact expected play rate of each policy when users arrive uniformly at random."""
    return {name: float((pi * p_true).sum(1).mean()) for name, pi in policies.items()}


def simulate_log(n: int, pi0: np.ndarray, p_true: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    users = rng.integers(0, pi0.shape[0], size=n)
    cdf = pi0[users].cumsum(1)
    actions = (rng.random(n)[:, None] > cdf).sum(1).clip(max=pi0.shape[1] - 1)
    rewards = (rng.random(n) < p_true[users, actions]).astype(np.float64)
    return users, actions, rewards


def _reward_model(X: np.ndarray, y: np.ndarray, seed: int) -> lgb.LGBMClassifier:
    m = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=4,
        verbose=-1,
    )
    m.fit(X, y)
    return m


def estimate(users, actions, rewards, pi0, policies, reward_X, seed: int, folds: int = 2) -> dict[str, dict[str, float]]:
    """DM / IPS / SNIPS / DR estimates for every target policy from one logged dataset.

    ``reward_X`` (n_users x K x F) are the features of every (user, slate item) pair for the reward model;
    DM and DR use 2-fold cross-fitting so no round is scored by a model that saw it.
    """
    n = len(users)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, size=n)
    k = pi0.shape[1]
    f = reward_X.shape[2]
    q_all = np.zeros((n, k))
    for j in range(folds):
        tr, te = fold != j, fold == j
        m = _reward_model(reward_X[users[tr], actions[tr]], rewards[tr], seed + j)
        q_grid = m.predict_proba(reward_X.reshape(-1, f))[:, 1].reshape(-1, k)  # every (user, slate item)
        q_all[te] = q_grid[users[te]]
    q_logged = q_all[np.arange(n), actions]
    p0 = pi0[users, actions]
    out: dict[str, dict[str, float]] = {}
    for name, pi in policies.items():
        w = pi[users, actions] / p0
        dm = (pi[users] * q_all).sum(1)
        out[name] = {
            "DM": float(dm.mean()),
            "IPS": float((w * rewards).mean()),
            "SNIPS": float((w * rewards).sum() / w.sum()),
            "DR": float((dm + w * (rewards - q_logged)).mean()),
            "max_weight": float(w.max()),
            "ess": float(w.sum() ** 2 / (w**2).sum()),
        }
    return out


def replicate(n_rounds: int, n_rep: int, pi0, p_true, policies, reward_X, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Run ``n_rep`` independent logs; return estimates[policy][estimator] arrays of length ``n_rep``."""
    res = {name: {e: np.zeros(n_rep) for e in [*ESTIMATORS, "ess"]} for name in policies}
    for r in range(n_rep):
        rng = np.random.default_rng(seed + 1000 * r)
        u, a, y = simulate_log(n_rounds, pi0, p_true, rng)
        est = estimate(u, a, y, pi0, policies, reward_X, seed=seed + r)
        for name in policies:
            for e in [*ESTIMATORS, "ess"]:
                res[name][e][r] = est[name][e]
    return res


def summarise(res: dict[str, dict[str, np.ndarray]], truth: dict[str, float]) -> dict:
    out = {}
    for name, ests in res.items():
        tv = truth[name]
        row = {"truth": tv, "mean_ess": float(ests["ess"].mean())}
        for e in ESTIMATORS:
            x = ests[e]
            row[e] = {
                "mean": float(x.mean()),
                "bias": float(x.mean() - tv),
                "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                "rmse": float(np.sqrt(((x - tv) ** 2).mean())),
                "rel_rmse": float(np.sqrt(((x - tv) ** 2).mean()) / tv),
            }
        out[name] = row
    return out
