"""Data: MovieLens-1M loader (if reachable) and a seeded SYNTHETIC streaming world.

Both paths return a :class:`Dataset` with the same schema, so every model runs unchanged:

* ``interactions``: DataFrame ``user, item, t`` of implicit positive events, ``t`` normalised to [0, 1].
* ``item_genres``: ``n_items x n_genres`` multi-hot matrix (content features).
* ``item_bucket``: an integer release-period bucket per item (content feature).

The synthetic world also exposes its ground truth (user tastes, item factors, item quality) so that the
off-policy evaluation can compute the true online reward of any policy exactly.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


@dataclass
class WorldConfig:
    """Parameters of the SYNTHETIC streaming world (sized like MovieLens-1M)."""

    n_users: int = 6000
    n_items: int = 3000
    dim: int = 16
    n_genres: int = 18
    n_buckets: int = 10
    mean_events: float = 60.0
    min_events: int = 16
    max_events: int = 300
    taste_scale: float = 8.0  # weight of user-item affinity in the choice model
    bias_sd: float = 0.7  # spread of item quality / popularity
    fresh_boost: float = 0.7  # extra appeal of a newly released title
    fresh_decay: float = 0.08  # how fast the release boost fades (in normalised time)
    drift: float = 0.5  # fraction of a quarter-turn that each user's taste rotates over their lifetime
    seed: int = 20261103


@dataclass
class World:
    """Ground truth of the synthetic world."""

    cfg: WorldConfig
    user_start: np.ndarray  # taste at the start of the user's window (n_users x dim)
    user_end: np.ndarray  # taste at the end of the window
    item_factors: np.ndarray  # n_items x dim
    item_bias: np.ndarray  # n_items
    item_release: np.ndarray  # n_items, normalised time

    def affinity_now(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Latent utility of ``items`` (users x K) for each user's *current* (end-of-window) taste."""
        u = self.user_end[users]  # n x d
        v = self.item_factors[items]  # n x K x d
        return self.cfg.taste_scale * np.einsum("nd,nkd->nk", u, v) + self.item_bias[items]


@dataclass
class Dataset:
    name: str
    path_used: str
    interactions: pd.DataFrame
    n_users: int
    n_items: int
    item_genres: np.ndarray
    item_bucket: np.ndarray
    world: World | None = None
    notes: dict = field(default_factory=dict)


def _unit(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.linalg.norm(x, axis=axis, keepdims=True).clip(1e-9)


def generate_world(cfg: WorldConfig | None = None) -> Dataset:
    """Simulate ``cfg.n_users`` viewers choosing titles over time (SYNTHETIC).

    Choice model at each event: utility = taste_scale * <u(t), v_i> + b_i + freshness_i(t) + Gumbel noise,
    over titles already released and not yet watched. ``u(t)`` interpolates between a start and an end taste,
    so recent history is more predictive than old history; new releases get a decaying boost.
    """
    cfg = cfg or WorldConfig()
    rng = np.random.default_rng(cfg.seed)
    d, n_i, n_u, n_g = cfg.dim, cfg.n_items, cfg.n_users, cfg.n_genres

    centroids = _unit(rng.normal(size=(n_g, d)))
    genre_share = rng.dirichlet(np.full(n_g, 1.5))
    primary = rng.choice(n_g, size=n_i, p=genre_share)
    item_factors = _unit(0.8 * centroids[primary] + 0.6 * rng.normal(size=(n_i, d)) / np.sqrt(d) * 2.0)
    # multi-hot genres: primary plus 0-2 extra genres, drawn by similarity of the title to each genre centroid
    affin = item_factors @ centroids.T * 4.0
    probs = np.exp(affin - affin.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    item_genres = np.zeros((n_i, n_g), dtype=np.float32)
    item_genres[np.arange(n_i), primary] = 1.0
    n_extra = rng.choice(3, size=n_i, p=[0.4, 0.4, 0.2])
    for i in np.flatnonzero(n_extra):
        extra = rng.choice(n_g, size=n_extra[i], replace=False, p=probs[i])
        item_genres[i, extra] = 1.0
    item_bias = rng.normal(0.0, cfg.bias_sd, size=n_i)
    item_release = rng.uniform(-1.0, 0.95, size=n_i)
    item_bucket = np.clip(((item_release + 1.0) / 1.95 * cfg.n_buckets).astype(int), 0, cfg.n_buckets - 1)

    # users: a taste near one or two genres, which rotates over the user's active window
    g1 = rng.choice(n_g, size=n_u, p=genre_share)
    g2 = rng.choice(n_g, size=n_u, p=genre_share)
    mix = rng.uniform(0.5, 1.0, size=(n_u, 1))
    start = _unit(mix * centroids[g1] + (1 - mix) * centroids[g2] + 0.5 * rng.normal(size=(n_u, d)) / np.sqrt(d) * 2.0)
    w = rng.normal(size=(n_u, d))
    w = _unit(w - (w * start).sum(1, keepdims=True) * start)
    theta = cfg.drift * np.pi / 2
    end = np.cos(theta) * start + np.sin(theta) * w

    s_u = rng.uniform(0.0, 0.7, size=n_u)
    l_u = rng.uniform(0.15, 1.0 - s_u)
    sigma = 0.8
    n_ev = rng.lognormal(np.log(cfg.mean_events) - sigma**2 / 2, sigma, size=n_u)
    n_ev = np.clip(np.round(n_ev), cfg.min_events, cfg.max_events).astype(int)
    max_ev = int(n_ev.max())
    times = np.sort(rng.uniform(size=(n_u, max_ev)), axis=1) * l_u[:, None] + s_u[:, None]
    seen = np.zeros((n_u, n_i), dtype=bool)
    items = np.full((n_u, max_ev), -1, dtype=np.int64)
    vT = (cfg.taste_scale * item_factors.T).astype(np.float32)
    rel32, bias32 = item_release.astype(np.float32), item_bias.astype(np.float32)
    for k in range(max_ev):
        act = np.flatnonzero(n_ev > k)
        t = times[act, k].astype(np.float32)
        tau = ((t - s_u[act]) / l_u[act])[:, None].astype(np.float32)
        taste = ((1 - tau) * start[act] + tau * end[act]).astype(np.float32)
        age = t[:, None] - rel32[None, :]
        util = taste @ vT + bias32[None, :]
        util += np.float32(cfg.fresh_boost) * np.exp(-np.clip(age, 0, None) / np.float32(cfg.fresh_decay)) * (age >= 0)
        util[(age < 0) | seen[act]] = -np.inf
        util -= np.log(-np.log(rng.random(size=util.shape, dtype=np.float32).clip(1e-12, None)))  # Gumbel noise
        choice = util.argmax(1)
        items[act, k] = choice
        seen[act, choice] = True

    uu, kk = np.nonzero(items >= 0)
    df = pd.DataFrame({"user": uu.astype(np.int64), "item": items[uu, kk], "t": times[uu, kk]})
    df = df.sort_values(["user", "t"], kind="stable").reset_index(drop=True)
    world = World(cfg, start, end, item_factors, item_bias, item_release)
    return Dataset(
        name="synthetic_streaming_world",
        path_used="synthetic_generator",
        interactions=df,
        n_users=n_u,
        n_items=n_i,
        item_genres=item_genres,
        item_bucket=item_bucket,
        world=world,
        notes={"seed": cfg.seed, "label": "SYNTHETIC"},
    )


def parse_movielens_1m(ratings_text: str, movies_text: str, min_rating: int = 4, min_events: int = 16) -> Dataset:
    """Parse ML-1M ``ratings.dat`` / ``movies.dat`` text into a :class:`Dataset` of implicit positives."""
    r = pd.read_csv(io.StringIO(ratings_text), sep="::", engine="python", names=["uid", "mid", "rating", "ts"])
    m = pd.read_csv(io.StringIO(movies_text), sep="::", engine="python", names=["mid", "title", "genres"])
    r = r[r["rating"] >= min_rating]
    counts = r.groupby("uid")["mid"].transform("size")
    r = r[counts >= min_events]
    mids = np.sort(m["mid"].unique())
    item_index = {mid: i for i, mid in enumerate(mids)}
    uids = np.sort(r["uid"].unique())
    user_index = {u: i for i, u in enumerate(uids)}
    ts = r["ts"].to_numpy(dtype=float)
    t = (ts - ts.min()) / max(ts.max() - ts.min(), 1.0)
    df = pd.DataFrame({"user": r["uid"].map(user_index).to_numpy(), "item": r["mid"].map(item_index).to_numpy(), "t": t})
    df = df.sort_values(["user", "t"], kind="stable").reset_index(drop=True)
    genres = np.zeros((len(mids), len(ML_GENRES)), dtype=np.float32)
    year = np.full(len(mids), 1990, dtype=int)
    for mid, title, gs in m[["mid", "title", "genres"]].itertuples(index=False):
        i = item_index[mid]
        for g in str(gs).split("|"):
            if g in ML_GENRES:
                genres[i, ML_GENRES.index(g)] = 1.0
        s = str(title).strip()
        if s.endswith(")") and s[-5:-1].isdigit():
            year[i] = int(s[-5:-1])
    bucket = np.clip((year - 1930) // 7, 0, 9).astype(int)
    return Dataset(
        name="movielens_1m",
        path_used="public_movielens_1m",
        interactions=df,
        n_users=len(uids),
        n_items=len(mids),
        item_genres=genres,
        item_bucket=bucket,
        notes={"source": ML1M_URL, "positives": f"rating >= {min_rating}"},
    )


def try_download_movielens(cache_dir: Path, timeout: float = 20.0) -> tuple[Dataset | None, str]:
    """Return (dataset, reason). ``dataset`` is None when MovieLens is unreachable."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    zpath = cache_dir / "ml-1m.zip"
    try:
        if not zpath.exists():
            with urllib.request.urlopen(ML1M_URL, timeout=timeout) as resp:  # noqa: S310 (fixed public URL)
                zpath.write_bytes(resp.read())
        with zipfile.ZipFile(zpath) as z:
            ratings = z.read("ml-1m/ratings.dat").decode("latin-1")
            movies = z.read("ml-1m/movies.dat").decode("latin-1")
        return parse_movielens_1m(ratings, movies), "downloaded"
    except Exception as exc:  # network blocked, proxy 403, corrupt zip ...
        if zpath.exists() and zpath.stat().st_size < 1000:
            zpath.unlink()
        return None, f"{type(exc).__name__}: {exc}"[:300]


def load_dataset(mode: str | None = None, cache_dir: Path | None = None, world_cfg: WorldConfig | None = None) -> tuple[Dataset, dict]:
    """Load data according to ``mode`` (``auto`` | ``movielens`` | ``synthetic``; env ``RECSYS_DATA``)."""
    mode = (mode or os.environ.get("RECSYS_DATA", "auto")).lower()
    info: dict = {"mode": mode}
    if mode in ("auto", "movielens"):
        ds, reason = try_download_movielens(cache_dir or Path("data/raw"))
        info["movielens_attempt"] = reason
        if ds is not None:
            info["path_used"] = ds.path_used
            return ds, info
        if mode == "movielens":
            raise RuntimeError(f"MovieLens requested but unreachable: {reason}")
    ds = generate_world(world_cfg)
    info["path_used"] = ds.path_used
    return ds, info


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def temporal_split(df: pd.DataFrame, n_test: int = 5, n_val: int = 5) -> Split:
    """Per-user temporal leave-last-out: last ``n_test`` events -> test, the ``n_val`` before -> validation."""
    df = df.sort_values(["user", "t"], kind="stable")
    rank_from_end = df.groupby("user").cumcount(ascending=False)
    test = df[rank_from_end < n_test]
    val = df[(rank_from_end >= n_test) & (rank_from_end < n_test + n_val)]
    train = df[rank_from_end >= n_test + n_val]
    return Split(train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))
