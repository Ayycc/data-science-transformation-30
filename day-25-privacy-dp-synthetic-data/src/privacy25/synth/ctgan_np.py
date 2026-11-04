"""CTGAN-style conditional tabular GAN in NumPy (not DP).

It follows Xu et al. 2019 (the algorithm behind SDV's ``CTGANSynthesizer``): mode-specific normalisation of
numerics with a variational Gaussian mixture, a conditional vector over the discrete columns with
training-by-sampling (log-frequency), a generator cross-entropy term that forces the conditioned category,
Gumbel-softmax outputs (tau = 0.2) and PacGAN (pac = 10) in the discriminator.
Differences from the reference implementation: a non-saturating BCE GAN loss instead of WGAN-GP, plain MLPs
(no batch norm or residual concatenation), no discriminator dropout and hand-written backpropagation. The PyTorch package could not
be installed here (see README), so the network is written directly in NumPy.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

from ..data import CATEGORICAL, COLUMNS, NUMERIC_BOUNDS


class _Linear:
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator) -> None:
        bound = 1 / np.sqrt(n_in)
        self.W = rng.uniform(-bound, bound, (n_in, n_out)).astype(np.float32)
        self.b = np.zeros(n_out, dtype=np.float32)
        self.params = [self.W, self.b]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b

    def backward(self, g: np.ndarray) -> np.ndarray:
        self.grads = [self.x.T @ g, g.sum(0)]
        return g @ self.W.T


class _Adam:
    def __init__(self, layers: list[_Linear], lr: float, betas: tuple[float, float], wd: float) -> None:
        self.layers, self.lr, self.b1, self.b2, self.wd, self.t = layers, lr, betas[0], betas[1], wd, 0

    def step(self) -> None:
        self.t += 1
        c1, c2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        for layer in self.layers:
            for p, g, m, v in zip(layer.params, layer.grads, layer.m, layer.v):
                g = g + self.wd * p
                m *= self.b1
                m += (1 - self.b1) * g
                v *= self.b2
                v += (1 - self.b2) * g * g
                p -= self.lr * (m / c1) / (np.sqrt(v / c2) + 1e-8)


class _MLP:
    """Linear -> act -> [dropout] ... -> Linear. act is ReLU (generator) or LeakyReLU 0.2 (discriminator)."""

    def __init__(self, sizes: list[int], rng: np.random.Generator, leaky: float, dropout: float) -> None:
        self.layers = [_Linear(a, b, rng) for a, b in zip(sizes[:-1], sizes[1:])]
        self.leaky, self.dropout, self.rng = leaky, dropout, rng

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        self.cache = []
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                pos = x > 0
                x = np.where(pos, x, self.leaky * x)
                mask = None
                if self.dropout and train:
                    mask = (self.rng.random(x.shape) >= self.dropout).astype(np.float32) / (1 - self.dropout)
                    x = x * mask
                self.cache.append((pos, mask))
        return x

    def backward(self, g: np.ndarray) -> np.ndarray:
        for i in range(len(self.layers) - 1, -1, -1):
            if i < len(self.layers) - 1:
                pos, mask = self.cache[i]
                if mask is not None:
                    g = g * mask
                g = np.where(pos, g, self.leaky * g)
            g = self.layers[i].backward(g)
        return g


def _bce_logits(logits: np.ndarray, target: float) -> tuple[float, np.ndarray]:
    """Mean BCE with logits and its gradient."""
    if target == 1.0:
        loss = np.logaddexp(0, -logits)
        grad = -1 / (1 + np.exp(logits))
    else:
        loss = np.logaddexp(0, logits)
        grad = 1 / (1 + np.exp(-logits))
    return float(loss.mean()), (grad / logits.size).astype(np.float32)


@dataclass
class CTGANLite:
    epochs: int = 300
    batch_size: int = 500
    z_dim: int = 128
    hidden: int = 256
    pac: int = 10
    lr: float = 2e-4
    tau: float = 0.2
    d_dropout: float = 0.0  # CTGAN uses 0.5; with the BCE loss here 0.0 gave lower marginal error in a 60-epoch pilot
    n_modes: int = 10
    seed: int = 0
    name: str = "CTGAN-style GAN"

    # ---------- data transformer ----------
    def _fit_transformer(self, df: pd.DataFrame) -> None:
        self.spans_: list[tuple[str, str, int, int]] = []  # (column, kind, start, width) kind in tanh / softmax
        self.gmm_: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        pos = 0
        for c in COLUMNS:
            if c in NUMERIC_BOUNDS:
                v = df[c].to_numpy(dtype=float)[:, None]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    g = BayesianGaussianMixture(
                        n_components=self.n_modes,
                        weight_concentration_prior_type="dirichlet_process",
                        weight_concentration_prior=1e-3,
                        max_iter=200,
                        random_state=self.seed,
                    ).fit(v)
                keep = np.where(g.weights_ > 0.005)[0]
                mu = g.means_[keep, 0]
                sd = np.sqrt(g.covariances_[keep, 0, 0])
                self.gmm_[c] = (g, keep, np.column_stack([mu, sd]))
                self.spans_.append((c, "tanh", pos, 1))
                self.spans_.append((c, "softmax", pos + 1, len(keep)))
                pos += 1 + len(keep)
            else:
                k = len(CATEGORICAL[c])
                self.spans_.append((c, "softmax", pos, k))
                pos += k
        self.dim_ = pos
        self.cat_spans_ = [(c, s, w) for c, kind, s, w in self.spans_ if kind == "softmax" and c in CATEGORICAL]
        self.cond_dim_ = sum(w for _, _, w in self.cat_spans_)

    def _transform(self, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        out = np.zeros((len(df), self.dim_), dtype=np.float32)
        for c, kind, s, _w in self.spans_:
            if kind == "tanh":
                g, keep, ms = self.gmm_[c]
                v = df[c].to_numpy(dtype=float)
                p = g.predict_proba(v[:, None])[:, keep] + 1e-9
                p = p / p.sum(1, keepdims=True)
                mode = np.minimum((p.cumsum(1) < rng.random(len(v))[:, None]).sum(1), len(keep) - 1)
                alpha = (v - ms[mode, 0]) / (4 * ms[mode, 1])
                out[:, s] = np.clip(alpha, -0.99, 0.99)
                out[np.arange(len(v)), s + 1 + mode] = 1.0
            elif c in CATEGORICAL:
                code = pd.Categorical(df[c], categories=CATEGORICAL[c]).codes
                out[np.arange(len(df)), s + code] = 1.0
        return out

    # ---------- generator output ----------
    def _activate(self, logits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = np.empty_like(logits)
        for _, kind, s, w in self.spans_:
            seg = logits[:, s : s + w]
            if kind == "tanh":
                out[:, s : s + w] = np.tanh(seg)
            else:
                g = -np.log(-np.log(rng.random(seg.shape, dtype=np.float32) * 0.999998 + 1e-6))
                y = (seg + g) / self.tau
                y = np.exp(y - y.max(1, keepdims=True))
                out[:, s : s + w] = y / y.sum(1, keepdims=True)
        return out

    def _activate_backward(self, act: np.ndarray, g_act: np.ndarray) -> np.ndarray:
        g = np.empty_like(g_act)
        for _, kind, s, w in self.spans_:
            a, ga = act[:, s : s + w], g_act[:, s : s + w]
            if kind == "tanh":
                g[:, s : s + w] = ga * (1 - a * a)
            else:
                g[:, s : s + w] = a * (ga - (ga * a).sum(1, keepdims=True)) / self.tau
        return g

    # ---------- conditional sampling ----------
    def _sample_cond(self, n: int, rng: np.random.Generator, log_freq: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cols = rng.integers(0, len(self.cat_spans_), n)
        cond = np.zeros((n, self.cond_dim_), dtype=np.float32)
        cats = np.zeros(n, dtype=np.int64)
        for j in range(len(self.cat_spans_)):
            idx = np.where(cols == j)[0]
            if len(idx) == 0:
                continue
            p = self.cat_logp_[j] if log_freq else self.cat_p_[j]
            cats[idx] = rng.choice(len(p), size=len(idx), p=p)
            cond[idx, self.cond_offsets_[j] + cats[idx]] = 1.0
        return cond, cols, cats

    def fit(self, df: pd.DataFrame, log=None) -> CTGANLite:
        rng = np.random.default_rng(self.seed)
        self._fit_transformer(df)
        data = self._transform(df, rng)
        n = len(data)
        self.cond_offsets_ = np.cumsum([0] + [w for _, _, w in self.cat_spans_])[:-1]
        self.cat_p_, self.cat_logp_, order, starts, counts = [], [], [], [], []
        for _c, s, w in self.cat_spans_:
            cnt = data[:, s : s + w].sum(0)
            self.cat_p_.append(cnt / cnt.sum())
            lp = np.log(cnt + 1)
            self.cat_logp_.append(lp / lp.sum())
            code = data[:, s : s + w].argmax(1)
            order.append(np.argsort(code, kind="stable"))
            counts.append(cnt.astype(np.int64))
            starts.append(np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64))

        def real_rows(cols: np.ndarray, cats: np.ndarray) -> np.ndarray:
            idx = np.empty(len(cols), dtype=np.int64)
            for j in range(len(self.cat_spans_)):
                m = cols == j
                k = cats[m]
                idx[m] = order[j][starts[j][k] + (rng.random(m.sum()) * counts[j][k]).astype(np.int64)]
            return idx

        G = _MLP([self.z_dim + self.cond_dim_, self.hidden, self.hidden, self.dim_], rng, leaky=0.0, dropout=0.0)
        D = _MLP([self.pac * (self.dim_ + self.cond_dim_), self.hidden, self.hidden, 1], rng, leaky=0.2, dropout=self.d_dropout)
        optG = _Adam(G.layers, self.lr, (0.5, 0.9), 1e-6)
        optD = _Adam(D.layers, self.lr, (0.5, 0.9), 1e-6)
        bs = self.batch_size
        steps = max(1, n // bs)
        self.history_ = []
        for ep in range(self.epochs):
            ld = lg = 0.0
            for _ in range(steps):
                # ---- discriminator step ----
                cond, cols, cats = self._sample_cond(bs, rng, log_freq=True)
                real_idx = real_rows(cols, cats)
                real = data[real_idx]
                z = rng.standard_normal((bs, self.z_dim), dtype=np.float32)
                fake = self._activate(G.forward(np.hstack([z, cond])), rng)
                d_real = D.forward(np.hstack([real, cond]).reshape(bs // self.pac, -1))
                l1, g1 = _bce_logits(d_real, 1.0)
                D.backward(g1)
                gr = [list(layer.grads) for layer in D.layers]
                d_fake = D.forward(np.hstack([fake, cond]).reshape(bs // self.pac, -1))
                l2, g2 = _bce_logits(d_fake, 0.0)
                D.backward(g2)
                for layer, g_prev in zip(D.layers, gr):
                    layer.grads = [a + b for a, b in zip(layer.grads, g_prev)]
                optD.step()
                ld += l1 + l2
                # ---- generator step ----
                cond, cols, cats = self._sample_cond(bs, rng, log_freq=True)
                z = rng.standard_normal((bs, self.z_dim), dtype=np.float32)
                logits = G.forward(np.hstack([z, cond]))
                fake = self._activate(logits, rng)
                d_fake = D.forward(np.hstack([fake, cond]).reshape(bs // self.pac, -1))
                l3, g3 = _bce_logits(d_fake, 1.0)
                g_in = D.backward(g3).reshape(bs, -1)[:, : self.dim_]
                g_logits = self._activate_backward(fake, g_in)
                # conditional cross-entropy on the conditioned column (raw logits, as in CTGAN)
                ce = 0.0
                for j, (_, s, w) in enumerate(self.cat_spans_):
                    idx = np.where(cols == j)[0]
                    if len(idx) == 0:
                        continue
                    seg = logits[idx, s : s + w]
                    p = np.exp(seg - seg.max(1, keepdims=True))
                    p /= p.sum(1, keepdims=True)
                    ce += float(-np.log(p[np.arange(len(idx)), cats[idx]] + 1e-9).sum()) / bs
                    p[np.arange(len(idx)), cats[idx]] -= 1.0
                    g_logits[idx, s : s + w] += p / bs
                G.backward(g_logits)
                optG.step()
                lg += l3 + ce
            self.history_.append((ld / steps, lg / steps))
            if log and (ep + 1) % 50 == 0:
                log(f"  CTGAN epoch {ep + 1}/{self.epochs}: D loss {ld / steps:.3f}, G loss {lg / steps:.3f}")
        self.G_ = G
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + 1 if seed is None else seed)
        rows = []
        for s0 in range(0, n, self.batch_size):
            m = min(self.batch_size, n - s0)
            cond, _, _ = self._sample_cond(m, rng, log_freq=False)
            z = rng.standard_normal((m, self.z_dim), dtype=np.float32)
            rows.append(self.G_.forward(np.hstack([z, cond]), train=False))
        logits = np.vstack(rows)
        cols = {}
        for c, kind, s, w in self.spans_:
            seg = logits[:, s : s + w]
            if kind == "tanh":
                continue
            g = -np.log(-np.log(rng.random(seg.shape) * 0.999998 + 1e-6))
            pick = np.argmax(seg + g, axis=1)  # Gumbel-max = a draw from softmax(logits)
            if c in NUMERIC_BOUNDS:
                _, _, ms = self.gmm_[c]
                tanh_s = [sp for sp in self.spans_ if sp[0] == c and sp[1] == "tanh"][0][2]
                alpha = np.tanh(logits[:, tanh_s])
                v = alpha * 4 * ms[pick, 1] + ms[pick, 0]
                lo, hi = NUMERIC_BOUNDS[c]
                cols[c] = np.clip(np.round(v), lo, hi).astype(int)
            else:
                cols[c] = np.asarray(CATEGORICAL[c], dtype=object)[pick]
        return pd.DataFrame(cols)[COLUMNS]
