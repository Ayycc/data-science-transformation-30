"""Two-tower retrieval model in PyTorch trained with in-batch sampled softmax (+ logQ correction).

* Item tower: item-id embedding + genre multi-hot projection + release-bucket embedding -> residual MLP -> L2-normalised.
* User tower: mean of the last ``hist_len`` watched titles and mean of the last 10 (optionally a user-id embedding) -> MLP.
* Training is causal: the target is the next title in a user's training sequence and the history is only what came
  before it, so the model learns next-title prediction (what the homepage has to do).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TwoTowerConfig:
    dim: int = 64
    hidden: int = 256
    hist_len: int = 50
    recent_len: int = 10
    batch_size: int = 1024
    epochs: int = 12
    lr: float = 3e-3
    weight_decay: float = 1e-4
    temperature: float = 0.1
    min_history: int = 3
    use_user_id: bool = False  # off: an id embedding memorised training users (validation NDCG@10 dropped)
    dropout: float = 0.3
    logq_correction: bool = True  # subtract log item frequency from in-batch logits (sampling-bias correction)
    tie_history: bool = True  # pool the item tower's own outputs over the history instead of separate embeddings
    seed: int = 7
    threads: int = 4


class _MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, item_genres: np.ndarray, item_bucket: np.ndarray, cfg: TwoTowerConfig):
        super().__init__()
        self.cfg = cfg
        self.n_items = n_items
        d = cfg.dim
        self.register_buffer("genres", torch.as_tensor(item_genres, dtype=torch.float32))
        self.register_buffer("bucket", torch.as_tensor(item_bucket, dtype=torch.long))
        self.item_id = nn.Embedding(n_items, d)
        self.item_genre = nn.Linear(item_genres.shape[1], d, bias=False)
        self.item_bucket = nn.Embedding(int(item_bucket.max()) + 1, d)
        self.item_mlp = _MLP(d, cfg.hidden, d)
        self.hist_emb = nn.Embedding(n_items + 1, d, padding_idx=n_items)
        self.user_id = nn.Embedding(n_users, d)
        self.user_mlp = _MLP((3 if cfg.use_user_id else 2) * d, cfg.hidden, d)
        self.drop = nn.Dropout(cfg.dropout)
        for emb in (self.item_id, self.hist_emb, self.user_id, self.item_bucket):
            nn.init.normal_(emb.weight, std=0.05)

    def item_vec(self, items: torch.Tensor) -> torch.Tensor:
        x = self.item_id(items) + self.item_genre(self.genres[items]) + self.item_bucket(self.bucket[items])
        return F.normalize(x + self.item_mlp(x), dim=-1)

    def item_table(self) -> torch.Tensor:
        """All item-tower outputs plus a zero row for the padding id."""
        return torch.cat([self.item_vec(torch.arange(self.n_items)), torch.zeros(1, self.cfg.dim)])

    def user_vec(self, users: torch.Tensor, hist: torch.Tensor, item_table: torch.Tensor | None = None) -> torch.Tensor:
        """``hist``: batch x hist_len item ids, most recent first, padded with ``n_items``.

        With tied history, serving passes a precomputed ``item_table`` so a request does not re-run the item tower.
        """
        mask = (hist != self.n_items).float().unsqueeze(-1)
        if self.cfg.tie_history:
            e = (self.item_table() if item_table is None else item_table)[hist]
        else:
            e = self.hist_emb(hist)
        pooled = (e * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        r = self.cfg.recent_len
        recent = (e[:, :r] * mask[:, :r]).sum(1) / mask[:, :r].sum(1).clamp(min=1.0)
        parts = [pooled, recent] + ([self.user_id(users)] if self.cfg.use_user_id else [])
        x = self.drop(torch.cat(parts, dim=-1))
        return F.normalize(self.user_mlp(x), dim=-1)


def build_sequences(df: pd.DataFrame, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    """Chronological padded item sequences (n_users x max_len, pad = -1) and their lengths."""
    df = df.sort_values(["user", "t"], kind="stable")
    lens = np.bincount(df["user"].to_numpy(), minlength=n_users)
    seq = np.full((n_users, max(int(lens.max()), 1)), -1, dtype=np.int64)
    pos = df.groupby("user").cumcount().to_numpy()
    seq[df["user"].to_numpy(), pos] = df["item"].to_numpy()
    return seq, lens


def recent_history(seq: np.ndarray, lens: np.ndarray, hist_len: int, pad: int) -> np.ndarray:
    """Last ``hist_len`` items of each sequence, most recent first, padded with ``pad``."""
    idx = lens[:, None] - 1 - np.arange(hist_len)[None, :]
    valid = idx >= 0
    out = np.take_along_axis(seq, np.clip(idx, 0, None), 1)
    return np.where(valid, out, pad)


class TwoTowerRecommender:
    def __init__(self, n_users: int, n_items: int, item_genres: np.ndarray, item_bucket: np.ndarray, cfg: TwoTowerConfig | None = None):
        self.cfg = cfg or TwoTowerConfig()
        self.n_users, self.n_items = n_users, n_items
        self.item_genres, self.item_bucket = item_genres, item_bucket
        self.model: TwoTower | None = None
        self.history: dict = {}

    def fit(self, train_df: pd.DataFrame) -> TwoTowerRecommender:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        torch.use_deterministic_algorithms(True)  # bit-identical embeddings across runs (CPU scatter-adds otherwise race)
        torch.set_num_threads(cfg.threads)
        rng = np.random.default_rng(cfg.seed)
        seq, lens = build_sequences(train_df, self.n_users)
        users, positions = np.nonzero(seq >= 0)
        keep = positions >= cfg.min_history
        users, positions = users[keep], positions[keep]
        targets = seq[users, positions]
        freq = np.bincount(targets, minlength=self.n_items).astype(np.float64) + 1.0
        log_q = torch.as_tensor(np.log(freq / freq.sum()), dtype=torch.float32)

        self.model = TwoTower(self.n_users, self.n_items, self.item_genres, self.item_bucket, cfg)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        seq_t = torch.as_tensor(np.where(seq >= 0, seq, self.n_items))
        offsets = torch.arange(cfg.hist_len)
        n = len(users)
        steps_per_epoch = int(np.ceil(n / cfg.batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1)
        losses = []
        t0 = time.perf_counter()
        self.model.train()
        for _ in range(cfg.epochs):
            perm = rng.permutation(n)
            ep_loss = 0.0
            for b in range(steps_per_epoch):
                idx = perm[b * cfg.batch_size : (b + 1) * cfg.batch_size]
                u = torch.as_tensor(users[idx])
                p = torch.as_tensor(positions[idx])
                tgt = torch.as_tensor(targets[idx])
                hpos = p[:, None] - 1 - offsets[None, :]
                hist = seq_t[u[:, None], hpos.clamp(min=0)]
                hist = torch.where(hpos >= 0, hist, torch.full_like(hist, self.n_items))
                uv = self.model.user_vec(u, hist)
                iv = self.model.item_vec(tgt)
                logits = uv @ iv.T / cfg.temperature
                if cfg.logq_correction:
                    logits = logits - log_q[tgt][None, :]
                same = tgt[:, None] == tgt[None, :]
                eye = torch.eye(len(idx), dtype=torch.bool)
                logits = logits.masked_fill(same & ~eye, float("-inf"))  # duplicates in batch are not negatives
                loss = F.cross_entropy(logits, torch.arange(len(idx)))
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                ep_loss += loss.item() * len(idx)
            losses.append(ep_loss / n)
        self.history = {"epoch_loss": losses, "train_seconds": time.perf_counter() - t0, "n_pairs": int(n)}
        self.model.eval()
        with torch.no_grad():
            self._table = self.model.item_table()
        return self

    @torch.no_grad()
    def item_embeddings(self) -> np.ndarray:
        return self.model.item_vec(torch.arange(self.n_items)).numpy().astype(np.float32)

    def histories(self, fit_df: pd.DataFrame) -> np.ndarray:
        seq, lens = build_sequences(fit_df, self.n_users)
        return recent_history(seq, lens, self.cfg.hist_len, self.n_items)

    @torch.no_grad()
    def user_embeddings(self, hist: np.ndarray, users: np.ndarray | None = None) -> np.ndarray:
        users = np.arange(hist.shape[0]) if users is None else users
        out = [
            self.model.user_vec(torch.as_tensor(users[i : i + 4096]), torch.as_tensor(hist[i : i + 4096]), self._table).numpy()
            for i in range(0, len(users), 4096)
        ]
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def user_embedding_one(self, user: int, hist_row: np.ndarray) -> np.ndarray:
        return self.model.user_vec(torch.as_tensor([user]), torch.as_tensor(hist_row[None, :]), self._table).numpy()
