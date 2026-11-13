"""Sample selection: budget sampling, semi-hard triplet mining, and
prototype selection.

- ``BalancedBatchSampler``: P classes x K graphs mini-batches (Eq. of Sec.
  IV-B, metric sample construction).
- ``semi_hard_triplet_loss`` / ``random_triplet_loss``: the training
  objective and its random-mining ablation. With no semi-hard negative in
  the batch, the hardest negative (closest sample of another class) is used.
- ``kmedoids`` + ``select_prototypes``: R-medoids prototype selection, with
  R automatically capped by the class size for tail classes (R=min(25, N)).
- ``calibrate_thresholds``: class-specific acceptance thresholds, the 98th
  percentile of own-class nearest-prototype distances on held-out
  calibration flows (Eq. of Sec. IV-C).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class BalancedBatchSampler:
    """Yield batch_size indices covering classes_per_batch distinct classes."""

    def __init__(self, labels, batch_size: int, classes_per_batch: int, seed: int = 42):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.P = classes_per_batch
        self.K = max(1, batch_size // classes_per_batch)
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        order = self.rng.permutation(len(self.labels))
        by_class = {c: order[self.labels[order] == c] for c in np.unique(self.labels)}
        cursors = {c: 0 for c in by_class}
        classes = list(by_class)
        n_batches = len(self.labels) // self.batch_size
        for _ in range(n_batches):
            chosen = self.rng.choice(classes, self.P, replace=False)
            idx = []
            for c in chosen:
                pool = by_class[c]
                take = min(self.K, len(pool) - cursors[c])
                if take <= 0:
                    cursors[c] = 0
                    take = min(self.K, len(pool))
                idx.extend(pool[cursors[c]:cursors[c] + take])
                cursors[c] += take
            yield list(idx[: self.batch_size])

    def __len__(self):
        return len(self.labels) // self.batch_size


def semi_hard_triplet_loss(emb, labels, margin):
    """Vectorized batch triplet loss with semi-hard negative selection."""
    n = emb.size(0)
    labels_t = torch.as_tensor(labels, device=emb.device)
    same = labels_t.unsqueeze(0) == labels_t.unsqueeze(1)
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    pos_mask = same & ~eye
    neg_mask = ~same

    dist = torch.cdist(emb, emb)  # pairwise Euclidean

    total = emb.new_zeros(())
    n_valid = 0
    for i in range(n):
        pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0:
            continue
        p = pos_idx[torch.randint(pos_idx.numel(), (1,), device=emb.device)]
        d_ap = dist[i, p]
        neg_idx = neg_mask[i].nonzero(as_tuple=True)[0]
        if neg_idx.numel() == 0:
            continue
        d_an_all = dist[i, neg_idx]
        semi = (d_an_all > d_ap) & (d_an_all < d_ap + margin)
        if semi.any():
            d_an = d_an_all[semi].min()
        else:
            d_an = d_an_all.min()  # hardest negative fallback
        total = total + F.relu(d_ap - d_an + margin)
        n_valid += 1
    if n_valid == 0:
        return emb.new_zeros(())
    return total / n_valid


def random_triplet_loss(emb, labels, margin):
    n = emb.size(0)
    labels_t = torch.as_tensor(labels, device=emb.device)
    same = labels_t.unsqueeze(0) == labels_t.unsqueeze(1)
    eye = torch.eye(n, dtype=torch.bool, device=emb.device)
    pos_mask = same & ~eye
    neg_mask = ~same
    dist = torch.cdist(emb, emb)
    total = emb.new_zeros(())
    n_valid = 0
    for i in range(n):
        pos_idx = pos_mask[i].nonzero(as_tuple=True)[0]
        neg_idx = neg_mask[i].nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        p = pos_idx[torch.randint(pos_idx.numel(), (1,), device=emb.device)]
        ng = neg_idx[torch.randint(neg_idx.numel(), (1,), device=emb.device)]
        total = total + F.relu(dist[i, p] - dist[i, ng] + margin)
        n_valid += 1
    if n_valid == 0:
        return emb.new_zeros(())
    return total / n_valid


def kmedoids(X: np.ndarray, n_clusters: int, iters: int = 20, seed: int = 0):
    """Small deterministic k-medoids (R-medoids per class).

    Classes with fewer samples than R keep every sample as a prototype,
    i.e., R = min(R, N_class)."""
    rng = np.random.RandomState(seed)
    n = len(X)
    if n <= n_clusters:
        return np.arange(n)
    medoid_idx = rng.choice(n, n_clusters, replace=False)
    for _ in range(iters):
        d = np.linalg.norm(X[:, None, :] - X[medoid_idx][None, :, :], axis=2)
        assign = d.argmin(axis=1)
        new_idx = medoid_idx.copy()
        for c in range(n_clusters):
            members = np.where(assign == c)[0]
            if len(members) == 0:
                continue
            sub = np.linalg.norm(X[members][:, None, :] - X[members][None, :, :], axis=2)
            new_idx[c] = members[sub.sum(axis=1).argmin()]
        if np.array_equal(new_idx, medoid_idx):
            break
        medoid_idx = new_idx
    return medoid_idx


def select_prototypes(embeddings: np.ndarray, labels: np.ndarray,
                      class_names, R: int = 25, seed: int = 0):
    """R-medoids prototypes per known class -> (protos, proto_labels)."""
    protos, proto_labels = [], []
    for c in np.unique(labels):
        emb_c = embeddings[labels == c]
        idx = kmedoids(emb_c, R, seed=seed + int(c))
        for p in emb_c[idx]:
            protos.append(p)
            proto_labels.append(int(c))
    return np.stack(protos).astype(np.float32), np.array(proto_labels)


def calibrate_thresholds(embeddings: np.ndarray, labels: np.ndarray,
                         protos: np.ndarray, proto_labels: np.ndarray,
                         percentile: float = 98):
    """Class-specific acceptance thresholds: the P-th percentile of own-class
    nearest-prototype distances on calibration flows."""
    d = np.linalg.norm(embeddings[:, None, :] - protos[None, :, :], axis=2)
    thresholds = {}
    for c in np.unique(labels):
        own = d[:, proto_labels == c]
        d_c = own.min(axis=1)[labels == c]
        thresholds[int(c)] = float(np.percentile(d_c, percentile))
    return thresholds
