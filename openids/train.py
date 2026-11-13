"""Triplet metric training for the small-model encoder (Sec. IV-B)."""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch_geometric.data import Batch

from .model import build_encoder
from .selection import BalancedBatchSampler, random_triplet_loss, semi_hard_triplet_loss


def _batch_forward(model, graphs, device):
    batched = Batch.from_data_list(graphs).to(device)
    return model(batched.x_dict, batched.edge_index_dict, None, batched)


@torch.no_grad()
def compute_embeddings(model, graphs, device, batch_size=256):
    model.eval()
    embs, ys = [], []
    for start in range(0, len(graphs), batch_size):
        chunk = graphs[start : start + batch_size]
        emb = _batch_forward(model, chunk, device)
        embs.append(emb.cpu().numpy())
        ys.extend([int(g.y.item()) if hasattr(g, "y") else -1 for g in chunk])
    return np.concatenate(embs), np.array(ys)


def train_encoder(
    train_graphs,
    train_labels,
    strategy: str = "semi-hard",
    hidden: int = 64,
    emb_dim: int = 128,
    margin: float = 1.0,
    epochs: int = 80,
    batch_size: int = 32,
    classes_per_batch: int = 8,
    lr: float = 5e-4,
    device: str = None,
    seed: int = 42,
    ckpt_path: str = None,
    log_every: int = 5,
    norm: str = "layer",
):
    """Train the HGNN encoder with semi-hard triplet mining (default) or
    uniformly random mining (ablation)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_encoder(train_graphs[0], hidden=hidden, out_dim=emb_dim,
                          norm=norm).to(device)

    with torch.no_grad():  # lazy-param init needs >= 2 graphs
        _batch_forward(model, train_graphs[:2], device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sampler = BalancedBatchSampler(train_labels, batch_size, classes_per_batch,
                                   seed=seed)
    best_loss = float("inf")
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss, n_b = 0.0, 0
        for idxs in sampler:
            graphs = [train_graphs[i] for i in idxs]
            labels_b = [int(train_labels[i]) for i in idxs]
            emb = _batch_forward(model, graphs, device)
            if strategy == "semi-hard":
                loss = semi_hard_triplet_loss(emb, labels_b, margin)
            else:
                loss = random_triplet_loss(emb, labels_b, margin)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        avg = epoch_loss / max(n_b, 1)
        if avg < best_loss:
            best_loss = avg
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{epochs} loss={avg:.6f} ({time.time()-t0:.0f}s)",
                  flush=True)
    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, best_loss
