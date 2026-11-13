"""Prototype memory: distance-weighted voting + class-specific boundary
(the deployed matcher of Sec. IV-C)."""
from __future__ import annotations

import json

import numpy as np
import torch
from torch_geometric.data import Batch


class PrototypeMemory:
    """Loads memory.npz (prototypes, labels, class names, thresholds) and
    classifies / detects with Eq. (voting) and Eq. (decision) of the paper:

    - voting readout: candidate class = argmax of the summed inverse
      distances over the k nearest prototypes;
    - boundary: accept iff the distance to the candidate class's nearest
      prototype does not exceed its class-specific threshold.
    """

    def __init__(self, npz_path: str, k: int = 5, eps: float = 1e-3):
        z = np.load(npz_path, allow_pickle=True)
        self.protos = z["protos"].astype(np.float32)
        self.proto_labels = z["proto_labels"].astype(np.int64)
        self.class_names = [str(c) for c in z["class_names"]]
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        self.thresholds = {int(kk): float(vv)
                           for kk, vv in zip(z["thr_labels"], z["thresholds"])}
        self.k = int(meta.get("k", k))
        self.eps = float(meta.get("eps", eps))
        self.percentile = meta.get("percentile", 98)
        self.R = meta.get("R", 25)

    # -- embeddings --
    @staticmethod
    def encode(model, graphs, device, batch_size=256):
        model.eval()
        embs = []
        with torch.no_grad():
            for s in range(0, len(graphs), batch_size):
                b = Batch.from_data_list(graphs[s : s + batch_size]).to(device)
                embs.append(model(b.x_dict, b.edge_index_dict, None, b).cpu().numpy())
        return np.concatenate(embs) if embs else np.zeros((0, 128), np.float32)

    # -- Eq. (voting): candidate class by inverse-distance weighted votes --
    def vote(self, queries: np.ndarray):
        d = np.linalg.norm(queries[:, None, :] - self.protos[None, :, :], axis=2)
        kk = min(self.k, len(self.protos))
        nn_idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        n_cls = len(self.class_names)
        scores = np.zeros((len(queries), n_cls), np.float64)
        for qi in range(len(queries)):
            w = 1.0 / (d[qi, nn_idx[qi]] + self.eps)
            np.add.at(scores[qi], self.proto_labels[nn_idx[qi]], w)
        return scores.argmax(axis=1), scores

    # -- Eq. (decision): boundary verification on the candidate class --
    def decide(self, queries: np.ndarray):
        cand, _ = self.vote(queries)
        d_all = np.linalg.norm(queries[:, None, :] - self.protos[None, :, :], axis=2)
        n_cls = len(self.class_names)
        d_cand = np.full(len(queries), np.inf, np.float64)
        for c in range(n_cls):
            m = cand == c
            if m.any():
                d_cand[m] = d_all[m][:, self.proto_labels == c].min(axis=1)
        thr = np.array([self.thresholds.get(int(c), np.inf)
                        for c in cand], np.float64)
        accepted = d_cand <= thr
        return cand, accepted, d_cand
