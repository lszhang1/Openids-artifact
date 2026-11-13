"""Effect A: known-class fine-grained classification (Sec. V, RQ1).

Loads the shipped checkpoint + prototype memory, embeds the prepared known
test graphs, and reports per-class P/R/F1, Accuracy, and Macro averages with
the deployed distance-weighted voting readout (Eq. voting).

Input layout: <known_dir>/<class>/data_*.pt
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from openids.memory import PrototypeMemory  # noqa: E402
from openids.model import build_encoder  # noqa: E402


def load_graphs(root):
    graphs, names = [], []
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".pt"):
                g = torch.load(os.path.join(d, f), weights_only=False,
                               map_location="cpu")
                if g is not None:
                    graphs.append(g)
                    names.append(cls)
    return graphs, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-dir", required=True,
                    help="directory of per-class graph folders (test flows)")
    ap.add_argument("--ckpt", default=os.path.join(
        os.path.dirname(__file__), "..", "assets", "checkpoint.pt"))
    ap.add_argument("--memory", default=os.path.join(
        os.path.dirname(__file__), "..", "assets", "memory.npz"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    graphs, names = load_graphs(args.known_dir)
    print(f"[1] loaded {len(graphs)} known test flows "
          f"({len(set(names))} classes)")
    model = build_encoder(graphs[0], hidden=64, out_dim=128, norm="layer")
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model = model.to(device).eval()
    mem = PrototypeMemory(args.memory)

    print("[2] embedding + voting readout ...", flush=True)
    emb = mem.encode(model, graphs, device)
    pred_idx, _ = mem.vote(emb)
    pred = [mem.class_names[i] for i in pred_idx]

    labels = sorted(set(names) | set(pred))
    p, r, f1, sup = precision_recall_fscore_support(
        names, pred, labels=labels, zero_division=0)
    acc = float(np.mean([a == b for a, b in zip(names, pred)]))

    print(f"\n{'class':34s} {'P':>7s} {'R':>7s} {'F1':>7s} {'n':>6s}")
    print("-" * 66)
    by_class = defaultdict(list)
    for i, lab in enumerate(labels):
        by_class["all"].append((lab, p[i], r[i], f1[i], sup[i]))
    rows = [(lab, p[i], r[i], f1[i], sup[i]) for i, lab in enumerate(labels)]
    for lab, pp, rr, ff, ss in rows:
        print(f"{lab:34s} {100*pp:7.2f} {100*rr:7.2f} {100*ff:7.2f} {ss:6d}")
    macro = precision_recall_fscore_support(names, pred, labels=labels,
                                            average="macro", zero_division=0)
    print("-" * 66)
    print(f"{'Macro':34s} {100*macro[0]:7.2f} {100*macro[1]:7.2f} "
          f"{100*macro[2]:7.2f}")
    print(f"Accuracy: {100*acc:.2f}%   (flows: {len(names)})")


if __name__ == "__main__":
    main()
