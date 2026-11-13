"""Train the small-model encoder (Sec. IV-B) with a per-class label budget.

Expected data layout (produced by the dataset construction of the paper;
only the training split is used here):
    <data_root>/<class>/processed/train/data_*.pt

The 10% calibration share is held out from the budget and used only for
threshold calibration in build_memory.py.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from openids.train import compute_embeddings, train_encoder  # noqa: E402

CIC_IOT_2023_CLASSES = [
    "Backdoor_Malware", "BrowserHijacking", "CommandInjection",
    "DDoS-ACK_Fragmenttation", "DDoS-HTTP_Flood", "DDoS-ICMP_Flood",
    "DDoS-ICMP_Fragmentation", "DDoS-PSHACK_Flood", "DDoS-RSTFINFlood",
    "DDoS-SYN_Flood", "DDoS-SlowLoris", "DDoS-SynonymousIP_Flood",
    "DDoS-TCP_Flood", "DDoS-UDP_Flood", "DDoS-UDP_Fragmentation",
    "DNS_Spoofing", "DictionaryBruteForce", "DoS-HTTP_Flood",
    "DoS-SYN_Flood", "DoS-TCP-Flood", "DoS-UDP_Flood", "MITM-ArpSpoofing",
    "Mirai-greeth_flood", "Mirai-greip_flood", "Mirai-udpplain",
    "Recon-HostDiscovery", "Recon-OSScan", "Recon-PingSweep",
    "Recon-PortScan", "SqlInjection", "Uploading_Attack",
    "VulnerabilityScan", "XSS", "benign",
]


def load_train(data_root, budget, seed):
    rng = random.Random(seed)
    graphs, labels = [], []
    for ci, cls in enumerate(CIC_IOT_2023_CLASSES):
        p = os.path.join(data_root, cls, "processed", "train")
        if not os.path.isdir(p):
            continue
        files = sorted(f for f in os.listdir(p) if f.endswith(".pt"))
        rng.shuffle(files)
        for fname in (files[:budget] if budget else files):
            g = torch.load(os.path.join(p, fname), weights_only=False,
                           map_location="cpu")
            if g is None:
                continue
            g.y = torch.tensor([ci], dtype=torch.long)
            graphs.append(g)
            labels.append(ci)
    return graphs, np.array(labels)


def split_calib(graphs, labels, frac=0.1, seed=42):
    """Hold out `frac` of each class only for threshold calibration."""
    rng = np.random.RandomState(seed)
    keep, calib = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_cal = max(int(len(idx) * frac), 1)
        calib.extend(idx[:n_cal])
        keep.extend(idx[n_cal:])
    return ([graphs[i] for i in keep], labels[keep],
            [graphs[i] for i in calib], labels[calib])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--budget", type=int, default=200,
                    help="labeled graphs per class (train+calibration)")
    ap.add_argument("--strategy", default="semi-hard",
                    choices=["semi-hard", "random"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--classes-per-batch", type=int, default=8)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="output checkpoint (.pt)")
    args = ap.parse_args()

    print(f"[1] loading train split (budget={args.budget}) ...", flush=True)
    graphs, labels = load_train(args.data_root, args.budget, args.seed)
    train_g, train_y, calib_g, calib_y = split_calib(graphs, labels,
                                                     seed=args.seed)
    print(f"    train={len(train_g)} calib={len(calib_g)}", flush=True)
    print(f"[2] training ({args.strategy}) ...", flush=True)
    model, best = train_encoder(
        train_g, train_y, strategy=args.strategy, epochs=args.epochs,
        batch_size=args.batch_size, classes_per_batch=args.classes_per_batch,
        margin=args.margin, lr=args.lr, seed=args.seed,
        ckpt_path=args.out, norm="layer",
    )
    print(f"done, best_loss={best:.6f}, ckpt={args.out}")


if __name__ == "__main__":
    main()
