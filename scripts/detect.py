"""Effect B: known + unknown detection (Sec. V, RQ2).

Loads the same single model (checkpoint + prototype memory), embeds the
known test flows and the unknown flows together, and applies the deployed
matcher: distance-weighted voting picks the candidate class, and the
class-specific boundary decides known vs. unknown (Eq. decision).

Reports: known/unknown precision, recall, F1, binary Macro-F1, PR-AUC
(score = candidate-class nearest-prototype distance), per-known-class F1
(closed set), and per-unknown-attack recall.

Input layout:
    <known-dir>/<class>/data_*.pt
    <unknown-dir>/<attack>/data_*.pt   (omit --unknown-dir for known-only)
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from openids.memory import PrototypeMemory  # noqa: E402
from openids.model import build_encoder  # noqa: E402
from scripts.classify import load_graphs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-dir", required=True)
    ap.add_argument("--unknown-dir", default=None)
    ap.add_argument("--ckpt", default=os.path.join(
        os.path.dirname(__file__), "..", "assets", "checkpoint.pt"))
    ap.add_argument("--memory", default=os.path.join(
        os.path.dirname(__file__), "..", "assets", "memory.npz"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kg, knames = load_graphs(args.known_dir)
    ug, unames = ([], [])
    if args.unknown_dir and os.path.isdir(args.unknown_dir):
        ug, unames = load_graphs(args.unknown_dir)
    print(f"[1] known={len(kg)} flows / unknown={len(ug)} flows")

    model = build_encoder(kg[0], hidden=64, out_dim=128, norm="layer")
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model = model.to(device).eval()
    mem = PrototypeMemory(args.memory)

    print("[2] embedding + boundary decision ...", flush=True)
    ke = mem.encode(model, kg, device)
    kc, kacc, kd = mem.decide(ke)
    if ug:
        ue = mem.encode(model, ug, device)
        uc, uacc, ud = mem.decide(ue)
        scores = np.concatenate([kd, ud])          # higher = more unknown
        y_bin = np.concatenate([np.zeros(len(kd)), np.ones(len(ud))])
        prauc = average_precision_score(y_bin, scores)
        pred_bin = np.concatenate([~kacc, ~uacc])   # True = predicted unknown
    else:
        pred_bin = ~kacc

    # ---- binary known/unknown metrics ----
    y = np.concatenate([np.zeros(len(kd)), np.ones(len(ud))]) if ug else np.zeros(len(kd))
    pr = precision_recall_fscore_support(y, pred_bin.astype(int),
                                         labels=[0, 1], zero_division=0)
    names_bin = ["Known", "Unknown"]
    print(f"\n{'class':12s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    print("-" * 38)
    for i, nb in enumerate(names_bin):
        print(f"{nb:12s} {100*pr[0][i]:7.2f} {100*pr[1][i]:7.2f} "
              f"{100*pr[2][i]:7.2f}")
    macro_f1 = 100 * float(np.mean(pr[2]))
    print("-" * 38)
    line = f"{'Macro':12s} {'':7s} {'':7s} {macro_f1:7.2f}"
    if ug:
        line += f"   PR-AUC: {prauc:.4f}"
    print(line)

    # ---- per-known-class closed-set F1 ----
    kpred = [mem.class_names[i] for i in kc]
    labels = sorted(set(knames) | set(kpred))
    p, r, f1, sup = precision_recall_fscore_support(
        knames, kpred, labels=labels, zero_division=0)
    print(f"\nper-known-class (closed set, {len(knames)} flows):")
    print(f"{'class':34s} {'F1':>7s} {'n':>6s}")
    for i, lab in enumerate(labels):
        print(f"{lab:34s} {100*f1[i]:7.2f} {sup[i]:6d}")

    # ---- per-unknown-attack recall ----
    if ug:
        print(f"\nper-unknown-attack recall (detected-as-unknown rate):")
        print(f"{'attack':34s} {'R':>7s} {'n':>6s}")
        recs = []
        for att in sorted(set(unames)):
            idx = [i for i, u in enumerate(unames) if u == att]
            rr = float(np.mean([not uacc[i] for i in idx]))  # rejected = detected as unknown
            recs.append(rr)
            print(f"{att:34s} {100*rr:7.2f} {len(idx):6d}")
        print("-" * 49)
        print(f"{'Mean':34s} {100*float(np.mean(recs)):7.2f}")


if __name__ == "__main__":
    main()
