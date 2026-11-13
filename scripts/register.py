"""Register emerging attacks as new classes (Sec. IV-D, few-shot registration).

Takes M flows from each unknown attack directory, encodes ALL traffic
(known + unknown) in a single pass to ensure consistent embeddings,
registers M prototypes per attack into the memory, and evaluates the
remaining flows in the expanded class space.

Reports: known-class F1, registered-class F1, and overall F1.

Input layout:
    <known_dir>/<class>/data_*.pt       34 known classes
    <unknown_dir>/<attack>/data_*.pt    emerging attacks (first M per attack
                                        are used for registration)
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from openids.model import build_encoder  # noqa: E402
from openids.memory import PrototypeMemory  # noqa: E402


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


def loo_threshold(embeddings):
    """Leave-one-out nearest-neighbour distances -> p98 threshold."""
    n = len(embeddings)
    if n < 2:
        return 1.0
    d = np.linalg.norm(embeddings[:, None, :] - embeddings[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(np.percentile(d.min(axis=1), 98))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-dir", required=True)
    ap.add_argument("--unknown-dir", required=True)
    ap.add_argument("--M", type=int, default=10,
                    help="registration samples per attack")
    ap.add_argument("--ckpt", default=os.path.join(
        os.path.dirname(__file__), "..", "assets", "checkpoint.pt"))
    ap.add_argument("--memory", default=os.path.join(
        os.path.dirname(__file__), "..", "assets", "memory.npz"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load all graphs ----
    kg, knames = load_graphs(args.known_dir)
    ug_all, unames_all = load_graphs(args.unknown_dir)
    print(f"[1] known={len(kg)} unknown={len(ug_all)} M={args.M}")

    model = build_encoder(kg[0], hidden=64, out_dim=128, norm="layer")
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model = model.to(device).eval()
    mem = PrototypeMemory(args.memory)
    print(f"    base memory: {len(mem.protos)} prototypes, "
          f"{len(mem.class_names)} classes")

    # ---- encode ALL traffic in a single pass (consistent embeddings) ----
    all_graphs = list(kg) + list(ug_all)
    all_emb = mem.encode(model, all_graphs, device)
    ke = all_emb[:len(kg)]
    ue = all_emb[len(kg):]
    un_arr = np.array(unames_all)

    # ---- register: first M embeddings per attack ----
    attacks = sorted(set(un_arr))
    new_protos, new_labels, new_names = [], [], []
    eval_idx = []
    for ci, atk in enumerate(attacks):
        idx = [i for i in range(len(un_arr)) if un_arr[i] == atk]
        for e in ue[idx[:args.M]]:
            new_protos.append(e)
            new_labels.append(len(mem.class_names) + ci)
        new_names.append(atk)
        eval_idx.extend(idx[args.M:])

    protos45 = np.vstack([mem.protos, np.array(new_protos, np.float32)])
    labels45 = np.concatenate([mem.proto_labels,
                               np.array(new_labels, dtype=np.int64)])
    all_names = list(mem.class_names) + new_names
    print(f"[2] registered {len(new_protos)} flows -> {len(attacks)} new classes, "
          f"total {len(all_names)} classes, {len(protos45)} prototypes")

    # ---- evaluate in expanded class space (same embeddings) ----
    eval_e = ue[eval_idx]
    eval_n = un_arr[eval_idx]
    print(f"[3] evaluating {len(kg)} known + {len(eval_idx)} new-class flows...")

    d = np.linalg.norm(eval_e[:, None, :] - protos45[None, :, :], axis=2)
    kk = min(mem.k, len(protos45))
    nn_idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
    scores = np.zeros((len(eval_e), len(all_names)))
    for qi in range(len(eval_e)):
        w = 1.0 / (d[qi, nn_idx[qi]] + mem.eps)
        np.add.at(scores[qi], labels45[nn_idx[qi]], w)
    pred_idx = scores.argmax(axis=1)
    pred_new = np.array([all_names[i] for i in pred_idx])

    # also evaluate known flows (same embeddings)
    d_k = np.linalg.norm(ke[:, None, :] - protos45[None, :, :], axis=2)
    nn_k = np.argpartition(d_k, kk - 1, axis=1)[:, :kk]
    s_k = np.zeros((len(ke), len(all_names)))
    for qi in range(len(ke)):
        w = 1.0 / (d_k[qi, nn_k[qi]] + mem.eps)
        np.add.at(s_k[qi], labels45[nn_k[qi]], w)
    pred_k = np.array([all_names[i] for i in s_k.argmax(axis=1)])
    kn_arr = np.array(knames)

    # ---- report ----
    known_set = sorted(mem.class_names)
    new_set = sorted(attacks)

    mk = precision_recall_fscore_support(kn_arr, pred_k, labels=known_set,
                                          average="macro", zero_division=0)
    print(f"\n=== Known classes ({len(known_set)}) ===")
    print(f"Macro-F1: {100*mk[2]:.2f}")

    mn = precision_recall_fscore_support(eval_n, pred_new, labels=new_set,
                                          average="macro", zero_division=0)
    print(f"\n=== Registered classes ({len(new_set)}) ===")
    print(f"Macro-F1: {100*mn[2]:.2f}")
    _, _, f1n, supn = precision_recall_fscore_support(
        eval_n, pred_new, labels=new_set, zero_division=0)
    print(f"{'attack':34s} {'F1':>7s} {'n':>5s}")
    for i, lab in enumerate(new_set):
        if supn[i] > 0:
            print(f"  {lab:34s} {100*f1n[i]:6.2f} {supn[i]:5d}")

    mo = precision_recall_fscore_support(
        np.concatenate([kn_arr, eval_n]),
        np.concatenate([pred_k, pred_new]),
        labels=sorted(known_set + new_set), average="macro", zero_division=0)
    print(f"\n=== Overall ({len(known_set + new_set)} classes) ===")
    print(f"Macro-F1: {100*mo[2]:.2f}")
    print(f"Known F1: {100*mk[2]:.2f}  New-class F1: {100*mn[2]:.2f}")


if __name__ == "__main__":
    main()
