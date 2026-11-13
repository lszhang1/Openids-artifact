"""Build the prototype memory from a trained checkpoint (Sec. IV-B/C).

Loads the training split under the same per-class budget used for training,
embeds it with the frozen encoder, selects R-medoids prototypes per class
(R = min(25, N_class)), and calibrates the class-specific thresholds
(98th percentile of own-class nearest-prototype distances) on the 10%
calibration share.

Output: memory.npz with protos / proto_labels / class_names / thresholds.
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from openids.model import build_encoder  # noqa: E402
from openids.selection import calibrate_thresholds, select_prototypes  # noqa: E402
from openids.train import compute_embeddings  # noqa: E402
from scripts.train_model import CIC_IOT_2023_CLASSES, load_train, split_calib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--R", type=int, default=25)
    ap.add_argument("--percentile", type=float, default=98)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    graphs, labels = load_train(args.data_root, args.budget, args.seed)
    train_g, train_y, calib_g, calib_y = split_calib(graphs, labels,
                                                     seed=args.seed)
    print(f"train={len(train_g)} calib={len(calib_g)} classes={len(CIC_IOT_2023_CLASSES)}")

    model = build_encoder(train_g[0], hidden=64, out_dim=128, norm="layer")
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model = model.to(device).eval()

    print("[1] embedding train + calib ...", flush=True)
    emb_tr, y_tr = compute_embeddings(model, train_g, device)
    emb_ca, y_ca = compute_embeddings(model, calib_g, device)

    print("[2] R-medoids prototypes ...", flush=True)
    protos, proto_labels = select_prototypes(emb_tr, y_tr,
                                             CIC_IOT_2023_CLASSES,
                                             R=args.R, seed=args.seed)

    print("[3] calibrating thresholds (p%g) ..." % args.percentile, flush=True)
    thr = calibrate_thresholds(emb_ca, y_ca, protos, proto_labels,
                               percentile=args.percentile)

    meta = {"k": args.k, "R": args.R, "percentile": args.percentile,
            "eps": 1e-3, "budget": args.budget, "seed": args.seed}
    np.savez(
        args.out,
        protos=protos, proto_labels=proto_labels,
        class_names=np.array(CIC_IOT_2023_CLASSES),
        thresholds=np.array([thr[c] for c in sorted(thr)]),
        thr_labels=np.array(sorted(thr)),
        meta=np.array(json.dumps(meta)),
    )
    print(f"saved {args.out}: {len(protos)} prototypes, "
          f"{len(thr)} thresholds")


if __name__ == "__main__":
    main()
