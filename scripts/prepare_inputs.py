"""Prepare detection inputs: pcap (or a pre-extracted nfstream flow CSV)
-> per-class directories of graph .pt files, ready for classify/detect.

Layout produced:
    <out>/<class_name>/data_000001.pt ...

Use assets/feature_columns.json shipped with this repository so the
flow-node feature order matches the trained encoder.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from openids.preprocess import (  # noqa: E402
    graphs_from_flows_csv, load_feature_columns, pcap_to_graphs,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="pcap file, or a flows CSV extracted with the "
                         "shipped nfstream settings")
    ap.add_argument("--feature-columns",
                    default=os.path.join(os.path.dirname(__file__), "..",
                                         "assets", "feature_columns.json"))
    ap.add_argument("--class-name", required=True,
                    help="label of the flows in this input (directory name)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cols = load_feature_columns(args.feature_columns)
    if args.input.endswith(".pcap"):
        graphs = pcap_to_graphs(args.input, cols)
    else:
        graphs = graphs_from_flows_csv(args.input, cols)

    d = os.path.join(args.out, args.class_name)
    os.makedirs(d, exist_ok=True)
    for i, g in enumerate(graphs):
        torch.save(g, os.path.join(d, f"data_{i+1:06d}.pt"))
    print(f"saved {len(graphs)} graphs -> {d}")


if __name__ == "__main__":
    main()
