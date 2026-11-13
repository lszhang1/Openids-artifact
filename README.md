# Open-IDS Artifact

Artifact for **"OPEN-IDS: Small-Large Model Collaboration for Label-Efficient, Fine-Grained, and Self-Improving IoT Intrusion Detection"**.

This repository releases the **small-model side** of Open-IDS: the label-efficient triplet training of the HGNN encoder, the sample-selection code (class-balanced budget sampling, semi-hard negative mining, R-medoids prototype selection, percentile threshold calibration), and the deployed prototype matcher. The LLM-side attribution pipeline is not part of this release.

## Quick Start

```bash
git clone https://github.com/<your-username>/openids-artifact.git
cd openids-artifact
pip install -r requirements.txt
```

### Effect A: Fine-Grained Classification (34 classes)

```bash
python scripts/classify.py --known-dir assets/data/known
```

Output: per-class P/R/F1 + Accuracy + Macro-F1 (~96.8%).

### Effect B: Known + Unknown Detection

```bash
python scripts/detect.py --known-dir assets/data/known \
                         --unknown-dir assets/data/unknown
```

Output: Known/Unknown P/R/F1 + binary Macro-F1 (~95.7%) + PR-AUC + per-attack recall.

### Effect C: Few-Shot Registration + 45-Class Evaluation

```bash
python scripts/register.py --known-dir assets/data/known \
                           --unknown-dir assets/data/unknown \
                           --M 10
```

Output: known-class F1 (preserved, ~96.8%) + registered-class F1 (~92%) + overall 45-class F1.

## Using Your Own Traffic

```bash
# From a pcap file:
python scripts/prepare_inputs.py --input capture.pcap --class-name myclass --out mydata

# From a pre-extracted flows CSV:
python scripts/prepare_inputs.py --input capture.flows.csv --class-name myclass --out mydata

# Then classify:
python scripts/classify.py --known-dir mydata
```

## Retraining from Scratch

```bash
# 1) Train the encoder (B=200 per class, semi-hard triplet, 80 epochs):
python scripts/train_model.py --data-root <CICIoT2023-root> --budget 200 \
                              --strategy semi-hard --out ckpt.pt

# 2) Build the prototype memory + thresholds:
python scripts/build_memory.py --data-root <CICIoT2023-root> --ckpt ckpt.pt \
                               --out memory.npz
```

`<CICIoT2023-root>` expects `<class>/processed/train/data_*.pt` graph files built with the preprocessing in `openids/preprocess.py`.

## Repository Structure

```
openids/                  Core library
  preprocess.py           pcap / flows-CSV -> heterogeneous traffic graphs
  model.py                HGNN encoder (frozen after training)
  selection.py            Budget sampling, semi-hard mining, R-medoids, p98 thresholds
  train.py                Triplet metric training
  memory.py               Prototype memory: voting + class-specific boundary

scripts/
  train_model.py          Train the encoder under a per-class label budget
  build_memory.py         Checkpoint -> prototypes + thresholds (memory.npz)
  prepare_inputs.py       pcap or flows CSV -> graph .pt files
  classify.py             Effect A: fine-grained classification
  detect.py               Effect B: known + unknown detection
  register.py             Effect C: few-shot registration + evaluation

assets/
  checkpoint.pt           Trained encoder (3.3 MB)
  memory.npz              Prototype memory + thresholds (440 KB)
  feature_columns.json    Flow-node feature order
  data/known/             34 known-class test graphs (curated)
  data/unknown/           11 emerging-attack test graphs (curated)
```

## Paper Correspondence

| Paper component | Code |
|---|---|
| Heterogeneous traffic graph (Sec. IV-A) | `openids/preprocess.py` |
| Semi-hard triplet training, P=8, K=4, m=1.0 (Sec. IV-B) | `openids/train.py`, `openids/selection.py` |
| R-medoids prototypes, R=min(25,N) (Sec. IV-B) | `selection.select_prototypes` |
| p98 thresholds on 10% calibration share (Sec. IV-C) | `selection.calibrate_thresholds` |
| Distance-weighted voting, k=5 (Eq. 12) | `memory.PrototypeMemory.vote` |
| Class-specific boundary decision (Eq. 14) | `memory.PrototypeMemory.decide` |
| LOO threshold for new classes (Eq. 24) | `register.loo_threshold` |

## Hyperparameters

| Parameter | Value |
|---|---|
| Hidden dim | 64 |
| Embedding dim | 128 |
| Margin | 1.0 |
| Classes per batch (P) | 8 |
| Graphs per class (K) | 4 |
| Learning rate | 5e-4 |
| Epochs | 80 |
| Prototypes per class (R) | 25 (capped by class size) |
| Voting neighbors (k) | 5 |
| Threshold percentile | 98 |
| Registration samples (M) | 10 |
| ε (voting) | 1e-3 |
| Normalization | LayerNorm |
| Distance | Euclidean (L2-normalized embeddings) |

## License

MIT
