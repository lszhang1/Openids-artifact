"""Heterogeneous graph neural network traffic encoder.

The encoder mirrors the XG-NID heterogeneous traffic representation: each
bidirectional flow (at most 20 packets) becomes a graph with one `flow` node
(flow-level statistics + rolling context features), `packet` nodes carrying
byte-wise payload, containment edges (`flow`->`packet`, with direction and
protocol-layer sizes) and sequential edges (`packet`->`packet`, with time
intervals).  Two SAGEConv hetero layers build node states; flow state is
mean-pooled, packet states are read with a BiLSTM over the packet sequence,
both are concatenated and projected to the embedding space, then L2
normalized so Euclidean distance is cosine-like bounded.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool


class GNNEncoder(nn.Module):
    def __init__(self, metadata, hidden: int = 64, out_dim: int = 128,
                 norm: str = "batch"):
        super().__init__()
        self.conv1 = HeteroConv(
            {edge_type: SAGEConv((-1, -1), hidden) for edge_type in metadata[1]}
        )
        self.conv2 = HeteroConv(
            {edge_type: SAGEConv((-1, -1), hidden) for edge_type in metadata[1]}
        )
        # LayerNorm keeps the embedding geometry identical between train and
        # eval (BatchNorm running stats cause a train/test shift that hurts
        # distance-based unknown detection)
        norm_cls = nn.LayerNorm if norm == "layer" else nn.BatchNorm1d
        self.bn1 = nn.ModuleDict({nt: norm_cls(hidden) for nt in metadata[0]})
        self.bn2 = nn.ModuleDict({nt: norm_cls(hidden) for nt in metadata[0]})
        self.bilstm = nn.LSTM(hidden, hidden // 2, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, out_dim)
        self.hidden = hidden

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch):
        x = self.conv1(x_dict, edge_index_dict)
        x = {k: F.leaky_relu(self.bn1[k](v)) for k, v in x.items()}
        x = self.conv2(x_dict, edge_index_dict)
        x = {k: F.leaky_relu(self.bn2[k](v)) for k, v in x.items()}
        flow_emb = global_mean_pool(x["flow"], batch.batch_dict["flow"])

        px, pb = x["packet"], batch.batch_dict["packet"]
        lengths = pb.bincount()
        max_pk = int(lengths.max().item())
        ng = flow_emb.size(0)
        # graphs are contiguous in the batched packet list -> build the padded
        # (ng, max_pk, hidden) sequence with a vectorized gather instead of a
        # python loop
        offsets = torch.cumsum(lengths, 0) - lengths
        pos = offsets[:, None] + torch.arange(max_pk, device=px.device)[None, :]
        pm = pos < (offsets + lengths)[:, None]
        safe_pos = pos.clamp(max=px.size(0) - 1)
        pseq = px[safe_pos] * pm.unsqueeze(-1).to(px.dtype)
        lstm_out, _ = self.bilstm(pseq)
        valid_len = pm.sum(dim=1).clamp(min=1).float().unsqueeze(1)
        pkt_emb = (lstm_out * pm.unsqueeze(2).float()).sum(dim=1) / valid_len
        emb = self.proj(torch.cat([flow_emb, pkt_emb], dim=1))
        return F.normalize(emb, p=2, dim=1)


def build_encoder(sample_graph, hidden: int = 64, out_dim: int = 128,
                  norm: str = "batch") -> GNNEncoder:
    metadata = (sample_graph.node_types, sample_graph.edge_types)
    return GNNEncoder(metadata, hidden=hidden, out_dim=out_dim, norm=norm)
