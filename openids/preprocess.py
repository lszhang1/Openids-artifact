"""PCAP / flow-table -> heterogeneous traffic graphs.

Two entry points:
- ``pcap_to_graphs(pcap, feature_columns)``: raw pcap -> graphs (nfstream).
- ``graphs_from_flows_csv(csv, feature_columns)``: a pre-extracted nfstream
  flow table (CSV) -> graphs; rolling features are always added here.

Graph layout (one bidirectional flow, at most 20 packets):
- ``flow`` node: flow-level statistics + rolling temporal context (87 dims);
- ``packet`` nodes: 8 TCP flags + first 1500 raw payload bytes;
- containment edges ``flow->packet`` (direction, ip/transport/payload size);
- sequential edges ``packet_i->packet_{i+1}`` (inter-arrival time);
- made undirected.

``feature_columns`` fixes the flow-node feature order: use
``assets/feature_columns.json`` shipped with this repository so that new
traffic produces the exact layout the encoder was trained on.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

PACKET_BYTE_DIM = 1500
ROLLING_WINDOW = 350
HTTP_PORTS = [443, 8080, 80]
VULN_PORTS = [20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3389, 8080]
DNS_PORTS = [53]

DROP_BASE = {
    "id", "src_ip", "src_mac", "src_oui", "dst_ip", "dst_mac", "dst_oui",
    "vlan_id", "tunnel_id",
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "bidirectional_duration_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
    "bidirectional_packets", "bidirectional_bytes",
    "bidirectional_syn_packets", "bidirectional_cwr_packets",
    "bidirectional_ece_packets", "bidirectional_urg_packets",
    "bidirectional_ack_packets", "bidirectional_psh_packets",
    "bidirectional_rst_packets", "bidirectional_fin_packets",
}


class _PacketCollector:
    """nfstream NFPlugin collecting per-packet features, flow capped at 20 pkts."""

    def __init__(self, limit: int = 20):
        from nfstream import NFPlugin

        class _Plugin(NFPlugin):
            def __init__(pl):
                super().__init__()
                pl.limit = limit

            def on_init(pl, packet, flow):
                if pl.limit == 1:
                    flow.expiration_id = -1
                flow.udps.payload_data = [
                    packet.ip_packet[-packet.payload_size:].hex() if packet.payload_size > 0 else "00"
                ]
                flow.udps.delta_time = [packet.delta_time]
                flow.udps.packet_direction = [packet.direction]
                flow.udps.ip_size = [packet.ip_size]
                flow.udps.transport_size = [packet.transport_size]
                flow.udps.payload_size = [packet.payload_size]
                for f in ("syn", "cwr", "ece", "urg", "ack", "psh", "rst", "fin"):
                    setattr(flow.udps, f, [getattr(packet, f)])

            def on_update(pl, packet, flow):
                flow.udps.payload_data.append(
                    packet.ip_packet[-packet.payload_size:].hex() if packet.payload_size > 0 else "00"
                )
                flow.udps.delta_time.append(packet.delta_time)
                flow.udps.packet_direction.append(packet.direction)
                flow.udps.ip_size.append(packet.ip_size)
                flow.udps.transport_size.append(packet.transport_size)
                flow.udps.payload_size.append(packet.payload_size)
                for f in ("syn", "cwr", "ece", "urg", "ack", "psh", "rst", "fin"):
                    getattr(flow.udps, f).append(getattr(packet, f))
                if pl.limit == flow.bidirectional_packets:
                    flow.expiration_id = -1

        self._plugin_cls = _Plugin

    def plugin(self):
        return self._plugin_cls()


def pcap_to_csv(pcap_path: str, csv_path: str, flow_packets: int = 20):
    from nfstream import NFStreamer

    collector = _PacketCollector(flow_packets)
    nf = NFStreamer(
        source=pcap_path,
        accounting_mode=1,
        idle_timeout=120,
        statistical_analysis=True,
        n_dissections=0,
        udps=collector.plugin(),
    )
    nf.to_csv(path=csv_path, columns_to_anonymize=[], flows_per_file=0)
    return csv_path


def add_rolling_features(df: pandas.DataFrame) -> pandas.DataFrame:
    """Rolling temporal context, computed chronologically over the 350 most
    recent flows per source-destination pair and per destination."""
    from sklearn.preprocessing import LabelEncoder

    df = df.sort_values(by="bidirectional_first_seen_ms").reset_index(drop=True)
    le1, le2 = LabelEncoder(), LabelEncoder()
    df["src_dst_ip"] = df["src_ip"].astype(str) + "-" + df["dst_ip"].astype(str)
    df["src_dst_encoded"] = le1.fit_transform(df["src_dst_ip"])
    df["dst_ip_encoded"] = le2.fit_transform(df["dst_ip"].astype(str))
    W = ROLLING_WINDOW
    hp, vp, dpn = HTTP_PORTS, VULN_PORTS, DNS_PORTS

    rs = lambda g, w: g.rolling(window=w, min_periods=1).sum()
    ru = lambda g, w: g.rolling(window=w, min_periods=1).apply(lambda x: len(set(x)), raw=True)
    rm = lambda g, w: g.rolling(window=w, min_periods=1).mean()

    df["packet_size_variation"] = df[
        ["src2dst_min_ps", "src2dst_max_ps", "dst2src_min_ps", "dst2src_max_ps"]
    ].std(axis=1)
    df["_u"] = (df["protocol"] == 17).astype(int)
    df["_t"] = (df["protocol"] == 6).astype(int)
    df["_i"] = (df["protocol"] == 1).astype(int)
    df["Rolling_UDP_Requests_SourceDestination"] = df.groupby("src_dst_encoded")["_u"].transform(lambda x: rs(x, W))
    df["Rolling_UDP_Requests_Destination"] = df.groupby("dst_ip_encoded")["_u"].transform(lambda x: rs(x, W))
    df["Rolling_TCP_Requests_SourceDestination"] = df.groupby("src_dst_encoded")["_t"].transform(lambda x: rs(x, W))
    df["Rolling_TCP_Requests_Destination"] = df.groupby("dst_ip_encoded")["_t"].transform(lambda x: rs(x, W))
    for fl in ["ack", "fin", "rst", "psh", "syn"]:
        c = f"bidirectional_{fl}_packets"
        df[f"Rolling_{fl.upper()}_Packets_SourceDestination"] = df.groupby("src_dst_encoded")[c].transform(lambda x: rs(x, W))
        df[f"Rolling_{fl.upper()}_Packets_Destination"] = df.groupby("dst_ip_encoded")[c].transform(lambda x: rs(x, W))
    df["Unique_Ports_In_SourceDestinationIP"] = df.groupby("src_dst_encoded")["dst_port"].transform(lambda x: ru(x, W))
    df["Rolling_ICMP_Requests_SourceDestination"] = df.groupby("src_dst_encoded")["_i"].transform(lambda x: rs(x, W))
    df["Rolling_ICMP_Requests_Destination"] = df.groupby("dst_ip_encoded")["_i"].transform(lambda x: rs(x, W))
    df["_h"] = df["dst_port"].isin(hp).astype(int)
    df["Rolling_http_port_SourceDestination"] = df.groupby("src_dst_encoded")["_h"].transform(lambda x: rs(x, W))
    df["Rolling_http_port_Destination"] = df.groupby("dst_ip_encoded")["_h"].transform(lambda x: rs(x, W))
    df["Rolling_Duration_SourceDestination"] = df.groupby("src_dst_encoded")["bidirectional_duration_ms"].transform(lambda x: rm(x, W))
    df["Rolling_Duration_Destination"] = df.groupby("dst_ip_encoded")["bidirectional_duration_ms"].transform(lambda x: rm(x, W))
    df["_dd"] = df["dst_port"].isin(dpn).astype(int)
    df["Rolling_DNS_request_SourceDestination"] = df.groupby("src_dst_encoded")["_dd"].transform(lambda x: rs(x, W))
    df["Rolling_DNS_request_Destination"] = df.groupby("dst_ip_encoded")["_dd"].transform(lambda x: rs(x, W))
    df["_ds"] = df["src_port"].isin(dpn).astype(int)
    df["Rolling_DNS_request_SourceDestination2"] = df.groupby("src_dst_encoded")["_ds"].transform(lambda x: rs(x, W))
    df["Rolling_DNS_request_Destination2"] = df.groupby("dst_ip_encoded")["_ds"].transform(lambda x: rs(x, W))
    df["_v"] = df["dst_port"].isin(vp).astype(int)
    df["Rolling_vulnerable_port"] = df.groupby("src_dst_encoded")["_v"].transform(lambda x: rs(x, W))
    df["Rolling_packets_destination"] = df.groupby("dst_ip_encoded")["src2dst_packets"].transform(lambda x: rs(x, W))
    df["Rolling_bipackets_destination"] = df.groupby("dst_ip_encoded")["bidirectional_packets"].transform(lambda x: rs(x, W))

    df["expiration_id"] = pandas.Categorical(df["expiration_id"], categories=[0, -1])
    df["protocol"] = pandas.Categorical(df["protocol"], categories=[1, 2, 6, 17, 58])
    df = pandas.get_dummies(df, prefix=["Exp", "proto"], columns=["expiration_id", "protocol"], dtype=int)
    drop_cols = [
        "src_dst_ip", "src_dst_encoded", "dst_ip_encoded", "_u", "_t", "_i",
        "_v", "_h", "_dd", "_ds",
    ]
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    return df


def _parse_list(val):
    import ast

    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def payload_stats(payload_hex: list):
    """Shannon entropy and ASCII-byte ratio of the flow payload."""
    try:
        all_bytes = b"".join(
            bytes.fromhex(h) if h and h != "00" else b"" for h in payload_hex
        )
    except Exception:
        return 0.0, 0.0
    if len(all_bytes) == 0:
        return 0.0, 0.0
    counts = np.bincount(np.frombuffer(all_bytes, dtype=np.uint8), minlength=256)
    probs = counts.astype(np.float64) / len(all_bytes)
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    n_ascii = int(
        np.sum(
            (np.frombuffer(all_bytes, dtype=np.uint8) >= 0x20)
            & (np.frombuffer(all_bytes, dtype=np.uint8) <= 0x7E)
        )
    )
    return entropy, n_ascii / len(all_bytes)


def df_to_graphs(df: pandas.DataFrame, ref_columns: list) -> list:
    """Convert an enriched flow table to HeteroData graphs (one per flow)."""
    for c in ref_columns:
        if c not in df.columns:
            df[c] = 0
    graphs = []
    for ri in range(len(df)):
        row = df.iloc[ri]
        payload_hex = _parse_list(row.get("udps.payload_data"))
        npk = len(payload_hex)
        if npk == 0:
            continue
        ff = np.nan_to_num(row[ref_columns].values.astype(np.float32), 0.0)
        ent, asc = payload_stats(payload_hex)
        ff = np.append(ff, [ent, asc]).astype(np.float32)
        pl = []
        for pi in range(npk):
            pf = []
            for fk in ["syn", "cwr", "ece", "urg", "ack", "psh", "rst", "fin"]:
                try:
                    flags = _parse_list(row.get(f"udps.{fk}"))
                    pf.append(float(flags[pi]) if pi < len(flags) else 0.0)
                except Exception:
                    pf.append(0.0)
            hs = payload_hex[pi] if pi < len(payload_hex) else "00"
            try:
                ba = bytes.fromhex(hs) if hs and hs != "00" else b""
            except Exception:
                ba = b""
            bl = list(ba)
            bl = (bl + [0] * (PACKET_BYTE_DIM - len(bl))) if len(bl) < PACKET_BYTE_DIM else bl[:PACKET_BYTE_DIM]
            pf.extend(bl)
            pl.append(pf)
        packet_feats = np.array(pl, dtype=np.float32)

        contain_edge = []
        for pi in range(npk):
            try:
                dr = _parse_list(row.get("udps.packet_direction"))
                ip = _parse_list(row.get("udps.ip_size"))
                tr = _parse_list(row.get("udps.transport_size"))
                pls = _parse_list(row.get("udps.payload_size"))
                contain_edge.append([
                    float(dr[pi]) if pi < len(dr) else 0.0,
                    float(ip[pi]) if pi < len(ip) else 0.0,
                    float(tr[pi]) if pi < len(tr) else 0.0,
                    float(pls[pi]) if pi < len(pls) else 0.0,
                ])
            except Exception:
                contain_edge.append([0.0, 0.0, 0.0, 0.0])
        contain_edge = np.array(contain_edge, dtype=np.float32)

        dt = _parse_list(row.get("udps.delta_time"))
        seq_edge = []
        for pi in range(1, npk):
            try:
                seq_edge.append([float(dt[pi]) if pi < len(dt) else 0.0])
            except Exception:
                seq_edge.append([0.0])
        seq_edge = np.array(seq_edge, dtype=np.float32) if seq_edge else np.zeros((0, 1), dtype=np.float32)

        cei = np.vstack([np.zeros(npk, dtype=np.int64), np.arange(npk, dtype=np.int64)])
        lei = (
            np.vstack([np.arange(npk - 1, dtype=np.int64), np.arange(1, npk, dtype=np.int64)])
            if npk > 1
            else np.zeros((2, 0), dtype=np.int64)
        )

        data = HeteroData()
        data["flow"].x = torch.tensor(ff, dtype=torch.float32).unsqueeze(0)
        data["packet"].x = torch.tensor(packet_feats, dtype=torch.float32)
        data["flow", "to", "packet"].edge_index = torch.tensor(cei, dtype=torch.int64)
        data["packet", "to", "packet"].edge_index = torch.tensor(lei, dtype=torch.int64)
        data["flow", "to", "packet"].edge_attr = torch.tensor(contain_edge, dtype=torch.float32)
        data["packet", "to", "packet"].edge_attr = torch.tensor(seq_edge, dtype=torch.float32)
        data = ToUndirected()(data)
        graphs.append(data)
    return graphs


def load_feature_columns(path: str) -> list:
    """Flow-node feature order the encoder was trained with."""
    with open(path) as f:
        return json.load(f)


def graphs_from_flows_csv(csv_path: str, feature_columns: list) -> list:
    df = pandas.read_csv(csv_path)
    if len(df) == 0:
        return []
    df = add_rolling_features(df)
    return df_to_graphs(df, feature_columns)


def pcap_to_graphs(pcap_path: str, feature_columns: list,
                   tmp_csv: str = None) -> list:
    tmp_csv = tmp_csv or (pcap_path + ".flows.csv")
    pcap_to_csv(pcap_path, tmp_csv)
    return graphs_from_flows_csv(tmp_csv, feature_columns)
