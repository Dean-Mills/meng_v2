"""
TriGAT --- Triplet-aware Graph Attention Network.

Reformulates pose grouping over skeleton *triplets* (3-joint chains A-B-C
where A-B and B-C are skeletal edges) rather than individual joints. Each
triplet encodes an articulation pattern: the angle at the pivot B, the
bone lengths BA and BC, and the triplet's type (which of the 19 canonical
COCO triplets it is).

Motivation: joint embeddings capture position and type but not articulation
structure. Two people standing close together have joints in similar positions
but their articulation patterns (shoulder-elbow-wrist angles, hip-knee-ankle
stances) are distinct. Embedding triplets directly exposes this signal.

The 19 COCO triplets are enumerated from the skeleton adjacency at module
load. Each person instance in a scene contributes up to 19 triplet instances
(fewer if some joints are invisible).

Architecture:
  1. For each joint scene, build triplet-graph: nodes = triplet instances,
     node features = (pivot pos, wing offsets, angle, bone lengths, type emb).
  2. kNN edges over pivot positions.
  3. SA-GAT layers (reused from sa_gat.py) with simplified edge features
     (no skeleton-relative encoding — the triplet TYPE already encodes that).
  4. Output: per-triplet embedding.
  5. Grouping: cluster triplet embeddings, then vote joints to clusters.

Person label for a triplet = pivot joint's person ID.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from config import TriGATConfig


# COCO skeleton adjacency (mirror of sa_gat.COCO_SKELETON)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
NUM_JOINT_TYPES = 17


def _build_adjacency() -> List[set]:
    adj = [set() for _ in range(NUM_JOINT_TYPES)]
    for a, b in COCO_SKELETON:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _enumerate_triplets() -> List[tuple]:
    """
    Return all (A, B, C) triplets where A-B and B-C are skeleton edges.
    Ordered: for each pivot B, A < C to avoid duplicates.
    """
    adj = _build_adjacency()
    triplets = []
    for B in range(NUM_JOINT_TYPES):
        neighbours = sorted(adj[B])
        for i, A in enumerate(neighbours):
            for C in neighbours[i + 1:]:
                triplets.append((A, B, C))
    return triplets


# Fixed ordered list of the 19 canonical COCO triplets.
COCO_TRIPLETS = _enumerate_triplets()
NUM_TRIPLET_TYPES = len(COCO_TRIPLETS)  # 19

# Lookup: joint index -> list of triplet type IDs containing that joint
_JOINT_TO_TRIPLETS = [[] for _ in range(NUM_JOINT_TYPES)]
for tid, (a, b, c) in enumerate(COCO_TRIPLETS):
    _JOINT_TO_TRIPLETS[a].append(tid)
    _JOINT_TO_TRIPLETS[b].append(tid)
    _JOINT_TO_TRIPLETS[c].append(tid)


def build_triplet_graph(
    joint_graph: Data,
    k_neighbors: int = 8,
    image_size: int = 512,
) -> Optional[Data]:
    """
    Build a triplet-level PyG graph from a joint-level PyG graph.

    Args:
        joint_graph: PyG Data with x [N, D], joint_types [N], person_labels [N]
                     x expected in normalised [0,1] for (x, y). D can be 3 or 4.
        k_neighbors: kNN edges over triplet pivot positions
        image_size: for bone length normalisation reference

    Returns:
        PyG Data with:
            x:             [T, F] triplet features
                           [pivot_x, pivot_y, Δ_AB_x, Δ_AB_y, Δ_CB_x, Δ_CB_y,
                            cos_θ, sin_θ, len_AB, len_CB]  (10 values)
            triplet_types: [T] long, index in COCO_TRIPLETS
            person_labels: [T] long, pivot's person
            edge_index:    [2, T*k] kNN edges over pivot positions
            joint_pos_in_triplet: [T, 3] long, the node indices of (A, B, C)
                                  in the original joint graph (for voting)
        Or None if fewer than 2 triplets can be built.
    """
    # Positions from joint graph (x and y are the first two columns after
    # the preprocessor's normalisation).
    joint_pos = joint_graph.x[:, :2]  # [N, 2] in [0,1]
    joint_types = joint_graph.joint_types  # [N]
    person_labels = joint_graph.person_labels
    n = joint_pos.size(0)
    device = joint_pos.device

    # Build a lookup from (person_id, joint_type) -> node index in joint graph
    lookup = {}
    for node_idx in range(n):
        key = (int(person_labels[node_idx]), int(joint_types[node_idx]))
        lookup[key] = node_idx

    # Enumerate triplets per person
    unique_persons = torch.unique(person_labels).tolist()

    pivot_pos_list = []
    feat_list = []
    triplet_type_list = []
    person_list = []
    node_idx_list = []  # (A_node, B_node, C_node) for voting

    for pid in unique_persons:
        for tid, (A_type, B_type, C_type) in enumerate(COCO_TRIPLETS):
            A_node = lookup.get((pid, A_type))
            B_node = lookup.get((pid, B_type))
            C_node = lookup.get((pid, C_type))
            if A_node is None or B_node is None or C_node is None:
                continue  # missing joint, skip this triplet

            posA = joint_pos[A_node]  # [2]
            posB = joint_pos[B_node]
            posC = joint_pos[C_node]

            d_AB = posA - posB
            d_CB = posC - posB
            len_AB = d_AB.norm(p=2)
            len_CB = d_CB.norm(p=2)

            # Angle at B via dot product
            denom = (len_AB * len_CB).clamp(min=1e-6)
            cos_t = (d_AB * d_CB).sum() / denom
            cos_t = cos_t.clamp(-1.0, 1.0)
            sin_t = (1 - cos_t ** 2).clamp(min=0.0).sqrt()

            feat = torch.stack([
                posB[0], posB[1],
                d_AB[0], d_AB[1],
                d_CB[0], d_CB[1],
                cos_t, sin_t,
                len_AB, len_CB,
            ])

            pivot_pos_list.append(posB)
            feat_list.append(feat)
            triplet_type_list.append(tid)
            person_list.append(pid)
            node_idx_list.append([A_node, B_node, C_node])

    if len(feat_list) < 2:
        return None

    x = torch.stack(feat_list)  # [T, 10]
    pivot_pos = torch.stack(pivot_pos_list)  # [T, 2]
    triplet_types = torch.tensor(triplet_type_list, dtype=torch.long, device=device)
    person_labels_t = torch.tensor(person_list, dtype=torch.long, device=device)
    joint_pos_in_triplet = torch.tensor(node_idx_list, dtype=torch.long, device=device)

    # kNN edges over pivot positions
    T = x.size(0)
    k = min(k_neighbors, T - 1)
    dist = torch.cdist(pivot_pos, pivot_pos, p=2)
    dist.fill_diagonal_(float("inf"))
    _, indices = dist.topk(k, dim=1, largest=False)
    source = torch.arange(T, device=device).repeat_interleave(k)
    target = indices.flatten()
    edge_index = torch.stack([source, target], dim=0)

    return Data(
        x=x,
        edge_index=edge_index,
        triplet_types=triplet_types,
        person_labels=person_labels_t,
        joint_pos_in_triplet=joint_pos_in_triplet,
        num_people=joint_graph.num_people,
        num_joints=n,  # for voting step later
    )


class TriGATEmbedding(nn.Module):
    """
    Triplet-aware GAT embedding network.

    Input: triplet-level PyG Data from `build_triplet_graph`.
    Output: [T, output_dim] L2-normalised per-triplet embeddings.
    """

    def __init__(self, config: TriGATConfig):
        super().__init__()
        self.config = config
        c = config

        self.triplet_type_embedding = nn.Embedding(
            NUM_TRIPLET_TYPES, c.triplet_embedding_dim,
        )

        # Raw triplet feature dim: [pivot_x, pivot_y, d_AB_x, d_AB_y,
        # d_CB_x, d_CB_y, cos, sin, len_AB, len_CB] = 10
        in_dim = 10 + c.triplet_embedding_dim

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(c.num_layers):
            layer_in = in_dim if i == 0 else c.hidden_dim * c.num_heads
            is_last = i == c.num_layers - 1
            concat = not is_last
            out_dim = c.hidden_dim * c.num_heads if concat else c.hidden_dim

            self.gat_layers.append(
                GATv2Conv(
                    in_channels=layer_in,
                    out_channels=c.hidden_dim,
                    heads=c.num_heads,
                    concat=concat,
                    dropout=c.dropout,
                    add_self_loops=False,
                )
            )
            self.norms.append(
                nn.LayerNorm(out_dim) if c.use_layer_norm else nn.Identity()
            )

        self.projection = nn.Linear(c.hidden_dim, c.output_dim)
        self.final_norm = nn.LayerNorm(c.output_dim) if c.use_layer_norm else nn.Identity()
        self.act = nn.ELU()
        self.dropout_layer = nn.Dropout(c.dropout)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: triplet-level PyG Data

        Returns:
            [T, output_dim] per-triplet embeddings
        """
        x = data.x
        edge_index = data.edge_index

        # Append triplet-type embedding
        type_emb = self.triplet_type_embedding(data.triplet_types)
        x = torch.cat([x, type_emb], dim=-1)

        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            x = gat(x, edge_index)
            x = norm(x)
            x = self.act(x)
            if i < len(self.gat_layers) - 1:
                x = self.dropout_layer(x)

        emb = self.projection(x)
        emb = self.final_norm(emb)

        if self.config.l2_normalize:
            emb = F.normalize(emb, p=2, dim=-1)

        return emb


def vote_joint_labels(
    triplet_labels: torch.Tensor,    # [T] cluster ID per triplet
    joint_pos_in_triplet: torch.Tensor,  # [T, 3] joint node indices (A, B, C)
    num_joints: int,
    pivot_weight: float = 2.0,
) -> torch.Tensor:
    """
    Vote each joint to a cluster based on the triplets that contain it.

    Pivot position in a triplet gets higher weight (2.0) than wing positions
    (1.0) because the pivot's articulation is the most distinguishing feature.

    Args:
        triplet_labels: [T] cluster ID per triplet
        joint_pos_in_triplet: [T, 3] (A_node, B_node, C_node) for each triplet
        num_joints: total joints in the scene
        pivot_weight: weight for pivot voting

    Returns:
        [num_joints] cluster ID per joint
    """
    device = triplet_labels.device
    n_clusters = int(triplet_labels.max().item()) + 1

    # Accumulate votes [num_joints, n_clusters]
    votes = torch.zeros(num_joints, n_clusters, device=device)

    for t_idx in range(triplet_labels.size(0)):
        label = int(triplet_labels[t_idx].item())
        A, B, C = joint_pos_in_triplet[t_idx].tolist()
        votes[A, label] += 1.0
        votes[B, label] += pivot_weight
        votes[C, label] += 1.0

    return votes.argmax(dim=1)
