"""
SA-GAT (Skeleton-Aware Graph Attention Network).

Modifies the GAT backbone to make attention skeleton-aware. Three
independent modifications, each configurable for ablation:

1. Type-Pair Attention Bias — add a learned scalar bias to attention
   scores based on edge category (SAME_TYPE, SKELETAL_NEIGHBOR,
   SAME_LIMB, CROSS_BODY). Same-type edges get a different attention
   profile than cross-body edges.

2. Skeleton-Relative Position Encoding — inject spatial distance,
   skeleton hop distance, and same-type indicator as an MLP-encoded
   bias on attention scores.

3. Same-Type Repulsion Heads — dedicate specific attention heads
   exclusively to same-type neighbor interactions by masking out
   non-same-type edges on those heads.

Uses PyG's GATv2Conv as the base layer for numerical stability,
then applies modifications to the attention computation via hooks
on the edge features.

Implementation approach: wrap GATv2Conv and modify edge-level
attention via edge_attr. GATv2Conv supports edge_attr as an additive
bias on attention logits when configured with edge_dim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

from config import SAGATConfig


# COCO skeleton
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

NUM_JOINT_TYPES = 17

# Category indices
CAT_SAME_TYPE = 0
CAT_SKELETAL_NEIGHBOR = 1
CAT_SAME_LIMB = 2
CAT_CROSS_BODY = 3
NUM_CATEGORIES = 4

# Define limbs
LIMBS = [
    {0, 1, 2, 3, 4},        # head
    {5, 7, 9},                # left arm
    {6, 8, 10},               # right arm
    {5, 6, 11, 12},          # torso
    {11, 13, 15},            # left leg
    {12, 14, 16},            # right leg
]


def _build_skeleton_neighbors() -> set:
    neighbors = set()
    for a, b in COCO_SKELETON:
        neighbors.add((a, b))
        neighbors.add((b, a))
    return neighbors


def _build_same_limb() -> set:
    skel = _build_skeleton_neighbors()
    same_limb = set()
    for limb in LIMBS:
        for a in limb:
            for b in limb:
                if a != b and (a, b) not in skel:
                    same_limb.add((a, b))
    return same_limb


def _build_hop_distances() -> torch.Tensor:
    adj = [[] for _ in range(NUM_JOINT_TYPES)]
    for a, b in COCO_SKELETON:
        adj[a].append(b)
        adj[b].append(a)
    hops = torch.full((NUM_JOINT_TYPES, NUM_JOINT_TYPES), 8.0)
    for src in range(NUM_JOINT_TYPES):
        dist = [-1] * NUM_JOINT_TYPES
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        for dst in range(NUM_JOINT_TYPES):
            if dist[dst] >= 0:
                hops[src, dst] = float(dist[dst])
    return hops


_SKEL_NEIGHBORS = _build_skeleton_neighbors()
_SAME_LIMB = _build_same_limb()
_HOP_DISTANCES = _build_hop_distances()
_MAX_HOPS = _HOP_DISTANCES.max().item()


class SAGATLayer(nn.Module):
    """
    SA-GAT layer built on top of GATv2Conv.

    Uses GATv2Conv with edge_dim to inject skeleton-aware information
    as edge features that modify attention scores. This keeps PyG's
    optimised attention computation while adding the modifications.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        dropout: float,
        use_type_pair_attention: bool,
        use_position_encoding: bool,
        use_repulsion_heads: bool,
        n_repulsion_heads: int = 1,
        concat: bool = True,
    ):
        super().__init__()
        self.use_type_pair_attention = use_type_pair_attention
        self.use_position_encoding = use_position_encoding
        self.use_repulsion_heads = use_repulsion_heads
        self.n_repulsion_heads = n_repulsion_heads
        self.num_heads = num_heads
        self.concat = concat

        # Compute edge feature dimension
        edge_feat_dim = 0
        if use_type_pair_attention:
            edge_feat_dim += NUM_CATEGORIES  # one-hot category
        if use_position_encoding:
            edge_feat_dim += 3  # spatial_dist, hop_dist, same_type

        # Edge feature encoder → fixed edge_dim for GATv2Conv
        if edge_feat_dim > 0:
            edge_enc_dim = 16  # fixed intermediate dimension
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_feat_dim, 32),
                nn.ReLU(),
                nn.Linear(32, edge_enc_dim),
            )
            gat_edge_dim = edge_enc_dim
        else:
            self.edge_encoder = None
            gat_edge_dim = None

        # Main attention layer — standard or repulsion split
        if use_repulsion_heads and n_repulsion_heads > 0:
            n_standard = num_heads - n_repulsion_heads
            assert n_standard > 0, "Need at least 1 non-repulsion head"

            self.gat_standard = GATv2Conv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=n_standard,
                dropout=dropout,
                concat=True,
                edge_dim=gat_edge_dim,
            )
            self.gat_repulsion = GATv2Conv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=n_repulsion_heads,
                dropout=dropout,
                concat=True,
                edge_dim=gat_edge_dim,
            )
            self.n_standard = n_standard
        else:
            self.gat_standard = GATv2Conv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=num_heads,
                dropout=dropout,
                concat=True,
                edge_dim=gat_edge_dim,
            )
            self.gat_repulsion = None
            self.n_standard = num_heads

        self.out_dim = out_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features_raw: torch.Tensor | None = None,
        same_type_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                 [N, in_dim]
            edge_index:        [2, E]
            edge_features_raw: [E, F] raw edge features (categories + pos encoding)
            same_type_mask:    [E] bool mask for repulsion heads

        Returns:
            out: [N, out_dim * num_heads] if concat else [N, out_dim]
        """
        # Encode edge features
        edge_attr = None
        if self.edge_encoder is not None and edge_features_raw is not None:
            edge_attr = self.edge_encoder(edge_features_raw)  # [E, num_heads]

        if self.gat_repulsion is not None and same_type_mask is not None:
            # Standard heads: all edges
            h_std = self.gat_standard(x, edge_index, edge_attr=edge_attr)

            # Repulsion heads: same-type edges only
            st_edges = edge_index[:, same_type_mask]
            if st_edges.size(1) > 0:
                st_edge_attr = edge_attr[same_type_mask] if edge_attr is not None else None
                h_rep = self.gat_repulsion(x, st_edges, edge_attr=st_edge_attr)
            else:
                n = x.size(0)
                h_rep = torch.zeros(
                    n, self.n_repulsion_heads * self.out_dim,
                    device=x.device,
                )

            h = torch.cat([h_std, h_rep], dim=-1)
        else:
            h = self.gat_standard(x, edge_index, edge_attr=edge_attr)

        if not self.concat:
            # Average over heads
            n = x.size(0)
            h = h.view(n, self.num_heads, self.out_dim).mean(dim=1)

        return h


class SAGATEmbedding(nn.Module):
    """
    Skeleton-Aware GAT embedding network.

    Drop-in replacement for GATEmbedding — same forward signature,
    same output shape.
    """

    def __init__(self, config: SAGATConfig):
        super().__init__()
        self.config = config
        c = config

        self.joint_embedding = nn.Embedding(c.num_joint_types, c.joint_embedding_dim)

        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(c.num_layers):
            in_dim = c.input_dim if i == 0 else c.hidden_dim * c.num_heads
            is_last = i == c.num_layers - 1
            concat = not is_last
            out_dim = c.hidden_dim * c.num_heads if concat else c.hidden_dim

            self.gat_layers.append(
                SAGATLayer(
                    in_dim=in_dim,
                    out_dim=c.hidden_dim,
                    num_heads=c.num_heads,
                    dropout=c.dropout,
                    use_type_pair_attention=c.use_type_pair_attention,
                    use_position_encoding=c.use_position_encoding,
                    use_repulsion_heads=c.use_repulsion_heads,
                    n_repulsion_heads=c.n_repulsion_heads,
                    concat=concat,
                )
            )
            self.norms.append(
                nn.LayerNorm(out_dim) if c.use_layer_norm else nn.Identity()
            )

        self.projection = nn.Linear(c.hidden_dim, c.output_dim)
        self.final_norm = nn.LayerNorm(c.output_dim) if c.use_layer_norm else nn.Identity()
        self.act = nn.ELU()
        self.dropout_layer = nn.Dropout(c.dropout)

        # Precompute lookup tables as buffers
        self.register_buffer("hop_distances", _HOP_DISTANCES.clone())

        # Skeleton neighbor lookup [17, 17] bool
        skel_nb = torch.zeros(NUM_JOINT_TYPES, NUM_JOINT_TYPES, dtype=torch.bool)
        for a, b in COCO_SKELETON:
            skel_nb[a, b] = True
            skel_nb[b, a] = True
        self.register_buffer("skel_neighbor_mat", skel_nb)

        same_limb_mat = torch.zeros(NUM_JOINT_TYPES, NUM_JOINT_TYPES, dtype=torch.bool)
        for a, b in _SAME_LIMB:
            same_limb_mat[a, b] = True
        self.register_buffer("same_limb_mat", same_limb_mat)

    def _compute_edge_features(
        self,
        edge_index: torch.Tensor,
        joint_types: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Compute per-edge features and same-type mask.

        Returns:
            edge_features: [E, F] or None
            same_type_mask: [E] bool or None
        """
        src, dst = edge_index
        type_src = joint_types[src]
        type_dst = joint_types[dst]
        E = src.size(0)
        device = edge_index.device

        same_type = (type_src == type_dst)

        features_list = []

        if self.config.use_type_pair_attention:
            # One-hot category encoding [E, 4]
            is_skel = self.skel_neighbor_mat[type_src, type_dst]
            is_limb = self.same_limb_mat[type_src, type_dst]

            categories = torch.full((E,), CAT_CROSS_BODY, dtype=torch.long, device=device)
            categories[is_limb] = CAT_SAME_LIMB
            categories[is_skel] = CAT_SKELETAL_NEIGHBOR
            categories[same_type] = CAT_SAME_TYPE

            cat_onehot = F.one_hot(categories, NUM_CATEGORIES).float()  # [E, 4]
            features_list.append(cat_onehot)

        if self.config.use_position_encoding:
            pos_src = positions[src]
            pos_dst = positions[dst]
            spatial_dist = (pos_src - pos_dst).norm(dim=1, keepdim=True)  # [E, 1]
            hop_dist = (self.hop_distances[type_src, type_dst] / _MAX_HOPS).unsqueeze(1)  # [E, 1]
            same_type_feat = same_type.float().unsqueeze(1)  # [E, 1]
            features_list.append(torch.cat([spatial_dist, hop_dist, same_type_feat], dim=1))

        edge_features = torch.cat(features_list, dim=1) if features_list else None
        same_type_mask = same_type if self.config.use_repulsion_heads else None

        return edge_features, same_type_mask

    def forward(self, data: Data) -> torch.Tensor:
        assert data.x is not None
        x = data.x
        edge_index = data.edge_index

        # Joint type embedding
        joint_emb = self.joint_embedding(data.joint_types)
        x = torch.cat([x, joint_emb], dim=-1)

        # Compute edge features once
        positions = data.x[:, :2]
        edge_features, same_type_mask = self._compute_edge_features(
            edge_index, data.joint_types, positions,
        )

        # SA-GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            x = gat(x, edge_index, edge_features, same_type_mask)
            x = norm(x)
            x = self.act(x)
            if i < len(self.gat_layers) - 1:
                x = self.dropout_layer(x)

        emb = self.projection(x)
        emb = self.final_norm(emb)

        if self.config.l2_normalize:
            emb = F.normalize(emb, p=2, dim=-1)

        return emb
