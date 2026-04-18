"""
SA-GAT v2: SA-GAT with optional visual feature input.

Identical to SA-GAT in every way except that it can accept an additional
per-node feature vector (e.g. MobileNetV2 features sampled at each keypoint
location). The features are projected through a linear layer to a configurable
embedding dim and concatenated with the standard input (positions, type
embedding) before the first GAT layer.

If `data.features` is not present, behaves exactly like SA-GAT.

Config: SAGATV2Config (extends SAGATConfig with two extra fields)
    visual_feature_dim:        input dim of the cached features (e.g. 320 for MobileNetV2 layer 17)
    visual_feature_proj_dim:   output dim of the projection layer (e.g. 32)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from sa_gat import (
    SAGATLayer,
    NUM_JOINT_TYPES,
    COCO_SKELETON,
    _SAME_LIMB,
    _HOP_DISTANCES,
    _MAX_HOPS,
    CAT_CROSS_BODY,
    CAT_SAME_LIMB,
    CAT_SKELETAL_NEIGHBOR,
    CAT_SAME_TYPE,
    NUM_CATEGORIES,
)
from config import SAGATV2Config


class SAGATV2Embedding(nn.Module):
    """SA-GAT v2 with optional visual feature input."""

    def __init__(self, config: SAGATV2Config):
        super().__init__()
        self.config = config
        c = config

        self.joint_embedding = nn.Embedding(c.num_joint_types, c.joint_embedding_dim)

        # Visual feature projection (only used if data.features is present)
        self.visual_proj = nn.Sequential(
            nn.Linear(c.visual_feature_dim, c.visual_feature_proj_dim),
            nn.LayerNorm(c.visual_feature_proj_dim),
            nn.GELU(),
        )

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

        self.register_buffer("hop_distances", _HOP_DISTANCES.clone())

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
    ):
        src, dst = edge_index
        type_src = joint_types[src]
        type_dst = joint_types[dst]
        E = src.size(0)
        device = edge_index.device

        same_type = (type_src == type_dst)

        features_list = []

        if self.config.use_type_pair_attention:
            is_skel = self.skel_neighbor_mat[type_src, type_dst]
            is_limb = self.same_limb_mat[type_src, type_dst]

            categories = torch.full((E,), CAT_CROSS_BODY, dtype=torch.long, device=device)
            categories[is_limb] = CAT_SAME_LIMB
            categories[is_skel] = CAT_SKELETAL_NEIGHBOR
            categories[same_type] = CAT_SAME_TYPE

            cat_onehot = F.one_hot(categories, NUM_CATEGORIES).float()
            features_list.append(cat_onehot)

        if self.config.use_position_encoding:
            pos_src = positions[src]
            pos_dst = positions[dst]
            spatial_dist = (pos_src - pos_dst).norm(dim=1, keepdim=True)
            hop_dist = (self.hop_distances[type_src, type_dst] / _MAX_HOPS).unsqueeze(1)
            same_type_feat = same_type.float().unsqueeze(1)
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

        # Visual feature projection (if features provided)
        if hasattr(data, "features") and data.features is not None:
            visual = self.visual_proj(data.features)
            x = torch.cat([x, visual], dim=-1)

        positions = data.x[:, :2]
        edge_features, same_type_mask = self._compute_edge_features(
            edge_index, data.joint_types, positions,
        )

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
