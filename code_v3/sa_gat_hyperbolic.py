"""
SA-GAT with Lorentz-model hyperbolic output embeddings.

The internal GAT attention layers remain Euclidean (same as SA-GAT). Only
the final embedding is projected onto the Lorentz hyperboloid via
exponential map at origin. Contrastive loss uses hyperbolic distance
instead of Euclidean.

Hypothesis: skeletons are trees and multi-person pose scenes are forests.
Hyperbolic space has exponentially growing volume with radius, which is
a natural fit for tree-structured data. This could give cleaner separation
between different people's joints.

Two experimental hypotheses:
  H1: hyperbolic beats Euclidean at matched embedding dimension.
  H2: hyperbolic matches Euclidean at significantly lower dimension.

Uses geoopt's Lorentz manifold for numerical stability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

import geoopt

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
from config import SAGATHyperbolicConfig


class SAGATHyperbolicEmbedding(nn.Module):
    """
    SA-GAT with Lorentz-model hyperbolic output.

    Internal attention layers are Euclidean (identical to SA-GAT). The
    final projection outputs a tangent vector at origin, which is mapped
    to the Lorentz hyperboloid via expmap0. Forward returns the hyperboloid
    point. Use `hyperbolic_distance(a, b)` for pairwise distances.
    """

    def __init__(self, config: SAGATHyperbolicConfig):
        super().__init__()
        self.config = config
        c = config

        # Manifold (Lorentz with K=1 → curvature -1)
        self.manifold = geoopt.Lorentz(k=1.0)

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

        # Project to output_dim in tangent space. The Lorentz hyperboloid
        # lives in R^(output_dim + 1) with the extra time coordinate added
        # automatically by expmap0 when we pass a (output_dim)-vector that
        # we treat as a tangent vector at origin with zero time component.
        self.projection = nn.Linear(c.hidden_dim, c.output_dim)
        self.final_norm = nn.LayerNorm(c.output_dim) if c.use_layer_norm else nn.Identity()
        self.act = nn.ELU()
        self.dropout_layer = nn.Dropout(c.dropout)

        # Learnable scale for the tangent vector before expmap0.
        # Keeps hyperbolic distances in a reasonable range at initialisation.
        # Initialised so that random LayerNorm-scale tangent vectors (norm
        # ~sqrt(D)) get scaled to ~1.0 geodesic distance from origin.
        init_scale = 1.0 / (c.output_dim ** 0.5)
        self.tangent_scale = nn.Parameter(torch.tensor(init_scale))

        # Precompute lookup tables as buffers
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

    def _compute_edge_features(self, edge_index, joint_types, positions):
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

    def _to_hyperboloid(self, tangent_vec: torch.Tensor) -> torch.Tensor:
        """
        Project tangent vector in R^D to Lorentz hyperboloid in R^(D+1).

        Prepends a zero time component (so the vector is tangent at the
        origin), then applies expmap0.

        Args:
            tangent_vec: [N, D]

        Returns:
            [N, D+1] points on the hyperboloid
        """
        # Prepend zero time component
        zero_time = torch.zeros(
            tangent_vec.shape[0], 1,
            device=tangent_vec.device, dtype=tangent_vec.dtype,
        )
        v = torch.cat([zero_time, tangent_vec], dim=-1)  # [N, D+1]
        return self.manifold.expmap0(v)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Returns [N, output_dim + 1] points on the Lorentz hyperboloid.
        Use `self.hyperbolic_distance` for pairwise distances.
        """
        assert data.x is not None
        x = data.x
        edge_index = data.edge_index

        joint_emb = self.joint_embedding(data.joint_types)
        x = torch.cat([x, joint_emb], dim=-1)

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

        tangent = self.projection(x)
        tangent = self.final_norm(tangent)

        # Apply learnable scale to keep hyperbolic distances manageable
        tangent = tangent * self.tangent_scale

        # Map tangent vector at origin to hyperboloid
        emb = self._to_hyperboloid(tangent)

        return emb

    def hyperbolic_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Geodesic distance between two points on the Lorentz hyperboloid."""
        return self.manifold.dist(a, b)
