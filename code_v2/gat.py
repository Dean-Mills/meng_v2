import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data

from config import GATConfig


class GATEmbedding(nn.Module):
    """
    GAT-based joint embedding network.

    Takes a PyG graph of joints and produces an embedding per joint.
    Joints belonging to the same person should cluster together in
    embedding space — this is enforced by the contrastive loss.

    Architecture:
        - Joint type embedding concatenated to raw node features
        - N-1 GAT layers with concat heads (output = hidden_dim * num_heads)
        - Final GAT layer with averaged heads (output = hidden_dim)
        - Linear projection to output_dim
        - Optional L2 normalisation
    """

    def __init__(self, config: GATConfig):
        super().__init__()
        self.config = config
        c = config

        # Learned embedding per joint type (nose, left_knee, etc.)
        self.joint_embedding = nn.Embedding(c.num_joint_types, c.joint_embedding_dim)

        self.gat_layers = nn.ModuleList()
        self.norms       = nn.ModuleList()

        for i in range(c.num_layers):
            in_dim  = c.input_dim if i == 0 else c.hidden_dim * c.num_heads
            is_last = i == c.num_layers - 1
            concat  = not is_last
            out_dim = c.hidden_dim * c.num_heads if concat else c.hidden_dim

            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=c.hidden_dim,
                    heads=c.num_heads,
                    dropout=c.dropout,
                    concat=concat,
                )
            )
            self.norms.append(
                nn.LayerNorm(out_dim) if c.use_layer_norm else nn.Identity()
            )

        self.projection = nn.Linear(c.hidden_dim, c.output_dim)
        self.final_norm = nn.LayerNorm(c.output_dim) if c.use_layer_norm else nn.Identity()
        self.act        = nn.ELU()
        self.dropout    = nn.Dropout(c.dropout)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyG Data with:
                  x            [N, 4]   node features
                  joint_types  [N]      joint type indices 0-16
                  edge_index   [2, E]   kNN graph

        Returns:
            embeddings: [N, output_dim] L2-normalised joint embeddings
        """
        assert data.x is not None, "Graph has no node features"
        x: torch.Tensor = data.x       # [N, 4]
        edge_index = data.edge_index   # [2, E]

        # Concatenate learned joint type embedding to raw features
        joint_emb = self.joint_embedding(data.joint_types)
        x = torch.cat([x, joint_emb], dim=-1)   # [N, input_dim]

        # GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            x = gat(x, edge_index)
            x = norm(x)
            x = self.act(x)
            if i < len(self.gat_layers) - 1:
                x = self.dropout(x)

        # Project to output embedding space
        emb = self.projection(x)
        emb = self.final_norm(emb)

        if self.config.l2_normalize:
            emb = F.normalize(emb, p=2, dim=-1)

        return emb   # [N, output_dim]