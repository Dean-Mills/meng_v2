import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GraphPartitioningConfig


class EdgeClassifier(nn.Module):
    """
    Graph partitioning grouping head.

    Predicts for every pair of joints whether they belong to the same person.
    After thresholding, connected components on the resulting affinity graph
    recover the person groups.

    For each pair (i, j) the feature vector is [e_i - e_j, e_i * e_j],
    where e_i, e_j are L2-normalised GAT embeddings. The difference captures
    direction between the two points in embedding space; the product captures
    their element-wise co-activation. Together they give the MLP more signal
    than cosine similarity alone.

    This is option A — a learned classifier on top of the embeddings rather
    than simply thresholding dot products. The model learns what "same person"
    means in embedding space from labelled data.

    Args:
        config:        GraphPartitioningConfig
        embedding_dim: must match GAT output_dim

    Forward:
        embeddings: [N, D]  L2-normalised joint embeddings from GAT

    Returns:
        logits:     [E]     unnormalised edge scores, one per pair
        pairs:      [E, 2]  indices (i, j) of each pair, i < j
    """

    def __init__(self, config: GraphPartitioningConfig, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        d = embedding_dim

        # Input: [e_i - e_j || e_i * e_j] → 2D
        self.mlp = nn.Sequential(
            nn.Linear(2 * d, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: [N, D]

        Returns:
            logits: [E]     one score per pair (positive = same person)
            pairs:  [E, 2]  (i, j) index pairs with i < j
        """
        n = embeddings.size(0)
        device = embeddings.device

        # Build all upper-triangle pairs (i < j)
        idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=device)  # each [E]
        pairs = torch.stack([idx_i, idx_j], dim=1)                         # [E, 2]

        e_i = embeddings[idx_i]   # [E, D]
        e_j = embeddings[idx_j]   # [E, D]

        # Pair features: difference + element-wise product
        diff    = e_i - e_j               # [E, D]
        product = e_i * e_j               # [E, D]
        pair_features = torch.cat([diff, product], dim=-1)   # [E, 2D]

        logits = self.mlp(pair_features).squeeze(-1)   # [E]

        return logits, pairs