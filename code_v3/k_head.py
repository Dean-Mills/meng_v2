"""
Learned K estimation head.

Small MLP that predicts the number of people from pooled GAT embeddings
and the keypoint count. Trains jointly with the GAT via L1 loss on K.

Input features:
    - mean_pool(H):  [D]  global embedding average
    - max_pool(H):   [D]  global embedding max
    - n_keypoints:   [1]  number of valid keypoints (strong prior)

Output:
    K_pred: scalar (continuous, rounded at inference)

The n_keypoints feature is the key prior — with 17 COCO joints per
person, n_keypoints / 17 gives a rough K estimate. The embeddings
refine this when some joints are missing (occlusion) or when keypoints
from different people overlap.
"""
import torch
import torch.nn as nn


class KEstimationHead(nn.Module):

    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Input: mean_pool [D] + max_pool [D] + n_keypoints [1]
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [N, D] L2-normalised joint embeddings

        Returns:
            k_pred: scalar (continuous value, round for discrete K)
        """
        n = embeddings.size(0)

        mean_pool = embeddings.mean(dim=0)  # [D]
        max_pool = embeddings.max(dim=0).values  # [D]
        n_keypoints = torch.tensor([n / 17.0], device=embeddings.device)  # normalised

        features = torch.cat([mean_pool, max_pool, n_keypoints])  # [2D + 1]
        k_pred = self.mlp(features.unsqueeze(0)).squeeze()  # scalar

        return k_pred

    def predict(self, embeddings: torch.Tensor) -> int:
        """Predict discrete K at inference."""
        with torch.no_grad():
            k_continuous = self.forward(embeddings)
            k = max(1, round(k_continuous.item()))
        return k
