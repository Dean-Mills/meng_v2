"""
K-head with the per-type detection count histogram (PEng, 2026-10-26).

The diagnostics pass found that wrong-K scenes carry most of the remaining autonomous error
(HigherHRNet / YOLO: with K right the pipeline beats native grouping). The MEng KEstimationHead
(k_head.py, unchanged) sees the pooled embedding and only the total joint count n / 17; the bound
K_lb = max per-type count, which the PEng clamp max(K-head, K_lb) applies afterwards, is invisible
to it. This variant adds the count structure to the input:

    mean_pool(H) [D] + max_pool(H) [D] + n / 17 [1] + per-type counts [17] + K_lb [1]

Counts are raw (people units, like n / 17). K_lb is passed explicitly because a small ReLU MLP only
approximates a max over 17 inputs. Same MLP widths and L1-on-K training as KEstimationHead
(train_k_head_aug.py --head hist).

Interface differs from KEstimationHead (the head needs the joint types), so eval scripts call
predict_k(head, emb, joint_types), which also accepts the MEng head. Checkpoints carry
k_head_type = "hist" (eval_peng_gap.load_models branches on it).
"""
import torch
import torch.nn as nn

NUM_JOINT_TYPES = 17


class KHistHead(nn.Module):

    uses_type_counts = True

    def __init__(self, embedding_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 1 + NUM_JOINT_TYPES + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, embeddings: torch.Tensor, joint_types: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings:  [N, D] L2-normalised joint embeddings
            joint_types: [N] joint type ids (0..16)

        Returns:
            k_pred: scalar (continuous value, round for discrete K)
        """
        n = embeddings.size(0)
        counts = torch.bincount(joint_types.to(embeddings.device).long(),
                                minlength=NUM_JOINT_TYPES).float()           # [17]
        features = torch.cat([
            embeddings.mean(dim=0),
            embeddings.max(dim=0).values,
            torch.tensor([n / 17.0], device=embeddings.device),
            counts,
            counts.max().unsqueeze(0),                                         # K_lb
        ])
        return self.mlp(features.unsqueeze(0)).squeeze()

    def predict(self, embeddings: torch.Tensor, joint_types: torch.Tensor) -> int:
        """Predict discrete K at inference."""
        with torch.no_grad():
            return max(1, round(self.forward(embeddings, joint_types).item()))


def k_head_output(k_head, embeddings: torch.Tensor, joint_types: torch.Tensor) -> torch.Tensor:
    """Continuous K for training (either head)."""
    if getattr(k_head, "uses_type_counts", False):
        return k_head(embeddings, joint_types)
    return k_head(embeddings)


def predict_k(k_head, embeddings: torch.Tensor, joint_types: torch.Tensor) -> int:
    """Discrete K for either head (KEstimationHead ignores the joint types)."""
    if getattr(k_head, "uses_type_counts", False):
        return k_head.predict(embeddings, joint_types)
    return k_head.predict(embeddings)
