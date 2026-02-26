import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

from config import LossConfig


class GATOnlyLoss(nn.Module):
    """
    Contrastive loss for GAT isolation training.

    Pulls embeddings of joints belonging to the same person together
    and pushes embeddings of joints from different people apart.

    Args:
        config: LossConfig — uses contrastive_margin.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.margin = config.contrastive_margin

    def forward(
        self,
        embeddings: torch.Tensor,
        person_labels: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Args:
            embeddings:    [N, D] L2-normalised joint embeddings from GAT
            person_labels: [N]   ground truth person ID per joint

        Returns:
            {
                'total_loss':     scalar tensor
                'pos_similarity': float  avg similarity between same-person pairs
                'neg_similarity': float  avg similarity between different-person pairs
            }
        """
        n = embeddings.size(0)

        if n < 2:
            return {
                "total_loss":     torch.tensor(0.0, device=embeddings.device),
                "pos_similarity": 0.0,
                "neg_similarity": 0.0,
            }

        # Pairwise cosine similarities [N, N]
        sim_matrix = torch.mm(embeddings, embeddings.t())

        # Masks
        same_person = (person_labels.unsqueeze(0) == person_labels.unsqueeze(1)).float()
        mask        = 1.0 - torch.eye(n, device=embeddings.device)
        pos_mask    = same_person * mask
        neg_mask    = (1.0 - same_person) * mask

        pos_count = pos_mask.sum().clamp(min=1)
        neg_count = neg_mask.sum().clamp(min=1)

        # Pull same-person pairs toward similarity = 1
        pos_loss = ((1.0 - sim_matrix) * pos_mask).sum() / pos_count

        # Push different-person pairs below margin
        neg_loss = (F.relu(sim_matrix - self.margin) * neg_mask).sum() / neg_count

        avg_pos_sim = (sim_matrix * pos_mask).sum().item() / pos_count.item()
        avg_neg_sim = (sim_matrix * neg_mask).sum().item() / neg_count.item()

        return {
            "total_loss":     pos_loss + neg_loss,
            "pos_similarity": avg_pos_sim,
            "neg_similarity": avg_neg_sim,
        }