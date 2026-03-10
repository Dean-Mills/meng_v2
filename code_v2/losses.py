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


class SlotAttentionLoss(nn.Module):
    """
    Loss for slot attention grouping head.

    Uses Hungarian matching to find the optimal bijection between predicted
    slots and ground truth people, then computes cross entropy on the matched
    assignments.

    Unlike DEC this uses ground truth labels directly — the model learns
    from labelled data rather than self-supervising.

    Args:
        config: LossConfig — uses slot_weight.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.weight = config.slot_weight

    def forward(
        self,
        logits: torch.Tensor,
        person_labels: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Args:
            logits:        [N, K]  unnormalised per-joint slot scores
            person_labels: [N]     ground truth person ID per joint

        Returns:
            {
                'total_loss': scalar tensor
                'accuracy':   float  Hungarian-matched assignment accuracy
            }
        """
        n, k = logits.shape
        device = logits.device

        unique_labels = torch.unique(person_labels)
        num_people    = len(unique_labels)

        # Remap labels to 0..P-1 in case they aren't contiguous
        label_map     = {v.item(): i for i, v in enumerate(unique_labels)}
        mapped_labels = torch.tensor(
            [label_map[l.item()] for l in person_labels],
            device=device, dtype=torch.long,
        )

        # Soft assignments for matching
        probs = F.softmax(logits, dim=1)   # [N, K]

        # Build cost matrix [num_people, K] for Hungarian matching
        # Cost = negative overlap between ground truth person i and slot j
        cost = torch.zeros(num_people, k, device=device)
        for p in range(num_people):
            mask = (mapped_labels == p).float()   # [N]
            cost[p] = -(probs * mask.unsqueeze(1)).sum(dim=0)

        # Hungarian matching on CPU
        cost_np          = cost.detach().cpu().numpy()
        from scipy.optimize import linear_sum_assignment
        row_idx, col_idx = linear_sum_assignment(cost_np)

        # Build matched targets — each joint's target is the slot assigned
        # to its ground truth person
        slot_for_person = torch.zeros(num_people, dtype=torch.long, device=device)
        for r, c in zip(row_idx, col_idx):
            slot_for_person[r] = c

        targets = slot_for_person[mapped_labels]   # [N]

        # Cross entropy loss
        loss = F.cross_entropy(logits, targets) * self.weight

        # Accuracy
        pred     = logits.argmax(dim=1)
        accuracy = (pred == targets).float().mean().item()

        return {
            "total_loss": loss,
            "accuracy":   accuracy,
        }