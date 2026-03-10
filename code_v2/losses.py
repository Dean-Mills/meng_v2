import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List

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


class GraphPartitioningLoss(nn.Module):
    """
    Loss for graph partitioning grouping head.

    Binary cross entropy over all joint pairs — same person (positive) or
    different person (negative). Class imbalance is handled with pos_weight:
    with K people and J joints per person there are K*C(J,2) positive pairs
    and many more negative pairs. pos_weight upscales the positive class
    so the model does not collapse to predicting all-negative.

    Also computes edge F1 and grouping accuracy (connected components +
    Hungarian matching) for monitoring.

    Args:
        config: LossConfig — uses partition_weight.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.weight = config.partition_weight

    def forward(
        self,
        logits: torch.Tensor,
        pairs:  torch.Tensor,
        person_labels: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Args:
            logits:        [E]    unnormalised edge scores from EdgeClassifier
            pairs:         [E, 2] joint index pairs (i, j) with i < j
            person_labels: [N]    ground truth person ID per joint

        Returns:
            {
                'total_loss':      scalar tensor
                'edge_f1':         float  F1 on edge same/different prediction
                'edge_accuracy':   float  binary accuracy on edges
                'grouping_accuracy': float  Hungarian-matched grouping accuracy
            }
        """
        device = logits.device

        # Ground truth edge labels: 1 if same person, 0 if different
        labels_i = person_labels[pairs[:, 0]]
        labels_j = person_labels[pairs[:, 1]]
        edge_labels = (labels_i == labels_j).float()   # [E]

        # Class imbalance weight
        n_pos = edge_labels.sum().clamp(min=1)
        n_neg = (1 - edge_labels).sum().clamp(min=1)
        pos_weight = (n_neg / n_pos).clamp(max=10.0)

        loss = F.binary_cross_entropy_with_logits(
            logits, edge_labels,
            pos_weight=pos_weight,
        ) * self.weight

        # Edge metrics
        with torch.no_grad():
            preds = (logits.sigmoid() > 0.5).float()

            tp = (preds * edge_labels).sum().item()
            fp = (preds * (1 - edge_labels)).sum().item()
            fn = ((1 - preds) * edge_labels).sum().item()

            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            edge_f1   = 2 * precision * recall / (precision + recall + 1e-8)
            edge_acc  = (preds == edge_labels).float().mean().item()

            # Grouping accuracy via connected components + Hungarian matching
            grouping_acc = _grouping_accuracy(
                preds, pairs, person_labels, device
            )

        return {
            "total_loss":        loss,
            "edge_f1":           edge_f1,
            "edge_accuracy":     edge_acc,
            "grouping_accuracy": grouping_acc,
        }


def _grouping_accuracy(
    edge_preds:    torch.Tensor,
    pairs:         torch.Tensor,
    person_labels: torch.Tensor,
    device:        torch.device,
) -> float:
    """
    Recover groups from predicted edges via connected components,
    then compute Hungarian-matched accuracy against ground truth.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n = person_labels.size(0)
    same_mask = edge_preds.bool().cpu().numpy()
    pairs_np  = pairs.cpu().numpy()
    labels_np = person_labels.cpu().numpy()

    # Union-Find for connected components
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for (i, j), same in zip(pairs_np, same_mask):
        if same:
            union(int(i), int(j))

    pred_labels = np.array([find(i) for i in range(n)])

    # Remap to 0..C-1
    unique_pred = np.unique(pred_labels)
    pred_remap  = {v: i for i, v in enumerate(unique_pred)}
    pred_labels = np.array([pred_remap[l] for l in pred_labels])

    unique_true = np.unique(labels_np)
    true_remap  = {v: i for i, v in enumerate(unique_true)}
    true_labels = np.array([true_remap[l] for l in labels_np])

    k_pred = len(unique_pred)
    k_true = len(unique_true)
    k      = max(k_pred, k_true)

    # Confusion matrix + Hungarian
    confusion = np.zeros((k, k), dtype=np.int64)
    for p, t in zip(pred_labels, true_labels):
        confusion[p, t] += 1

    row_idx, col_idx = linear_sum_assignment(-confusion)
    correct = confusion[row_idx, col_idx].sum()
    return correct / n