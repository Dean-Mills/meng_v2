"""
Loss functions for GAT + DETR pose grouping.

Three components:
1. Existence Loss (λ=1.0) - BCE for person detection
2. Assignment Loss (λ=5.0) - Cross-entropy for joint assignment
3. Contrastive Loss (λ=2.0) - Embedding space structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass
class LossConfig:
    # Loss weights
    lambda_existence: float = 1.0
    lambda_assignment: float = 5.0
    lambda_contrastive: float = 2.0
    lambda_count: float = 2.0
    
    # Contrastive loss
    contrastive_margin: float = 0.5
    
    # Assignment loss
    label_smoothing: float = 0.0


class HungarianMatcher:
    """
    Matches predicted people to ground truth people using Hungarian algorithm.
    
    Cost is based on:
    1. Existence probability (want high for matched)
    2. Assignment scores (want high for correct joints)
    """
    
    def __init__(self, cost_existence: float = 1.0, cost_assignment: float = 1.0):
        self.cost_existence = cost_existence
        self.cost_assignment = cost_assignment
    
    @torch.no_grad()
    def match(
        self,
        existence_logits: torch.Tensor,
        assignment_scores: List[torch.Tensor],
        joint_indices_per_type: List[torch.Tensor],
        person_labels: torch.Tensor,
        joint_types: torch.Tensor,
        num_gt_people: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform Hungarian matching between predictions and ground truth.
        
        Args:
            existence_logits: [M] predicted existence logits
            assignment_scores: List of [M, K_i] assignment scores per joint type
            joint_indices_per_type: List of [K_i] global joint indices per type
            person_labels: [N] ground truth person ID for each joint
            joint_types: [N] joint type for each joint
            num_gt_people: Number of ground truth people
        
        Returns:
            pred_indices: [num_matched] indices of matched predictions
            gt_indices: [num_matched] indices of matched ground truth
        """
        M = existence_logits.size(0)
        P = num_gt_people
        
        if P == 0:
            # No ground truth people - all predictions should be "no person"
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        
        device = existence_logits.device
        
        # Build cost matrix [M predictions x P ground truth]
        cost_matrix = torch.zeros(M, P, device=device)
        
        # Existence cost: want high probability for matched
        existence_probs = torch.sigmoid(existence_logits)  # [M]
        cost_matrix -= self.cost_existence * existence_probs.unsqueeze(1)  # [M, 1] broadcast
        
        # Assignment cost: for each joint type, penalize wrong assignments
        for joint_type, (scores, indices) in enumerate(
            zip(assignment_scores, joint_indices_per_type)
        ):
            if len(indices) == 0:
                continue
            
            # Get ground truth person labels for joints of this type
            gt_labels_for_type = person_labels[indices]  # [K]
            
            # For each GT person, find which joint (if any) belongs to them
            for gt_person in range(P):
                # Find joint of this type belonging to this GT person
                joint_mask = (gt_labels_for_type == gt_person)
                
                if joint_mask.any():
                    # Get the local index of the correct joint
                    correct_joint_local = torch.where(joint_mask)[0][0]
                    
                    # Get assignment score for this joint
                    # scores: [M, K], we want [:, correct_joint_local]
                    correct_scores = scores[:, correct_joint_local]  # [M]
                    
                    # Subtract score (lower cost = better match)
                    cost_matrix[:, gt_person] -= self.cost_assignment * correct_scores
        
        # Run Hungarian algorithm (on CPU, numpy)
        cost_np = cost_matrix.cpu().numpy()
        pred_indices_np, gt_indices_np = linear_sum_assignment(cost_np)
        
        pred_indices = torch.tensor(pred_indices_np, dtype=torch.long, device=device)
        gt_indices = torch.tensor(gt_indices_np, dtype=torch.long, device=device)
        
        return pred_indices, gt_indices


class PoseGroupingLoss(nn.Module):
    """
    Combined loss for pose grouping model.
    
    Components:
    1. Existence Loss: BCE for person detection
    2. Assignment Loss: Cross-entropy for joint-to-person assignment
    3. Contrastive Loss: Embedding clustering
    """
    
    def __init__(self, config: Optional[LossConfig] = None):
        super().__init__()
        self.config = config if config is not None else LossConfig()
        self.matcher = HungarianMatcher()
    
    def existence_loss(
        self,
        existence_logits: torch.Tensor,
        pred_indices: torch.Tensor,
        gt_indices: torch.Tensor,
        num_gt_people: int
    ) -> torch.Tensor:
        """
        Binary cross-entropy for person existence.
        
        Matched predictions -> target = 1
        Unmatched predictions -> target = 0
        """
        M = existence_logits.size(0)
        device = existence_logits.device
        
        # Create targets: 1 for matched, 0 for unmatched
        targets = torch.zeros(M, device=device)
        if len(pred_indices) > 0:
            targets[pred_indices] = 1.0
        
        # BCE loss
        loss = F.binary_cross_entropy_with_logits(existence_logits, targets)
        
        return loss
    
    def assignment_loss(
        self,
        assignment_scores: List[torch.Tensor],
        joint_indices_per_type: List[torch.Tensor],
        pred_indices: torch.Tensor,
        gt_indices: torch.Tensor,
        person_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-entropy loss for joint assignment.
        
        For each matched (pred, gt) pair:
            For each joint type:
                Target = index of joint belonging to gt person
                Loss = cross-entropy(scores, target)
        """
        if len(pred_indices) == 0:
            return torch.tensor(0.0, device=person_labels.device)
        
        device = person_labels.device
        total_loss = torch.tensor(0.0, device=device)
        num_assignments = 0
        
        for joint_type, (scores, indices) in enumerate(
            zip(assignment_scores, joint_indices_per_type)
        ):
            if len(indices) == 0:
                continue
            
            # Get ground truth labels for joints of this type
            gt_labels_for_type = person_labels[indices]  # [K]
            
            # For each matched pair
            for pred_idx, gt_idx in zip(pred_indices, gt_indices):
                # Find joint of this type belonging to GT person
                joint_mask = (gt_labels_for_type == gt_idx)
                
                if joint_mask.any():
                    # Get local index of correct joint
                    correct_joint_local = torch.where(joint_mask)[0][0]
                    
                    # Get scores for this prediction [K]
                    pred_scores = scores[pred_idx]  # [K]
                    
                    # Cross-entropy loss
                    loss = F.cross_entropy(
                        pred_scores.unsqueeze(0),  # [1, K]
                        correct_joint_local.unsqueeze(0),  # [1]
                        label_smoothing=self.config.label_smoothing
                    )
                    
                    total_loss += loss
                    num_assignments += 1
        
        if num_assignments > 0:
            total_loss = total_loss / num_assignments
        
        return total_loss
    
    def contrastive_loss(
        self,
        embeddings: torch.Tensor,
        person_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Contrastive loss on GAT embeddings.
        
        Pull same-person joints together, push different-person apart.
        """
        n = embeddings.size(0)
        margin = self.config.contrastive_margin
        
        if n < 2:
            return torch.tensor(0.0, device=embeddings.device), 0.0, 0.0
        
        # Compute pairwise cosine similarities [N, N]
        # Embeddings should already be L2 normalized
        sim_matrix = torch.mm(embeddings, embeddings.t())
        
        # Create masks
        labels = person_labels.unsqueeze(0)  # [1, N]
        same_person = (labels == labels.T).float()  # [N, N]
        
        # Remove diagonal
        mask = 1.0 - torch.eye(n, device=embeddings.device)
        pos_mask = same_person * mask
        neg_mask = (1.0 - same_person) * mask
        
        # Counts
        pos_count = pos_mask.sum().clamp(min=1)
        neg_count = neg_mask.sum().clamp(min=1)
        
        # Positive loss: maximize similarity
        pos_loss = ((1.0 - sim_matrix) * pos_mask).sum() / pos_count
        
        # Negative loss: push below margin
        neg_loss = (F.relu(sim_matrix - margin) * neg_mask).sum() / neg_count
        
        # Metrics
        avg_pos_sim = (sim_matrix * pos_mask).sum().item() / pos_count.item()
        avg_neg_sim = (sim_matrix * neg_mask).sum().item() / neg_count.item()
        
        return pos_loss + neg_loss, avg_pos_sim, avg_neg_sim
    
    def count_loss(
        self,
        count_pred: torch.Tensor,
        num_gt_people: int
    ) -> torch.Tensor:
        """
        Smooth L1 loss for person count prediction.
        
        Provides a global signal: "there should be N people total."
        """
        target = torch.tensor(float(num_gt_people), device=count_pred.device)
        return F.smooth_l1_loss(count_pred, target)
    
    def forward(
        self,
        outputs: Dict[str, Any],
        person_labels: torch.Tensor,
        joint_types: torch.Tensor,
        num_gt_people: int
    ) -> Dict[str, Any]:
        """
        Compute all losses.
        
        Args:
            outputs: Dict from PoseGroupingModel.forward()
                - existence_logits: [M]
                - assignment_scores: List of [M, K_i]
                - joint_indices_per_type: List of [K_i]
                - embeddings: [N, D]
            person_labels: [N] ground truth person IDs
            joint_types: [N] joint type indices
            num_gt_people: Number of ground truth people
        
        Returns:
            Dict with individual losses and total loss
        """
        # Hungarian matching
        pred_indices, gt_indices = self.matcher.match(
            outputs['existence_logits'],
            outputs['assignment_scores'],
            outputs['joint_indices_per_type'],
            person_labels,
            joint_types,
            num_gt_people
        )
        
        # Existence loss
        l_exist = self.existence_loss(
            outputs['existence_logits'],
            pred_indices,
            gt_indices,
            num_gt_people
        )
        
        # Assignment loss
        l_assign = self.assignment_loss(
            outputs['assignment_scores'],
            outputs['joint_indices_per_type'],
            pred_indices,
            gt_indices,
            person_labels
        )
        
        # Contrastive loss
        l_contrast, pos_sim, neg_sim = self.contrastive_loss(
            outputs['embeddings'],
            person_labels
        )
        
        # Count loss
        l_count = self.count_loss(
            outputs['count_pred'],
            num_gt_people
        )
        
        # Total loss
        cfg = self.config
        total_loss = (
            cfg.lambda_existence * l_exist +
            cfg.lambda_assignment * l_assign +
            cfg.lambda_contrastive * l_contrast +
            cfg.lambda_count * l_count
        )
        
        return {
            'total_loss': total_loss,
            'existence_loss': l_exist,
            'assignment_loss': l_assign,
            'contrastive_loss': l_contrast,
            'count_loss': l_count,
            'count_pred': outputs['count_pred'].item(),
            'pos_similarity': pos_sim,
            'neg_similarity': neg_sim,
            'num_matched': len(pred_indices),
            'num_gt_people': num_gt_people
        }


class GATOnlyLoss(nn.Module):
    """
    Simplified loss for GAT-only training (no DETR).
    
    Just contrastive loss on embeddings.
    """
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        embeddings: torch.Tensor,
        person_labels: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: [N, D] joint embeddings
            person_labels: [N] person IDs
        
        Returns:
            Dict with loss and metrics
        """
        n = embeddings.size(0)
        
        if n < 2:
            return {
                'total_loss': torch.tensor(0.0, device=embeddings.device),
                'pos_similarity': 0.0,
                'neg_similarity': 0.0
            }
        
        # Pairwise similarities
        sim_matrix = torch.mm(embeddings, embeddings.t())
        
        # Masks
        labels = person_labels.unsqueeze(0)
        same_person = (labels == labels.T).float()
        mask = 1.0 - torch.eye(n, device=embeddings.device)
        pos_mask = same_person * mask
        neg_mask = (1.0 - same_person) * mask
        
        pos_count = pos_mask.sum().clamp(min=1)
        neg_count = neg_mask.sum().clamp(min=1)
        
        # Losses
        pos_loss = ((1.0 - sim_matrix) * pos_mask).sum() / pos_count
        neg_loss = (F.relu(sim_matrix - self.margin) * neg_mask).sum() / neg_count
        
        # Metrics
        avg_pos_sim = (sim_matrix * pos_mask).sum().item() / pos_count.item()
        avg_neg_sim = (sim_matrix * neg_mask).sum().item() / neg_count.item()
        
        return {
            'total_loss': pos_loss + neg_loss,
            'pos_similarity': avg_pos_sim,
            'neg_similarity': avg_neg_sim
        }