# detr_decoder.py
"""
DETR-Style Person Decoder for Multi-Person Pose Grouping

Takes GAT embeddings and uses learnable person queries to:
1. Predict how many people exist (existence head)
2. Assign joints to people (assignment heads)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Union


@dataclass
class DETRConfig:
    # Input dimensions
    embedding_dim: int = 128  # Must match GAT output_dim
    
    # Person queries
    max_people: int = 10  # Maximum number of people to detect
    
    # Transformer decoder
    num_decoder_layers: int = 3
    num_heads: int = 8
    ffn_dim: int = 512
    dropout: float = 0.1
    
    # Joint types
    num_joint_types: int = 17  # COCO keypoints


class PersonExistenceHead(nn.Module):
    """
    Predicts whether each person query represents a real person.
    
    Input: [M, D] person features
    Output: [M] existence logits
    """
    
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, person_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            person_features: [M, D] features from decoder
        Returns:
            existence_logits: [M] logits (use sigmoid for probability)
        """
        return self.mlp(person_features).squeeze(-1)


class JointAssignmentHead(nn.Module):
    """
    Assigns joints to people using dot-product attention.
    
    For each person query, computes similarity with all joint embeddings
    of a specific type, producing assignment scores.
    """
    
    def __init__(self, embed_dim: int):
        super().__init__()
        # Project person features for assignment scoring
        self.person_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(
        self, 
        person_features: torch.Tensor, 
        joint_embeddings: torch.Tensor,
        joint_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            person_features: [M, D] person query features
            joint_embeddings: [K, D] embeddings of joints of ONE type
            joint_mask: [K] optional mask (1 = valid, 0 = invalid)
        
        Returns:
            assignment_scores: [M, K] scores for each (person, joint) pair
        """
        # Project person features
        person_proj = self.person_proj(person_features)  # [M, D]
        
        # Dot product similarity
        scores = torch.mm(person_proj, joint_embeddings.t())  # [M, K]
        
        # Apply mask if provided (set invalid to -inf)
        if joint_mask is not None:
            scores = scores.masked_fill(~joint_mask.bool().unsqueeze(0), float('-inf'))
        
        return scores


class DETRDecoder(nn.Module):
    """
    DETR-style decoder for multi-person pose grouping.
    
    Architecture:
        1. Learnable person queries [M, D]
        2. Transformer decoder with cross-attention to joint embeddings
        3. Existence head: predicts if each query is a real person
        4. Assignment heads: assigns joints to people (one head per joint type)
    """
    
    def __init__(self, config: Optional[DETRConfig] = None):
        super().__init__()
        self.config = config if config is not None else DETRConfig()
        c = self.config
        
        # Learnable person queries
        self.person_queries = nn.Parameter(
            torch.randn(c.max_people, c.embedding_dim) * 0.02
        )
        
        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=c.embedding_dim,
            nhead=c.num_heads,
            dim_feedforward=c.ffn_dim,
            dropout=c.dropout,
            activation='relu',
            batch_first=True
        )
        
        # Stack of decoder layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=c.num_decoder_layers
        )
        
        # Existence head
        self.existence_head = PersonExistenceHead(c.embedding_dim, c.dropout)
        
        # One assignment head per joint type
        self.assignment_heads = nn.ModuleList([
            JointAssignmentHead(c.embedding_dim) 
            for _ in range(c.num_joint_types)
        ])
        
        # Layer norm for input embeddings
        self.joint_norm = nn.LayerNorm(c.embedding_dim)
        self.query_norm = nn.LayerNorm(c.embedding_dim)
    
    def forward(
        self, 
        joint_embeddings: torch.Tensor,
        joint_types: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Forward pass of DETR decoder.
        
        Args:
            joint_embeddings: [N, D] embeddings from GAT
            joint_types: [N] joint type indices (0-16)
        
        Returns:
            dict with:
                - existence_logits: [M] person existence scores
                - assignment_scores: List of [M, K_i] scores per joint type
                - person_features: [M, D] decoded person features
        """
        N = joint_embeddings.size(0)
        M = self.config.max_people
        device = joint_embeddings.device
        
        # Normalize inputs
        joint_embeddings = self.joint_norm(joint_embeddings)
        
        # Prepare person queries [1, M, D] for batch processing
        queries = self.query_norm(self.person_queries).unsqueeze(0)  # [1, M, D]
        
        # Prepare joint embeddings as memory [1, N, D]
        memory = joint_embeddings.unsqueeze(0)  # [1, N, D]
        
        # Decode: person queries attend to joint embeddings
        # Output: [1, M, D]
        decoded = self.decoder(
            tgt=queries,
            memory=memory
        )
        
        # Remove batch dimension
        person_features = decoded.squeeze(0)  # [M, D]
        
        # Existence prediction
        existence_logits = self.existence_head(person_features)  # [M]
        
        # Assignment predictions (one per joint type)
        assignment_scores: List[torch.Tensor] = []
        joint_indices_per_type: List[torch.Tensor] = []
        
        for joint_type in range(self.config.num_joint_types):
            # Find all joints of this type
            type_mask = (joint_types == joint_type)
            type_indices = torch.where(type_mask)[0]
            
            if len(type_indices) > 0:
                # Get embeddings for this joint type
                type_embeddings = joint_embeddings[type_indices]  # [K, D]
                
                # Compute assignment scores
                scores = self.assignment_heads[joint_type](
                    person_features, type_embeddings
                )  # [M, K]
            else:
                # No joints of this type - empty tensor
                scores = torch.empty(M, 0, device=device)
            
            assignment_scores.append(scores)
            joint_indices_per_type.append(type_indices)
        
        return {
            'existence_logits': existence_logits,
            'assignment_scores': assignment_scores,
            'joint_indices_per_type': joint_indices_per_type,
            'person_features': person_features
        }

    def predict(
        self,
        joint_embeddings: torch.Tensor,
        joint_types: torch.Tensor,
        existence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Inference: get predicted person poses with EXCLUSIVE assignment.
        
        Uses greedy assignment: highest-confidence query gets first pick,
        then that joint is removed from consideration.
        """
        outputs = self.forward(joint_embeddings, joint_types)
        
        # Get existence probabilities
        existence_probs = torch.sigmoid(outputs['existence_logits'])  # [M]
        person_mask = existence_probs > existence_threshold
        num_people = person_mask.sum().item()
        
        M = self.config.max_people
        device = joint_embeddings.device
        assignments = torch.full((M, 17), -1, dtype=torch.long, device=device)
        
        # Sort queries by existence probability (highest first)
        sorted_queries = torch.argsort(existence_probs, descending=True)
        
        for joint_type, (scores, indices) in enumerate(
            zip(outputs['assignment_scores'], outputs['joint_indices_per_type'])
        ):
            if len(indices) == 0:
                continue
            
            # Track which joints have been assigned
            assigned_mask = torch.zeros(len(indices), dtype=torch.bool, device=device)
            
            # Assign in order of query confidence (greedy)
            for query_idx in sorted_queries:
                if not person_mask[query_idx]:
                    continue
                
                # Get scores for this query, mask out already-assigned joints
                query_scores = scores[query_idx].clone()  # [K]
                query_scores[assigned_mask] = float('-inf')
                
                if query_scores.max() == float('-inf'):
                    continue  # No joints left
                
                # Pick best available joint
                best_local = query_scores.argmax().item()
                best_global = indices[best_local].item()
                
                assignments[query_idx, joint_type] = best_global
                assigned_mask[best_local] = True  # Mark as taken
        
        return {
            'num_people': num_people,
            'existence_probs': existence_probs,
            'person_mask': person_mask,
            'assignments': assignments
        }

    def predict_old(
        self,
        joint_embeddings: torch.Tensor,
        joint_types: torch.Tensor,
        existence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Inference: get predicted person poses.
        
        Args:
            joint_embeddings: [N, D] embeddings from GAT
            joint_types: [N] joint type indices
            existence_threshold: Threshold for person existence
        
        Returns:
            dict with:
                - num_people: Number of detected people
                - person_mask: [M] bool mask of valid people
                - assignments: [M, 17] joint index assigned to each person/type
                               (-1 if no joint of that type exists)
        """
        outputs = self.forward(joint_embeddings, joint_types)
        
        # Get existence probabilities
        existence_probs = torch.sigmoid(outputs['existence_logits'])  # [M]
        person_mask = existence_probs > existence_threshold
        num_people = person_mask.sum().item()
        
        # Get assignments for each person and joint type
        M = self.config.max_people
        assignments = torch.full((M, 17), -1, dtype=torch.long, device=joint_embeddings.device)
        
        for joint_type, (scores, indices) in enumerate(
            zip(outputs['assignment_scores'], outputs['joint_indices_per_type'])
        ):
            if len(indices) > 0:
                # Argmax over joints of this type
                best_joint_local = scores.argmax(dim=1)  # [M]
                # Map back to global joint indices
                best_joint_global = indices[best_joint_local]
                assignments[:, joint_type] = best_joint_global
        
        return {
            'num_people': num_people,
            'existence_probs': existence_probs,
            'person_mask': person_mask,
            'assignments': assignments
        }


class PoseGroupingModel(nn.Module):
    """
    Full model: GAT encoder + DETR decoder.
    
    This combines the GAT embedding network with the DETR decoder
    for end-to-end pose grouping.
    """
    
    def __init__(self, gat_model: nn.Module, detr_config: Optional[DETRConfig] = None):
        super().__init__()
        self.gat = gat_model
        self.detr = DETRDecoder(detr_config)
    
    def forward(self, data: Any) -> Dict[str, Any]:
        """
        Full forward pass.
        
        Args:
            data: PyG Data object with x, joint_types, edge_index
        
        Returns:
            DETR outputs dict
        """
        # Get GAT embeddings
        embeddings = self.gat(data)  # [N, D]
        
        # Run DETR decoder
        outputs = self.detr(embeddings, data.joint_types)
        
        # Also return embeddings for contrastive loss
        outputs['embeddings'] = embeddings
        
        return outputs
    
    def predict(self, data: Any, existence_threshold: float = 0.5) -> Dict[str, Any]:
        """Inference with threshold."""
        embeddings = self.gat(data)
        return self.detr.predict(embeddings, data.joint_types, existence_threshold)