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
    
    # Null tokens: learnable "background" embeddings in memory
    # Gives unused queries something distinct to attend to
    num_null_tokens: int = 8


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


class PersonCountHead(nn.Module):
    """
    Predicts the number of people from the joint embeddings directly.
    
    The count is a property of the INPUT (how many clusters exist in the 
    embeddings), not of the decoded queries. Previous version pooled person
    query features, which all look similar regardless of GT count because
    every query attends to real joints (no background).
    
    This version uses attention-pooling over joint embeddings to aggregate
    the clustering structure, then predicts a count.
    
    Input: [N, D] joint embeddings
    Output: scalar count prediction
    """
    
    def __init__(self, embed_dim: int, max_people: int, dropout: float = 0.1):
        super().__init__()
        self.max_people = max_people
        
        # Learned attention query for pooling joint embeddings
        self.pool_query = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.pool_proj = nn.Linear(embed_dim, embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim // 2),  # +1 for num_joints feature
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1)
        )
    
    def forward(self, joint_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            joint_embeddings: [N, D] embeddings from GAT (already normalized)
        Returns:
            count_pred: scalar predicted count (clamped to [0, max_people])
        """
        N = joint_embeddings.size(0)
        
        # Attention-pool the joint embeddings
        # Query: [1, D], Keys: [N, D]
        keys = self.pool_proj(joint_embeddings)  # [N, D]
        attn_weights = torch.mm(self.pool_query, keys.t())  # [1, N]
        attn_weights = F.softmax(attn_weights / (keys.size(-1) ** 0.5), dim=-1)
        pooled = torch.mm(attn_weights, joint_embeddings)  # [1, D]
        
        # Append num_joints as explicit feature (strong count signal)
        num_joints_feat = torch.tensor([[N / 17.0]], device=joint_embeddings.device)
        pooled = torch.cat([pooled, num_joints_feat], dim=-1)  # [1, D+1]
        
        count = self.mlp(pooled).squeeze()
        count = count.clamp(min=0.0, max=float(self.max_people))
        
        return count


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
    
    Key architectural features:
        1. Learnable null tokens in memory — gives unused queries "background"
           to attend to, so their features become distinct from matched queries.
           This is the critical fix: without null tokens, every query attends to
           real joints and the existence head can't discriminate.
        2. Count head reads joint embeddings directly (not query features)
        3. Orthogonal query initialization
        4. Intermediate layer auxiliary losses
    """
    
    def __init__(self, config: Optional[DETRConfig] = None):
        super().__init__()
        self.config = config if config is not None else DETRConfig()
        c = self.config
        
        # Learnable person queries - orthogonal initialization
        init_queries = torch.zeros(c.max_people, c.embedding_dim)
        nn.init.orthogonal_(init_queries)
        self.person_queries = nn.Parameter(init_queries)
        
        # Learnable NULL tokens appended to memory
        # These act as "background" — unused queries learn to attend here
        # instead of attending to real joints like matched queries do
        self.null_tokens = nn.Parameter(
            torch.randn(c.num_null_tokens, c.embedding_dim) * 0.02
        )
        
        # Individual decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=c.embedding_dim,
                nhead=c.num_heads,
                dim_feedforward=c.ffn_dim,
                dropout=c.dropout,
                activation='relu',
                batch_first=True
            )
            for _ in range(c.num_decoder_layers)
        ])
        
        # Existence head
        self.existence_head = PersonExistenceHead(c.embedding_dim, c.dropout)
        
        # Count head - reads from joint embeddings directly (not query features)
        self.count_head = PersonCountHead(c.embedding_dim, c.max_people, c.dropout)
        
        # One assignment head per joint type
        self.assignment_heads = nn.ModuleList([
            JointAssignmentHead(c.embedding_dim) 
            for _ in range(c.num_joint_types)
        ])
        
        # Layer norms
        self.joint_norm = nn.LayerNorm(c.embedding_dim)
        self.query_norm = nn.LayerNorm(c.embedding_dim)
        self.null_norm = nn.LayerNorm(c.embedding_dim)
    
    def get_query_params(self) -> List[nn.Parameter]:
        """Return person query + null token parameters for separate LR group."""
        return [self.person_queries, self.null_tokens]
    
    def get_non_query_params(self) -> List[nn.Parameter]:
        """Return all parameters except person queries and null tokens."""
        special_ids = {id(self.person_queries), id(self.null_tokens)}
        return [p for p in self.parameters() if id(p) not in special_ids]
    
    def _apply_heads(
        self,
        person_features: torch.Tensor,
        joint_embeddings: torch.Tensor,
        joint_types: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Apply existence and assignment heads to person features.
        Count head uses joint_embeddings directly.
        """
        M = self.config.max_people
        device = joint_embeddings.device
        
        existence_logits = self.existence_head(person_features)
        count_pred = self.count_head(joint_embeddings)
        
        assignment_scores: List[torch.Tensor] = []
        joint_indices_per_type: List[torch.Tensor] = []
        
        for joint_type in range(self.config.num_joint_types):
            type_mask = (joint_types == joint_type)
            type_indices = torch.where(type_mask)[0]
            
            if len(type_indices) > 0:
                type_embeddings = joint_embeddings[type_indices]
                scores = self.assignment_heads[joint_type](
                    person_features, type_embeddings
                )
            else:
                scores = torch.empty(M, 0, device=device)
            
            assignment_scores.append(scores)
            joint_indices_per_type.append(type_indices)
        
        return {
            'existence_logits': existence_logits,
            'count_pred': count_pred,
            'assignment_scores': assignment_scores,
            'joint_indices_per_type': joint_indices_per_type,
            'person_features': person_features
        }
    
    def forward(
        self, 
        joint_embeddings: torch.Tensor,
        joint_types: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Forward pass of DETR decoder.
        
        Memory = [joint_embeddings; null_tokens] so that unused queries
        can learn to attend to null tokens instead of real joints.
        """
        # Normalize inputs
        joint_embeddings = self.joint_norm(joint_embeddings)
        
        # Build memory: real joint embeddings + null tokens
        null_tokens = self.null_norm(self.null_tokens)  # [K_null, D]
        memory_tokens = torch.cat([joint_embeddings, null_tokens], dim=0)  # [N + K_null, D]
        memory = memory_tokens.unsqueeze(0)  # [1, N + K_null, D]
        
        # Prepare person queries
        queries = self.query_norm(self.person_queries).unsqueeze(0)
        
        # Run through decoder layers, collecting intermediate outputs
        auxiliary_outputs = []
        hidden = queries
        
        for i, layer in enumerate(self.decoder_layers):
            hidden = layer(tgt=hidden, memory=memory)
            
            if i < len(self.decoder_layers) - 1:
                intermediate_features = hidden.squeeze(0)
                aux = self._apply_heads(intermediate_features, joint_embeddings, joint_types)
                auxiliary_outputs.append(aux)
        
        # Final layer outputs
        person_features = hidden.squeeze(0)
        final_outputs = self._apply_heads(person_features, joint_embeddings, joint_types)
        final_outputs['auxiliary_outputs'] = auxiliary_outputs
        
        return final_outputs

    def _assignment_confidence(self, outputs: Dict) -> torch.Tensor:
        """
        Rank queries by how peaked their assignment distributions are.
        
        A query that's found a real person cluster has peaked (confident)
        softmax distributions over joints. An unspecialized query has
        diffuse distributions.
        
        Returns: [M] confidence scores (higher = more confident assignments)
        """
        M = self.config.max_people
        device = outputs['existence_logits'].device
        confidences = torch.zeros(M, device=device)
        n_types = 0
        
        for scores, indices in zip(outputs['assignment_scores'], outputs['joint_indices_per_type']):
            if len(indices) == 0:
                continue
            probs = F.softmax(scores, dim=1)  # [M, K]
            confidences += probs.max(dim=1).values  # max prob per query
            n_types += 1
        
        if n_types > 0:
            confidences /= n_types
        
        return confidences
    
    def _count_from_joints(self, joint_types: torch.Tensor) -> int:
        """
        Estimate person count from joint type frequencies.
        
        For each joint type, the number of instances equals the number of
        people (each person has one joint per type). We take the mode
        across types for robustness.
        
        For synthetic data: N_total / 17 (exact).
        For real data: mode of per-type counts.
        """
        counts_per_type = []
        for jtype in range(self.config.num_joint_types):
            n = (joint_types == jtype).sum().item()
            if n > 0:
                counts_per_type.append(n)
        
        if not counts_per_type:
            return 0
        
        # Mode of per-type counts (robust to missing types)
        from collections import Counter
        counter = Counter(counts_per_type)
        return counter.most_common(1)[0][0]
    
    def predict(
        self,
        joint_embeddings: torch.Tensor,
        joint_types: torch.Tensor,
        existence_threshold: float = 0.5,
        use_count_head: bool = False,
        selection_method: str = "confidence",
    ) -> Dict[str, Any]:
        """
        Inference: assign joints to people.
        
        Query selection strategy (from diagnostic analysis):
            1. Count people from joint type frequencies (no learning needed)
            2. Rank queries by assignment confidence (not existence probs)
            3. Keep top-N queries, do per-type Hungarian assignment
        
        Args:
            selection_method: "confidence" (default, best), "existence", or "all"
            use_count_head: if True, use learned count head (legacy, worse)
            existence_threshold: threshold for existence-based selection (legacy)
        """
        from scipy.optimize import linear_sum_assignment
        
        outputs = self.forward(joint_embeddings, joint_types)
        
        existence_probs = torch.sigmoid(outputs['existence_logits'])
        count_pred = outputs.get('count_pred', torch.tensor(0.0))
        confidences = self._assignment_confidence(outputs)
        
        M = self.config.max_people
        device = joint_embeddings.device
        
        # ── Determine person count ───────────────────────────────────
        if use_count_head:
            cp = count_pred.item() if isinstance(count_pred, torch.Tensor) else count_pred
            num_people = int(round(cp))
        else:
            num_people = self._count_from_joints(joint_types)
        
        num_people = max(0, min(num_people, M))
        
        # ── Select active queries ────────────────────────────────────
        if selection_method == "all":
            person_mask = torch.ones(M, dtype=torch.bool, device=device)
            num_people_active = M
        elif num_people == 0:
            person_mask = torch.zeros(M, dtype=torch.bool, device=device)
            num_people_active = 0
        else:
            if selection_method == "confidence":
                _, top_indices = confidences.topk(num_people)
            else:  # "existence"
                _, top_indices = existence_probs.topk(num_people)
            
            person_mask = torch.zeros(M, dtype=torch.bool, device=device)
            person_mask[top_indices] = True
            num_people_active = num_people
        
        # ── Hungarian assignment ─────────────────────────────────────
        assignments = torch.full((M, 17), -1, dtype=torch.long, device=device)
        
        if num_people_active > 0:
            active_query_indices = torch.where(person_mask)[0]
            
            for joint_type, (scores, indices) in enumerate(
                zip(outputs['assignment_scores'], outputs['joint_indices_per_type'])
            ):
                if len(indices) == 0:
                    continue
                
                active_scores = scores[active_query_indices]
                cost_matrix = -active_scores.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                for r, c in zip(row_ind, col_ind):
                    query_idx = active_query_indices[r]
                    global_joint_idx = indices[c]
                    assignments[query_idx, joint_type] = global_joint_idx
        
        return {
            'num_people': num_people,
            'count_pred': count_pred.item() if isinstance(count_pred, torch.Tensor) else count_pred,
            'existence_probs': existence_probs,
            'confidences': confidences,
            'person_mask': person_mask,
            'assignments': assignments,
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
    
    def get_param_groups(self, base_lr: float, query_lr_multiplier: float = 10.0):
        """
        Return parameter groups with separate LR for person queries.
        
        Person queries need a higher LR to diverge quickly from their
        initialization. All other params use the base LR.
        """
        return [
            {'params': list(self.gat.parameters()) + self.detr.get_non_query_params(),
             'lr': base_lr},
            {'params': self.detr.get_query_params(),
             'lr': base_lr * query_lr_multiplier},
        ]
    
    def forward(self, data: Any) -> Dict[str, Any]:
        """
        Full forward pass.
        
        Args:
            data: PyG Data object with x, joint_types, edge_index
        
        Returns:
            DETR outputs dict (includes auxiliary_outputs for intermediate losses)
        """
        # Get GAT embeddings
        embeddings = self.gat(data)  # [N, D]
        
        # Run DETR decoder
        outputs = self.detr(embeddings, data.joint_types)
        
        # Also return embeddings for contrastive loss
        outputs['embeddings'] = embeddings
        
        return outputs
    
    def predict(self, data: Any, existence_threshold: float = 0.5, 
                use_count_head: bool = False, selection_method: str = "confidence") -> Dict[str, Any]:
        """Inference with confidence-based selection (default) or legacy methods."""
        embeddings = self.gat(data)
        return self.detr.predict(embeddings, data.joint_types, existence_threshold, 
                                use_count_head, selection_method)