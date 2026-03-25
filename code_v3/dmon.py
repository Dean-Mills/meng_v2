"""
DMoN (Deep Modularity Networks) grouping head with pose-specific modifications.

Three modifications over vanilla DMoN:
1. Type exclusivity regularization — penalizes clusters with duplicate joint types
2. Supervised-modularity hybrid loss — Hungarian-matched CE alongside modularity
3. (Future) Skeleton-aware spatial null model — deferred until 1+2 are evaluated

Standard DMoN uses a fixed-K MLP for cluster assignments. This uses a pool of
k_max learned cluster center vectors, selecting the first K per scene. This
handles variable K across scenes while keeping assignments deterministic.
Hungarian matching in the loss makes the assignment permutation-invariant.

Args:
    config:        DMoNConfig
    embedding_dim: must match GAT output_dim

Forward:
    embeddings:  [N, D]  L2-normalised joint embeddings from GAT
    edge_index:  [2, E]  sparse edge indices from PyG graph
    k:           int     number of people / clusters
    joint_types: [N]     optional, COCO joint type indices (0-16) for type
                         exclusivity regularization

Returns:
    logits:        [N, K]  assignment logits (argmax for hard assignment)
    s:             [N, K]  soft assignment matrix
    spectral_loss: scalar  modularity loss
    ortho_loss:    scalar  orthogonality regularization
    cluster_loss:  scalar  collapse prevention
    type_loss:     scalar  type exclusivity (0 if joint_types not provided)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DMoNConfig


class DMoNHead(nn.Module):

    NUM_JOINT_TYPES = 17  # COCO keypoint types

    def __init__(self, config: DMoNConfig, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        d = embedding_dim
        h = config.hidden_dim

        # Node encoder — maps GAT embeddings to assignment space
        self.node_encoder = nn.Sequential(
            nn.Linear(d, h),
            nn.SELU(),
        )

        # Pool of learned cluster center vectors
        # For a scene with K people, the first K centers are used.
        # Hungarian matching in the loss makes this permutation-invariant.
        self.center_pool = nn.Parameter(torch.randn(config.k_max, h) * 0.5)

        self.dropout = config.dropout
        self.scale = h ** -0.5

    def forward(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        k: int,
        joint_types: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        """
        Args:
            embeddings: [N, D] L2-normalised joint embeddings from GAT
            edge_index: [2, E] sparse edge indices
            k:          number of people / clusters
            joint_types: [N] optional joint type indices 0-16

        Returns:
            logits:        [N, K]
            s:             [N, K]  soft assignments
            spectral_loss: scalar
            ortho_loss:    scalar
            cluster_loss:  scalar
            type_loss:     scalar
        """
        n = embeddings.size(0)
        device = embeddings.device

        # Encode nodes
        h = self.node_encoder(embeddings)  # [N, H]

        # Select K cluster centers from the learned pool
        # If k > k_max, pad with random centers for overflow
        if k <= self.center_pool.size(0):
            centers = self.center_pool[:k]  # [K, H]
        else:
            extra = torch.randn(
                k - self.center_pool.size(0),
                self.center_pool.size(1),
                device=device,
            ) * 0.5
            centers = torch.cat([self.center_pool, extra], dim=0)  # [K, H]

        # Soft assignment via scaled dot product
        logits = h @ centers.t() * self.scale  # [N, K]
        if self.dropout > 0 and self.training:
            logits = F.dropout(logits, self.dropout)
        s = F.softmax(logits, dim=-1)  # [N, K]

        # Dense adjacency from sparse edge_index (symmetrized for modularity)
        adj = torch.zeros(n, n, device=device)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj = (adj + adj.t()).clamp(max=1.0)  # undirected

        # DMoN losses
        spectral_loss = self._spectral_loss(s, adj)
        ortho_loss = self._ortho_loss(s, k)
        cluster_loss = self._cluster_loss(s, n, k)

        # Type exclusivity loss (modification 1)
        if joint_types is not None:
            type_loss = self._type_loss(s, joint_types)
        else:
            type_loss = s.sum() * 0  # zero but in graph

        return logits, s, spectral_loss, ortho_loss, cluster_loss, type_loss

    # ── DMoN structural losses ────────────────────────────────────────────────

    def _spectral_loss(
        self, s: torch.Tensor, adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Spectral modularity loss: -Tr(S^T B S) / (2m)
        where B = A - dd^T/(2m) is the modularity matrix.

        Maximizing modularity = minimizing this loss (it's negated).
        """
        d = adj.sum(dim=1)  # [N] degree vector
        m = d.sum() / 2     # total edge mass

        if m < 1e-8:
            return s.sum() * 0  # zero but in graph

        # S^T A S  [K, K]
        st_a_s = s.t() @ adj @ s

        # S^T d  [K, 1] — cluster degree sums
        st_d = s.t() @ d.unsqueeze(1)

        # Null model: (S^T d)(S^T d)^T / (2m)
        null_term = st_d @ st_d.t() / (2 * m)

        # Tr(S^T B S) = Tr(S^T A S - null)
        spectral_loss = -(st_a_s - null_term).trace() / (2 * m)

        return spectral_loss

    def _ortho_loss(self, s: torch.Tensor, k: int) -> torch.Tensor:
        """
        Orthogonality regularization:
        ||S^T S / ||S^T S||_F - I_K / sqrt(K)||_F

        Encourages cluster assignments to be quasi-orthogonal.
        """
        ss = s.t() @ s  # [K, K]
        ss_norm = ss / (ss.norm() + 1e-8)
        i_k = torch.eye(k, device=s.device) / (k ** 0.5)
        return (ss_norm - i_k).norm()

    def _cluster_loss(
        self, s: torch.Tensor, n: int, k: int,
    ) -> torch.Tensor:
        """
        Cluster collapse prevention:
        sqrt(K) / N * ||cluster_sizes|| - 1

        Equals 0 when clusters are balanced, > 0 when imbalanced.
        """
        cluster_size = s.sum(dim=0)  # [K]
        return (k ** 0.5) / n * cluster_size.norm() - 1

    # ── Pose-specific losses ──────────────────────────────────────────────────

    def _type_loss(
        self, s: torch.Tensor, joint_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Type exclusivity regularization (modification 1):
        ||ReLU(M^T S - 1)||^2_F

        M[i,t] = 1 if joint i has type t. M^T S gives the total soft
        assignment mass per type per cluster. For a valid pose, each entry
        should be <= 1 (at most one joint of each type per person).

        Penalizes any cluster that accumulates more than one joint of
        the same type.
        """
        # Type indicator matrix M [N, 17]
        m_matrix = F.one_hot(
            joint_types.long(), num_classes=self.NUM_JOINT_TYPES,
        ).float()

        # Total assignment mass per type per cluster [17, K]
        type_cluster = m_matrix.t() @ s

        # Penalize entries > 1
        excess = F.relu(type_cluster - 1.0)

        return (excess ** 2).sum()
