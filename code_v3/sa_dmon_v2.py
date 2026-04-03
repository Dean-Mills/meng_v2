"""
SA-DMoN v2 — Skeleton-Aware DMoN with decoupled graph architecture.

Fixes the signal cancellation problem in SA-DMoN v1 where both the
adjacency matrix and the null model encoded spatial proximity, causing
B_pose = A_spatial - R_spatial to be near-zero noise.

Two independent fixes, each configurable for ablation:

Fix 1: Decoupled graphs (use_feature_adjacency)
    Build the adjacency for modularity from GAT embedding similarity
    (kNN in embedding space) instead of spatial proximity. The spatial
    null model stays unchanged. Now B_pose = A_feature - R_spatial asks:
    "which keypoints are connected by learned feature similarity beyond
    what spatial proximity and anatomy predict?"

    The GAT still runs message passing on the original spatial kNN graph.
    Only the modularity computation switches to the feature graph.

Fix 2: Entropy type loss (use_entropy_type_loss)
    Replace the ReLU threshold penalty ||ReLU(M^T S - 1)||^2 with
    entropy minimisation on per-type per-cluster assignment distributions.
    For each cluster k and type t, normalise the soft assignments of
    type-t keypoints to cluster k into a distribution, then minimise
    its entropy. This says "each cluster-type pair should be dominated
    by one keypoint, not spread across many."

    The ReLU penalty fires based on absolute mass (>1.0), which is
    noisy with softmax assignments. The entropy version fires based on
    concentration, which is more gradient-friendly.

Forward signature is identical to SA-DMoN v1 for drop-in replacement.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SADMoNV2Config

# Reuse the skeleton constants and type affinity from sa_dmon
from sa_dmon import (
    COCO_SKELETON, NUM_JOINT_TYPES, _build_type_affinity, _TYPE_AFFINITY,
)


class SADMoNV2Head(nn.Module):

    def __init__(self, config: SADMoNV2Config, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_feature_adjacency = config.use_feature_adjacency
        self.use_entropy_type_loss = config.use_entropy_type_loss
        self.feature_knn_k = config.feature_knn_k
        d = embedding_dim
        h = config.hidden_dim

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(d, h),
            nn.SELU(),
        )

        # Cluster center pool
        self.center_pool = nn.Parameter(torch.randn(config.k_max, h) * 0.5)

        self.dropout = config.dropout
        self.scale = h ** -0.5

        # Type affinity buffer
        self.register_buffer("type_affinity", _TYPE_AFFINITY.clone())

    def _build_feature_adjacency(
        self, embeddings: torch.Tensor, k: int,
    ) -> torch.Tensor:
        """
        Build adjacency matrix from kNN in embedding space.
        Detached from gradient — the adjacency is a fixed structure
        for this forward pass, not a differentiable quantity.
        """
        n = embeddings.size(0)
        device = embeddings.device
        k_nn = min(self.feature_knn_k, n - 1)

        # Pairwise distances in embedding space
        with torch.no_grad():
            dist = torch.cdist(embeddings, embeddings, p=2)
            dist.fill_diagonal_(float("inf"))
            _, indices = dist.topk(k_nn, dim=1, largest=False)

        # Build symmetric adjacency
        adj = torch.zeros(n, n, device=device)
        source = torch.arange(n, device=device).repeat_interleave(k_nn)
        target = indices.flatten()
        adj[source, target] = 1.0
        adj = (adj + adj.t()).clamp(max=1.0)

        return adj

    def forward(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        k: int,
        positions: torch.Tensor,
        joint_types: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        n = embeddings.size(0)
        device = embeddings.device

        # ── Soft assignments ───────────────────────────────────────────
        h = self.node_encoder(embeddings)

        if k <= self.center_pool.size(0):
            centers = self.center_pool[:k]
        else:
            extra = torch.randn(
                k - self.center_pool.size(0),
                self.center_pool.size(1),
                device=device,
            ) * 0.5
            centers = torch.cat([self.center_pool, extra], dim=0)

        logits = h @ centers.t() * self.scale
        if self.dropout > 0 and self.training:
            logits = F.dropout(logits, self.dropout)
        s = F.softmax(logits, dim=-1)

        # ── Adjacency for modularity ──────────────────────────────────
        if self.use_feature_adjacency:
            # Fix 1: build adjacency from embedding similarity
            adj = self._build_feature_adjacency(embeddings, self.feature_knn_k)
        else:
            # Original: use spatial kNN adjacency from the graph
            adj = torch.zeros(n, n, device=device)
            adj[edge_index[0], edge_index[1]] = 1.0
            adj = (adj + adj.t()).clamp(max=1.0)

        # ── Losses ────────────────────────────────────────────────────
        spectral_loss = self._spectral_loss(s, adj, positions, joint_types)
        ortho_loss = self._ortho_loss(s, k)
        cluster_loss = self._cluster_loss(s, n, k)

        if self.use_entropy_type_loss:
            type_loss = self._entropy_type_loss(s, joint_types)
        else:
            type_loss = self._relu_type_loss(s, joint_types)

        return logits, s, spectral_loss, ortho_loss, cluster_loss, type_loss

    # ── Spectral loss (spatial null model, unchanged) ──────────────────

    def _spectral_loss(
        self,
        s: torch.Tensor,
        adj: torch.Tensor,
        positions: torch.Tensor,
        joint_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Skeleton-aware modularity: -Tr(S^T B_pose S) / (2m)
        B_pose = A - R_norm (A may be spatial or feature-based)
        R_norm is always spatial + skeletal.
        """
        d = adj.sum(dim=1)
        m = d.sum() / 2

        if m < 1e-8:
            return s.sum() * 0

        # Spatial kernel P with median sigma (detached)
        pairwise_dist = torch.cdist(positions, positions, p=2)
        sq_dist = pairwise_dist.pow(2)

        n = positions.size(0)
        triu_idx = torch.triu_indices(n, n, offset=1, device=positions.device)
        sigma = pairwise_dist[triu_idx[0], triu_idx[1]].median().detach()
        sigma = sigma.clamp(min=1e-4)

        P = torch.exp(-sq_dist / (2 * sigma.pow(2)))

        # Anatomical type affinity
        T_ij = self.type_affinity[joint_types][:, joint_types]
        R = T_ij * P

        # Normalise R to match edge mass
        R_sum = R.sum()
        if R_sum < 1e-8:
            st_a_s = s.t() @ adj @ s
            st_d = s.t() @ d.unsqueeze(1)
            null_term = st_d @ st_d.t() / (2 * m)
            return -(st_a_s - null_term).trace() / (2 * m)

        R_norm = R * (2 * m / R_sum)

        st_a_s = s.t() @ adj @ s
        st_r_s = s.t() @ R_norm @ s

        return -(st_a_s - st_r_s).trace() / (2 * m)

    # ── Structural losses (unchanged) ──────────────────────────────────

    def _ortho_loss(self, s: torch.Tensor, k: int) -> torch.Tensor:
        ss = s.t() @ s
        ss_norm = ss / (ss.norm() + 1e-8)
        i_k = torch.eye(k, device=s.device) / (k ** 0.5)
        return (ss_norm - i_k).norm()

    def _cluster_loss(self, s: torch.Tensor, n: int, k: int) -> torch.Tensor:
        cluster_size = s.sum(dim=0)
        return (k ** 0.5) / n * cluster_size.norm() - 1

    # ── Type exclusivity: original ReLU version ────────────────────────

    def _relu_type_loss(
        self, s: torch.Tensor, joint_types: torch.Tensor,
    ) -> torch.Tensor:
        """||ReLU(M^T S - 1)||^2_F — original SA-DMoN v1 penalty."""
        m_matrix = F.one_hot(
            joint_types.long(), num_classes=NUM_JOINT_TYPES,
        ).float()
        type_cluster = m_matrix.t() @ s
        excess = F.relu(type_cluster - 1.0)
        return (excess ** 2).sum()

    # ── Type exclusivity: entropy version (Fix 2) ──────────────────────

    def _entropy_type_loss(
        self, s: torch.Tensor, joint_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Entropy-based type exclusivity.

        For each cluster k and joint type t, compute the distribution
        of type-t keypoints' soft assignments to cluster k. Minimise
        entropy — the assignment should be peaked on one keypoint,
        not spread across many.

        Only computed for type-cluster pairs that have >= 2 keypoints
        of that type (otherwise entropy is trivially 0).
        """
        n, k = s.shape
        device = s.device

        total_entropy = torch.tensor(0.0, device=device)
        count = 0

        for t in range(NUM_JOINT_TYPES):
            type_mask = (joint_types == t)
            n_type = type_mask.sum().item()

            if n_type < 2:
                continue

            # Soft assignments of type-t keypoints to each cluster [n_type, K]
            s_type = s[type_mask]

            for c in range(k):
                # Distribution over type-t keypoints for cluster c
                weights = s_type[:, c]  # [n_type]
                w_sum = weights.sum()

                if w_sum < 1e-8:
                    continue

                # Normalise to a distribution
                p = weights / w_sum  # [n_type]

                # Entropy: -sum(p * log(p))
                entropy = -(p * (p + 1e-8).log()).sum()
                total_entropy = total_entropy + entropy
                count += 1

        if count > 0:
            total_entropy = total_entropy / count

        return total_entropy
