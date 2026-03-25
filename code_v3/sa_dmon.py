"""
SA-DMoN (Skeleton-Aware Deep Modularity Networks) grouping head.

Three modifications over vanilla DMoN:
1. Skeleton-aware null model — replaces the degree-based null dd⊤/(2m)
   with a domain-specific null combining spatial proximity and anatomical
   type affinity. Changes what the modularity objective measures: from
   "connected more than degree predicts" to "connected more than spatial
   proximity and anatomy predict."
2. Type exclusivity regularization — penalizes clusters with duplicate
   joint types. Encodes the constraint that a person has at most one
   keypoint of each type.
3. Supervised-modularity hybrid loss — Hungarian-matched CE alongside
   modularity (handled in losses.py, not here).

The skeleton-aware null model:

    P_{i,j} = exp(-||pos_i - pos_j||^2 / (2σ^2))     spatial kernel
    T_{a,b}                                            anatomical affinity (17x17)
    R_{i,j} = T[type(i), type(j)] * P_{i,j}           combined null
    R_norm  = R * (2m / sum(R))                        normalized to match edge mass

    B_pose  = A - R_norm

The modularity loss then maximizes Tr(S^T B_pose S) / (2m), which rewards
grouping joints that are connected *beyond* what spatial proximity and
anatomical adjacency would explain.

Args:
    config:        SADMoNConfig
    embedding_dim: must match GAT output_dim

Forward:
    embeddings:  [N, D]  L2-normalised joint embeddings from GAT
    edge_index:  [2, E]  sparse edge indices from PyG graph
    k:           int     number of people / clusters
    positions:   [N, 2]  normalised 2D positions (x, y) for spatial kernel
    joint_types: [N]     COCO joint type indices (0-16), required

Returns:
    logits:        [N, K]  assignment logits (argmax for hard assignment)
    s:             [N, K]  soft assignment matrix
    spectral_loss: scalar  skeleton-aware modularity loss
    ortho_loss:    scalar  orthogonality regularization
    cluster_loss:  scalar  collapse prevention
    type_loss:     scalar  type exclusivity
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SADMoNConfig


# COCO skeleton — pairs of joint indices that are anatomically connected.
# 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear,
# 5=left_shoulder, 6=right_shoulder, 7=left_elbow, 8=right_elbow,
# 9=left_wrist, 10=right_wrist, 11=left_hip, 12=right_hip,
# 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 7), (7, 9), (6, 8), (8, 10),        # arms
    (5, 6), (5, 11), (6, 12), (11, 12),     # torso
    (11, 13), (13, 15), (12, 14), (14, 16), # legs
]

NUM_JOINT_TYPES = 17


def _build_type_affinity() -> torch.Tensor:
    """
    Build the 17x17 anatomical type affinity matrix T from COCO skeleton.

    T[a, b] encodes the prior expected proximity between joint types a and b.
    - Directly connected in skeleton: 1.0
    - Two hops apart: 0.5
    - Three hops apart: 0.25
    - Further / unconnected: 0.0

    This captures that a left_shoulder near a left_elbow is expected
    (directly connected), while a left_shoulder near a right_ankle at
    the same distance is surprising (no anatomical connection).
    """
    T = torch.zeros(NUM_JOINT_TYPES, NUM_JOINT_TYPES)

    # Build adjacency for BFS
    adj = [[] for _ in range(NUM_JOINT_TYPES)]
    for a, b in COCO_SKELETON:
        adj[a].append(b)
        adj[b].append(a)

    # BFS from each joint to compute hop distances
    for src in range(NUM_JOINT_TYPES):
        dist = [-1] * NUM_JOINT_TYPES
        dist[src] = 0
        queue = [src]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)

        for dst in range(NUM_JOINT_TYPES):
            d = dist[dst]
            if d == 1:
                T[src, dst] = 1.0
            elif d == 2:
                T[src, dst] = 0.5
            elif d == 3:
                T[src, dst] = 0.25
            # d == 0 (self) or d > 3 or unreachable: 0.0

    return T


# Precompute once at module load — this is a constant
_TYPE_AFFINITY = _build_type_affinity()


class SADMoNHead(nn.Module):

    def __init__(self, config: SADMoNConfig, embedding_dim: int):
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
        self.center_pool = nn.Parameter(torch.randn(config.k_max, h) * 0.5)

        self.dropout = config.dropout
        self.scale = h ** -0.5

        # Spatial kernel bandwidth — learnable log(σ), clamped to [sigma_min, sigma_max]
        self.log_sigma = nn.Parameter(torch.tensor(config.sigma_init).log())
        self.log_sigma_min = torch.tensor(config.sigma_min).log().item()
        self.log_sigma_max = torch.tensor(config.sigma_max).log().item()

        # Register the type affinity matrix as a buffer (moved to device with model)
        self.register_buffer("type_affinity", _TYPE_AFFINITY.clone())

    @property
    def sigma(self) -> torch.Tensor:
        clamped = self.log_sigma.clamp(min=self.log_sigma_min, max=self.log_sigma_max)
        return clamped.exp()

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
        """
        Args:
            embeddings:  [N, D] L2-normalised joint embeddings from GAT
            edge_index:  [2, E] sparse edge indices
            k:           number of people / clusters
            positions:   [N, 2] normalised 2D positions (x_norm, y_norm)
            joint_types: [N] joint type indices 0-16

        Returns:
            logits:        [N, K]
            s:             [N, K]  soft assignments
            spectral_loss: scalar  skeleton-aware modularity
            ortho_loss:    scalar
            cluster_loss:  scalar
            type_loss:     scalar
        """
        n = embeddings.size(0)
        device = embeddings.device

        # ── Soft assignments ───────────────────────────────────────────────
        h = self.node_encoder(embeddings)  # [N, H]

        if k <= self.center_pool.size(0):
            centers = self.center_pool[:k]  # [K, H]
        else:
            extra = torch.randn(
                k - self.center_pool.size(0),
                self.center_pool.size(1),
                device=device,
            ) * 0.5
            centers = torch.cat([self.center_pool, extra], dim=0)

        logits = h @ centers.t() * self.scale  # [N, K]
        if self.dropout > 0 and self.training:
            logits = F.dropout(logits, self.dropout)
        s = F.softmax(logits, dim=-1)  # [N, K]

        # ── Dense adjacency ────────────────────────────────────────────────
        adj = torch.zeros(n, n, device=device)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj = (adj + adj.t()).clamp(max=1.0)  # undirected

        # ── Losses ─────────────────────────────────────────────────────────
        spectral_loss = self._spectral_loss(s, adj, positions, joint_types)
        ortho_loss = self._ortho_loss(s, k)
        cluster_loss = self._cluster_loss(s, n, k)
        type_loss = self._type_loss(s, joint_types)

        return logits, s, spectral_loss, ortho_loss, cluster_loss, type_loss

    # ── Skeleton-aware spectral loss ───────────────────────────────────────

    def _spectral_loss(
        self,
        s: torch.Tensor,
        adj: torch.Tensor,
        positions: torch.Tensor,
        joint_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Skeleton-aware modularity loss: -Tr(S^T B_pose S) / (2m)

        B_pose = A - R_norm

        where R combines spatial proximity and anatomical type affinity,
        normalized to match the total edge mass of the graph.
        """
        d = adj.sum(dim=1)  # [N] degree vector
        m = d.sum() / 2     # total edge mass

        if m < 1e-8:
            return s.sum() * 0

        # ── Spatial proximity kernel P ─────────────────────────────────
        # P_{i,j} = exp(-||pos_i - pos_j||^2 / (2σ^2))
        sq_dist = torch.cdist(positions, positions, p=2).pow(2)  # [N, N]
        sigma_sq = self.sigma.pow(2)
        P = torch.exp(-sq_dist / (2 * sigma_sq))  # [N, N]

        # ── Anatomical type affinity R ─────────────────────────────────
        # R_{i,j} = T[type(i), type(j)] * P_{i,j}
        T_ij = self.type_affinity[joint_types][:, joint_types]  # [N, N]
        R = T_ij * P  # [N, N]

        # ── Normalize R to match edge mass ─────────────────────────────
        R_sum = R.sum()
        if R_sum < 1e-8:
            # Fall back to degree null if R is degenerate
            st_a_s = s.t() @ adj @ s
            st_d = s.t() @ d.unsqueeze(1)
            null_term = st_d @ st_d.t() / (2 * m)
            return -(st_a_s - null_term).trace() / (2 * m)

        R_norm = R * (2 * m / R_sum)  # [N, N]

        # ── B_pose = A - R_norm ────────────────────────────────────────
        # Tr(S^T B_pose S) = Tr(S^T A S) - Tr(S^T R_norm S)
        st_a_s = s.t() @ adj @ s          # [K, K]
        st_r_s = s.t() @ R_norm @ s       # [K, K]

        spectral_loss = -(st_a_s - st_r_s).trace() / (2 * m)

        return spectral_loss

    # ── Standard DMoN structural losses (unchanged) ────────────────────────

    def _ortho_loss(self, s: torch.Tensor, k: int) -> torch.Tensor:
        """
        Orthogonality regularization:
        ||S^T S / ||S^T S||_F - I_K / sqrt(K)||_F
        """
        ss = s.t() @ s
        ss_norm = ss / (ss.norm() + 1e-8)
        i_k = torch.eye(k, device=s.device) / (k ** 0.5)
        return (ss_norm - i_k).norm()

    def _cluster_loss(
        self, s: torch.Tensor, n: int, k: int,
    ) -> torch.Tensor:
        """
        Cluster collapse prevention:
        sqrt(K) / N * ||cluster_sizes|| - 1
        """
        cluster_size = s.sum(dim=0)
        return (k ** 0.5) / n * cluster_size.norm() - 1

    # ── Type exclusivity (unchanged from vanilla) ──────────────────────────

    def _type_loss(
        self, s: torch.Tensor, joint_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Type exclusivity regularization:
        ||ReLU(M^T S - 1)||^2_F
        """
        m_matrix = F.one_hot(
            joint_types.long(), num_classes=NUM_JOINT_TYPES,
        ).float()

        type_cluster = m_matrix.t() @ s
        excess = F.relu(type_cluster - 1.0)

        return (excess ** 2).sum()
