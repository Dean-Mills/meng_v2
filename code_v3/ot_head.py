"""
Skeleton-Constrained Optimal Transport (SCOT) grouping head.

Frames pose grouping as a structured assignment problem rather than
clustering. Each person is a template with 17 typed slots (one per
COCO joint type). Keypoints are assigned to person-type slots via
differentiable optimal transport (Sinkhorn), with hard type constraints
that make it structurally impossible to assign a keypoint to the wrong
type slot.

This is fundamentally different from DMoN/slot attention which treat
grouping as clustering and enforce type exclusivity via soft penalties.
Here, a left elbow can only be assigned to a left-elbow slot — the
constraint is in the cost matrix, not the loss.

Args:
    config:        SCOTConfig
    embedding_dim: must match GAT output_dim

Forward:
    embeddings:  [N, D]  L2-normalised joint embeddings from GAT
    k:           int     number of people / clusters
    joint_types: [N]     COCO joint type indices (0-16)

Returns:
    logits:      [N, K]  assignment logits (person-level, after marginalising types)
    T:           [N, K*17] full transport plan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SCOTConfig

NUM_JOINT_TYPES = 17


class SCOTHead(nn.Module):

    def __init__(self, config: SCOTConfig, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sinkhorn_iters = config.sinkhorn_iters
        self.sinkhorn_tau = config.sinkhorn_tau
        d = embedding_dim
        h = config.hidden_dim

        # Node encoder — maps GAT embeddings to assignment space
        self.node_encoder = nn.Sequential(
            nn.Linear(d, h),
            nn.SELU(),
        )

        # Pool of learned person prototype vectors
        # For K people, the first K prototypes are used.
        self.prototype_pool = nn.Parameter(
            torch.randn(config.k_max, h) * 0.5
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        k: int,
        joint_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings:  [N, D] L2-normalised joint embeddings
            k:           number of people
            joint_types: [N] joint type indices 0-16

        Returns:
            logits: [N, K] person-level assignment logits
            T:      [N, K*17] full transport plan (soft assignments)
        """
        n = embeddings.size(0)
        device = embeddings.device

        # ── Encode nodes ───────────────────────────────────────────────
        h = self.node_encoder(embeddings)  # [N, H]

        # ── Select K prototypes ────────────────────────────────────────
        if k <= self.prototype_pool.size(0):
            prototypes = self.prototype_pool[:k]  # [K, H]
        else:
            extra = torch.randn(
                k - self.prototype_pool.size(0),
                self.prototype_pool.size(1),
                device=device,
            ) * 0.5
            prototypes = torch.cat([self.prototype_pool, extra], dim=0)

        # ── Cost matrix [N, K*17] ─────────────────────────────────────
        # Each person k has 17 typed slots. Slot index = k*17 + t.
        # Cost = squared distance between node encoding and prototype.
        # Infinite cost if node type != slot type.

        # Distance: each node to each person prototype [N, K]
        # (all type slots for the same person share the same prototype)
        dist = torch.cdist(h.unsqueeze(0), prototypes.unsqueeze(0),
                           p=2).squeeze(0).pow(2)  # [N, K]

        # Expand to [N, K*17] — repeat each person column 17 times
        cost = dist.repeat_interleave(NUM_JOINT_TYPES, dim=1)  # [N, K*17]

        # ── Type mask: hard constraint ─────────────────────────────────
        # type_mask[i, k*17+t] = True if joint_types[i] == t
        slot_types = torch.arange(
            NUM_JOINT_TYPES, device=device
        ).repeat(k)  # [K*17] — [0,1,...,16, 0,1,...,16, ...]

        type_mask = (joint_types.unsqueeze(1) == slot_types.unsqueeze(0))  # [N, K*17]

        # Set impossible assignments to large cost (Sinkhorn on log scale)
        cost = cost.masked_fill(~type_mask, 1e6)

        # ── Sinkhorn optimal transport ─────────────────────────────────
        T = self._sinkhorn(cost, n, k, type_mask)  # [N, K*17]

        # ── Person-level logits: sum over type slots per person ────────
        # logits[i, k] = sum_t T[i, k*17+t]
        T_reshaped = T.view(n, k, NUM_JOINT_TYPES)  # [N, K, 17]
        logits = T_reshaped.sum(dim=2)  # [N, K]

        # Convert to log-space for cross-entropy compatibility
        logits = (logits + 1e-8).log()

        return logits, T

    def _sinkhorn(
        self,
        cost: torch.Tensor,
        n: int,
        k: int,
        type_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sinkhorn-Knopp iterations for optimal transport.

        Source marginal: each keypoint assigned to exactly one slot (row sum = 1)
        Target marginal: each person-type slot gets at most one keypoint
                        (column sum <= 1, with slack for missing joints)

        Args:
            cost:      [N, K*17] cost matrix (high for impossible assignments)
            n:         number of keypoints
            k:         number of people
            type_mask: [N, K*17] boolean mask of valid assignments

        Returns:
            T: [N, K*17] transport plan (soft assignments)
        """
        n_slots = k * NUM_JOINT_TYPES
        device = cost.device

        # Log-domain Sinkhorn for numerical stability
        # Start from negative cost / temperature
        log_K = -cost / self.sinkhorn_tau  # [N, n_slots]

        # Add a slack column for unmatched slots (handles missing joints)
        # Slack cost is 0 (moderate), so exp(0/tau) = 1
        log_slack = torch.zeros(n, 1, device=device)
        log_K_ext = torch.cat([log_K, log_slack], dim=1)  # [N, n_slots+1]

        # Target marginal: 1 for each real slot, N for slack
        # (slack absorbs all unmatched mass)
        target_marginal = torch.ones(n_slots + 1, device=device)
        target_marginal[-1] = max(n - n_slots, 1)  # slack capacity
        log_v = torch.zeros(n_slots + 1, device=device)

        for _ in range(self.sinkhorn_iters):
            # Row normalisation (each keypoint sums to 1)
            log_u = -torch.logsumexp(log_K_ext + log_v.unsqueeze(0), dim=1)

            # Column normalisation (each slot sums to target marginal)
            log_sum = torch.logsumexp(
                log_K_ext + log_u.unsqueeze(1), dim=0
            )
            log_v = target_marginal.log() - log_sum

        # Final transport plan (drop slack column)
        T = torch.exp(log_K_ext + log_u.unsqueeze(1) + log_v.unsqueeze(0))
        T = T[:, :n_slots]  # [N, K*17]

        return T
