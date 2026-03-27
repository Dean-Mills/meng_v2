"""
Residual SCOT — Skeleton-Constrained Optimal Transport with spatial prior.

Extends SCOT by decomposing the cost matrix into two components:
    C_final = C_spatial + lambda_residual * C_learned

C_spatial: based on 2D position proximity to k-means centroids.
           Encodes geometric proximity — equivalent to what kNN uses.
           Dominates on easy scenes where people are well-separated.

C_learned: based on GAT embedding distance to learned prototypes.
           Encodes identity signal from contrastive learning.
           Provides discriminative correction on hard scenes where
           spatial proximity is ambiguous.

The spatial cost anchors easy cases (preventing OT from introducing
noise on problems kNN already solves), while the learned cost handles
hard cases (overlapping people where spatial proximity is insufficient).

Same type constraints as original SCOT — hard masking in cost matrix.

Args:
    config:        ResidualSCOTConfig
    embedding_dim: must match GAT output_dim

Forward:
    embeddings:  [N, D]  L2-normalised joint embeddings from GAT
    k:           int     number of people / clusters
    positions:   [N, 2]  normalised 2D positions (x_norm, y_norm)
    joint_types: [N]     COCO joint type indices (0-16)

Returns:
    logits:      [N, K]  assignment logits (person-level)
    T:           [N, K*17] full transport plan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ResidualSCOTConfig

NUM_JOINT_TYPES = 17


class ResidualSCOTHead(nn.Module):

    def __init__(self, config: ResidualSCOTConfig, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sinkhorn_iters = config.sinkhorn_iters
        self.sinkhorn_tau = config.sinkhorn_tau
        self.lambda_residual = config.lambda_residual
        d = embedding_dim
        h = config.hidden_dim

        # Node encoder — maps GAT embeddings to assignment space
        self.node_encoder = nn.Sequential(
            nn.Linear(d, h),
            nn.SELU(),
        )

        # Pool of learned person prototype vectors (embedding space)
        self.prototype_pool = nn.Parameter(
            torch.randn(config.k_max, h) * 0.5
        )

    def _compute_spatial_centroids(
        self, positions: torch.Tensor, k: int,
    ) -> torch.Tensor:
        """
        Compute K spatial centroids via k-means on 2D positions.
        Detached — spatial cost is a fixed prior, not learned.

        Args:
            positions: [N, 2]
            k:         number of clusters

        Returns:
            centroids: [K, 2]
        """
        n = positions.size(0)
        device = positions.device

        # k-means++ initialisation
        centroids = [positions[torch.randint(n, (1,), device=device).item()]]
        for _ in range(1, k):
            dists = torch.stack([
                (positions - c.unsqueeze(0)).pow(2).sum(dim=1)
                for c in centroids
            ], dim=1).min(dim=1).values  # [N]
            probs = dists / (dists.sum() + 1e-8)
            idx = torch.multinomial(probs, 1).item()
            centroids.append(positions[idx])
        centroids = torch.stack(centroids)  # [K, 2]

        # Run a few Lloyd iterations to refine
        for _ in range(10):
            dists = torch.cdist(positions, centroids, p=2)  # [N, K]
            assignments = dists.argmin(dim=1)  # [N]
            new_centroids = torch.zeros_like(centroids)
            for c in range(k):
                mask = assignments == c
                if mask.sum() > 0:
                    new_centroids[c] = positions[mask].mean(dim=0)
                else:
                    new_centroids[c] = centroids[c]
            centroids = new_centroids

        return centroids.detach()

    def forward(
        self,
        embeddings: torch.Tensor,
        k: int,
        positions: torch.Tensor,
        joint_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings:  [N, D] L2-normalised joint embeddings
            k:           number of people
            positions:   [N, 2] normalised 2D positions
            joint_types: [N] joint type indices 0-16

        Returns:
            logits: [N, K] person-level assignment logits
            T:      [N, K*17] full transport plan
        """
        n = embeddings.size(0)
        device = embeddings.device

        # ── Encode nodes ───────────────────────────────────────────────
        h = self.node_encoder(embeddings)  # [N, H]

        # ── Select K prototypes (embedding space) ──────────────────────
        if k <= self.prototype_pool.size(0):
            prototypes = self.prototype_pool[:k]  # [K, H]
        else:
            extra = torch.randn(
                k - self.prototype_pool.size(0),
                self.prototype_pool.size(1),
                device=device,
            ) * 0.5
            prototypes = torch.cat([self.prototype_pool, extra], dim=0)

        # ── C_learned: embedding distance to prototypes [N, K] ────────
        c_learned = torch.cdist(
            h.unsqueeze(0), prototypes.unsqueeze(0), p=2,
        ).squeeze(0).pow(2)  # [N, K]

        # ── C_spatial: position distance to k-means centroids [N, K] ──
        centroids = self._compute_spatial_centroids(positions, k)  # [K, 2]
        c_spatial = torch.cdist(
            positions.unsqueeze(0), centroids.unsqueeze(0), p=2,
        ).squeeze(0).pow(2)  # [N, K]

        # ── Combined cost ──────────────────────────────────────────────
        # C_final = C_spatial + lambda * C_learned
        cost_per_person = c_spatial + self.lambda_residual * c_learned  # [N, K]

        # Expand to [N, K*17]
        cost = cost_per_person.repeat_interleave(NUM_JOINT_TYPES, dim=1)

        # ── Type mask: hard constraint ─────────────────────────────────
        slot_types = torch.arange(
            NUM_JOINT_TYPES, device=device,
        ).repeat(k)

        type_mask = (joint_types.unsqueeze(1) == slot_types.unsqueeze(0))
        cost = cost.masked_fill(~type_mask, 1e6)

        # ── Sinkhorn ───────────────────────────────────────────────────
        T = self._sinkhorn(cost, n, k)  # [N, K*17]

        # ── Person-level logits ────────────────────────────────────────
        T_reshaped = T.view(n, k, NUM_JOINT_TYPES)
        logits = T_reshaped.sum(dim=2)  # [N, K]
        logits = (logits + 1e-8).log()

        return logits, T

    def _sinkhorn(
        self,
        cost: torch.Tensor,
        n: int,
        k: int,
    ) -> torch.Tensor:
        """
        Sinkhorn-Knopp iterations for optimal transport.
        Same as original SCOT — log-domain with slack column.
        """
        n_slots = k * NUM_JOINT_TYPES
        device = cost.device

        log_K = -cost / self.sinkhorn_tau
        log_slack = torch.zeros(n, 1, device=device)
        log_K_ext = torch.cat([log_K, log_slack], dim=1)

        target_marginal = torch.ones(n_slots + 1, device=device)
        target_marginal[-1] = max(n - n_slots, 1)
        log_v = torch.zeros(n_slots + 1, device=device)

        for _ in range(self.sinkhorn_iters):
            log_u = -torch.logsumexp(log_K_ext + log_v.unsqueeze(0), dim=1)
            log_sum = torch.logsumexp(
                log_K_ext + log_u.unsqueeze(1), dim=0,
            )
            log_v = target_marginal.log() - log_sum

        T = torch.exp(log_K_ext + log_u.unsqueeze(1) + log_v.unsqueeze(0))
        T = T[:, :n_slots]

        return T
