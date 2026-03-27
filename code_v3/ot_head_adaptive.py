"""
Adaptive SCOT — Skeleton-Constrained OT with scene-adaptive temperature.

Extends SCOT by predicting the Sinkhorn temperature (entropy regularisation)
from the scene rather than fixing it. On easy scenes where the cost matrix
is peaked, the model learns to output low tau (sharp, kNN-like assignments).
On hard scenes with ambiguous costs, it outputs higher tau (soft, letting
OT do global optimisation).

    tau = sigmoid(MLP(graph_summary)) * (tau_max - tau_min) + tau_min

where graph_summary is computed from cost matrix statistics (mean, std,
entropy of the row-wise softmin) and mean-pooled GAT embeddings.

Same type constraints as original SCOT — hard masking in cost matrix.

Args:
    config:        AdaptiveSCOTConfig
    embedding_dim: must match GAT output_dim

Forward:
    embeddings:  [N, D]  L2-normalised joint embeddings from GAT
    k:           int     number of people / clusters
    joint_types: [N]     COCO joint type indices (0-16)

Returns:
    logits:      [N, K]  assignment logits (person-level)
    T:           [N, K*17] full transport plan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AdaptiveSCOTConfig

NUM_JOINT_TYPES = 17


class AdaptiveSCOTHead(nn.Module):

    def __init__(self, config: AdaptiveSCOTConfig, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sinkhorn_iters = config.sinkhorn_iters
        self.tau_min = config.tau_min
        self.tau_max = config.tau_max
        d = embedding_dim
        h = config.hidden_dim

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(d, h),
            nn.SELU(),
        )

        # Person prototype pool
        self.prototype_pool = nn.Parameter(
            torch.randn(config.k_max, h) * 0.5
        )

        # Tau predictor: takes cost matrix stats + pooled embeddings → scalar
        # Features: cost_mean, cost_std, cost_entropy, emb_mean (D dims)
        tau_input_dim = 3 + d
        self.tau_predictor = nn.Sequential(
            nn.Linear(tau_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _predict_tau(
        self,
        cost: torch.Tensor,
        type_mask: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict scene-adaptive temperature from cost matrix statistics
        and global embedding summary.

        Args:
            cost:       [N, K*17] cost matrix (before masking infinities)
            type_mask:  [N, K*17] valid assignment mask
            embeddings: [N, D] raw GAT embeddings

        Returns:
            tau: scalar in [tau_min, tau_max]
        """
        # Cost statistics over valid entries only
        valid_costs = cost[type_mask]
        cost_mean = valid_costs.mean().unsqueeze(0)
        cost_std = valid_costs.std().unsqueeze(0) if valid_costs.numel() > 1 else torch.zeros(1, device=cost.device)

        # Row-wise entropy of softmin over valid slots (how peaked is each row)
        # For each keypoint, softmin over its valid slots gives assignment confidence
        masked_cost = cost.clone()
        masked_cost[~type_mask] = 1e6
        row_probs = F.softmax(-masked_cost, dim=1)  # [N, K*17]
        row_entropy = -(row_probs * (row_probs + 1e-8).log()).sum(dim=1)  # [N]
        cost_entropy = row_entropy.mean().unsqueeze(0)

        # Mean-pooled embeddings
        emb_mean = embeddings.mean(dim=0)  # [D]

        # Concatenate features
        features = torch.cat([cost_mean, cost_std, cost_entropy, emb_mean])  # [3 + D]

        # Predict tau in [tau_min, tau_max]
        raw = self.tau_predictor(features.unsqueeze(0)).squeeze()  # scalar
        tau = torch.sigmoid(raw) * (self.tau_max - self.tau_min) + self.tau_min

        return tau

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
            T:      [N, K*17] full transport plan
        """
        n = embeddings.size(0)
        device = embeddings.device

        # ── Encode nodes ───────────────────────────────────────────────
        h = self.node_encoder(embeddings)  # [N, H]

        # ── Select K prototypes ────────────────────────────────────────
        if k <= self.prototype_pool.size(0):
            prototypes = self.prototype_pool[:k]
        else:
            extra = torch.randn(
                k - self.prototype_pool.size(0),
                self.prototype_pool.size(1),
                device=device,
            ) * 0.5
            prototypes = torch.cat([self.prototype_pool, extra], dim=0)

        # ── Cost matrix [N, K*17] ─────────────────────────────────────
        dist = torch.cdist(
            h.unsqueeze(0), prototypes.unsqueeze(0), p=2,
        ).squeeze(0).pow(2)  # [N, K]

        cost = dist.repeat_interleave(NUM_JOINT_TYPES, dim=1)  # [N, K*17]

        # ── Type mask ──────────────────────────────────────────────────
        slot_types = torch.arange(
            NUM_JOINT_TYPES, device=device,
        ).repeat(k)

        type_mask = (joint_types.unsqueeze(1) == slot_types.unsqueeze(0))

        # ── Predict adaptive tau ───────────────────────────────────────
        tau = self._predict_tau(cost, type_mask, embeddings)

        # ── Apply type mask ────────────────────────────────────────────
        cost = cost.masked_fill(~type_mask, 1e6)

        # ── Sinkhorn with adaptive tau ─────────────────────────────────
        T = self._sinkhorn(cost, n, k, tau)

        # ── Person-level logits ────────────────────────────────────────
        T_reshaped = T.view(n, k, NUM_JOINT_TYPES)
        logits = T_reshaped.sum(dim=2)
        logits = (logits + 1e-8).log()

        return logits, T

    def _sinkhorn(
        self,
        cost: torch.Tensor,
        n: int,
        k: int,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sinkhorn with adaptive temperature.
        Same structure as original SCOT but tau is predicted per scene.
        """
        n_slots = k * NUM_JOINT_TYPES
        device = cost.device

        log_K = -cost / tau
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
