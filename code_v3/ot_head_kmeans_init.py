"""
SCOT with k-means initialised prototypes (SCOT-KI).

Same Sinkhorn OT assignment and type masking as SCOT, but replaces the
fixed learned prototype pool with scene-adaptive prototypes computed
from k-means on the current scene's embeddings.

This gives SCOT the same scene-adaptive initialisation as COP-Kmeans,
plus global optimal assignment via Sinkhorn rather than greedy
one-at-a-time assignment, plus hard type constraints via infinite cost.

No learned prototype parameters — the node encoder is the only learned
component. This means the head can be used without training if the
node encoder is initialised well, or trained on any data without
worrying about prototype diversity.
"""
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from config import SCOTConfig

NUM_JOINT_TYPES = 17


class SCOTKmeansInitHead(nn.Module):

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

        # No prototype_pool — prototypes come from k-means at inference

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

        # ── Scene-adaptive prototypes from k-means ─────────────────────
        h_np = h.detach().cpu().numpy()
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(h_np)
        prototypes = torch.tensor(
            km.cluster_centers_, dtype=h.dtype, device=device
        )  # [K, H]

        # ── Cost matrix [N, K*17] ─────────────────────────────────────
        dist = torch.cdist(
            h.unsqueeze(0), prototypes.unsqueeze(0), p=2
        ).squeeze(0).pow(2)  # [N, K]

        # Expand to [N, K*17]
        cost = dist.repeat_interleave(NUM_JOINT_TYPES, dim=1)

        # ── Type mask: hard constraint ─────────────────────────────────
        slot_types = torch.arange(
            NUM_JOINT_TYPES, device=device
        ).repeat(k)

        type_mask = (joint_types.unsqueeze(1) == slot_types.unsqueeze(0))
        cost = cost.masked_fill(~type_mask, 1e6)

        # ── Sinkhorn optimal transport ─────────────────────────────────
        T = self._sinkhorn(cost, n, k, type_mask)

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
        type_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sinkhorn-Knopp in log domain. Same as original SCOT."""
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
                log_K_ext + log_u.unsqueeze(1), dim=0
            )
            log_v = target_marginal.log() - log_sum

        T = torch.exp(log_K_ext + log_u.unsqueeze(1) + log_v.unsqueeze(0))
        T = T[:, :n_slots]

        return T
