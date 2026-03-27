"""
Unbalanced SCOT — Skeleton-Constrained OT without requiring known K.

Extends SCOT with unbalanced optimal transport so the number of people
emerges from the solution rather than being an input. Uses a generous
K_max (e.g. 30) person templates, and unbalanced Sinkhorn with KL
relaxation on the target marginal allows unused person slots to
naturally receive zero assignment mass.

The key difference from vanilla SCOT:
  - Vanilla: K is provided (ground truth at train, must be known at inference)
  - Unbalanced: K_max templates created, KL penalty allows empty slots,
    actual person count = number of slots with mass above threshold

Unbalanced Sinkhorn adds a KL divergence penalty on the target marginal:
    min_T  Σ T_{ij} C_{ij} + ε·H(T) + ρ·KL(T^T·1 | c)

With finite ρ, person slots without good matches gracefully empty out.
Standard Sinkhorn is recovered as ρ → ∞.

Same type constraints as original SCOT — hard masking in cost matrix.

Args:
    config:        UnbalancedSCOTConfig
    embedding_dim: must match GAT output_dim

Forward:
    embeddings:  [N, D]  L2-normalised joint embeddings from GAT
    joint_types: [N]     COCO joint type indices (0-16)
    k:           int|None  if provided, used as-is (training with GT K);
                           if None, uses k_max and infers K from solution

Returns:
    logits:      [N, K_eff]  assignment logits (person-level, active persons only)
    T:           [N, K_max*17] full transport plan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import UnbalancedSCOTConfig

NUM_JOINT_TYPES = 17


class UnbalancedSCOTHead(nn.Module):

    def __init__(self, config: UnbalancedSCOTConfig, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sinkhorn_iters = config.sinkhorn_iters
        self.sinkhorn_tau = config.sinkhorn_tau
        self.rho = config.rho
        self.k_max = config.k_max
        self.person_threshold = config.person_threshold
        d = embedding_dim
        h = config.hidden_dim

        # Node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(d, h),
            nn.SELU(),
        )

        # Full pool of K_max person prototypes
        self.prototype_pool = nn.Parameter(
            torch.randn(config.k_max, h) * 0.5
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        joint_types: torch.Tensor,
        k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings:  [N, D] L2-normalised joint embeddings
            joint_types: [N] joint type indices 0-16
            k:           number of people (if known, e.g. during training).
                         If None, uses k_max and infers from solution.

        Returns:
            logits: [N, K_eff] person-level assignment logits
            T:      [N, K_used*17] full transport plan
        """
        n = embeddings.size(0)
        device = embeddings.device

        # Use provided K or fall back to k_max
        k_used = k if k is not None else self.k_max

        # ── Encode nodes ───────────────────────────────────────────────
        h = self.node_encoder(embeddings)  # [N, H]

        # ── Select prototypes ──────────────────────────────────────────
        if k_used <= self.prototype_pool.size(0):
            prototypes = self.prototype_pool[:k_used]
        else:
            extra = torch.randn(
                k_used - self.prototype_pool.size(0),
                self.prototype_pool.size(1),
                device=device,
            ) * 0.5
            prototypes = torch.cat([self.prototype_pool, extra], dim=0)

        # ── Cost matrix [N, K_used*17] ─────────────────────────────────
        dist = torch.cdist(
            h.unsqueeze(0), prototypes.unsqueeze(0), p=2,
        ).squeeze(0).pow(2)  # [N, K_used]

        cost = dist.repeat_interleave(NUM_JOINT_TYPES, dim=1)  # [N, K_used*17]

        # ── Type mask ──────────────────────────────────────────────────
        slot_types = torch.arange(
            NUM_JOINT_TYPES, device=device,
        ).repeat(k_used)

        type_mask = (joint_types.unsqueeze(1) == slot_types.unsqueeze(0))
        cost = cost.masked_fill(~type_mask, 1e6)

        # ── Unbalanced Sinkhorn ────────────────────────────────────────
        T = self._unbalanced_sinkhorn(cost, n, k_used)  # [N, K_used*17]

        # ── Person-level logits ────────────────────────────────────────
        T_reshaped = T.view(n, k_used, NUM_JOINT_TYPES)  # [N, K_used, 17]
        person_mass = T_reshaped.sum(dim=2)  # [N, K_used] — mass per person per keypoint

        # Determine active persons (total mass above threshold)
        total_person_mass = T_reshaped.sum(dim=(0, 2))  # [K_used]
        active = total_person_mass > self.person_threshold  # [K_used]

        if active.sum() == 0:
            # Fallback: keep at least one person
            active[total_person_mass.argmax()] = True

        # If K was provided (training), use all K slots for logits
        # If K was inferred, filter to active persons
        if k is not None:
            logits = (person_mass + 1e-8).log()  # [N, K]
        else:
            logits = (person_mass[:, active] + 1e-8).log()  # [N, K_eff]

        return logits, T

    def _unbalanced_sinkhorn(
        self,
        cost: torch.Tensor,
        n: int,
        k: int,
    ) -> torch.Tensor:
        """
        Unbalanced Sinkhorn-Knopp in log domain.

        Standard Sinkhorn enforces exact marginals. Unbalanced Sinkhorn
        relaxes the target marginal via KL penalty with weight rho.
        Finite rho allows target slots to receive less mass than their
        marginal — unused person slots naturally empty out.

        The modification vs standard Sinkhorn is in the column update:
            standard:    log_v = log(c) - logsumexp(...)
            unbalanced:  log_v = rho/(rho+tau) * (log(c) - logsumexp(...))

        The scaling factor rho/(rho+tau) < 1 dampens the column
        normalisation, allowing columns to deviate from their target.
        As rho → ∞, the factor → 1 and we recover standard Sinkhorn.
        """
        n_slots = k * NUM_JOINT_TYPES
        device = cost.device
        tau = self.sinkhorn_tau
        rho = self.rho

        # KL relaxation factor
        kl_factor = rho / (rho + tau)

        log_K = -cost / tau  # [N, n_slots]

        # Target marginal: 1 for each slot
        log_c = torch.zeros(n_slots, device=device)  # log(1) = 0

        log_v = torch.zeros(n_slots, device=device)

        for _ in range(self.sinkhorn_iters):
            # Row normalisation (exact — each keypoint fully assigned)
            log_u = -torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)  # [N]

            # Column normalisation (relaxed via KL penalty)
            log_sum = torch.logsumexp(
                log_K + log_u.unsqueeze(1), dim=0,
            )  # [n_slots]
            log_v = kl_factor * (log_c - log_sum)  # [n_slots]

        T = torch.exp(log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0))  # [N, n_slots]

        return T
