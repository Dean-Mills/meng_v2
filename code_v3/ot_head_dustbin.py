"""
Dustbin SCOT — Skeleton-Constrained OT with dustbin for K-free inference.

Uses K_max person templates with standard balanced Sinkhorn, plus an
explicit dustbin column that absorbs keypoints from unused person slots.
The dustbin cost τ controls how easily keypoints get rejected — if a
keypoint's best person assignment has cost > τ, it goes to the dustbin.

After solving, person slots with total assignment mass above a threshold
are considered active. The number of people emerges from the solution.

Key difference from unbalanced SCOT:
  - Unbalanced: relaxes target marginals via KL penalty (noisy, hard to tune)
  - Dustbin: keeps exact balanced Sinkhorn (sharp assignments) with an
    explicit competing "no person" option

During training, GT K is provided and only K prototypes are used (same
as vanilla SCOT). During inference with unknown K, K_max prototypes are
used and the dustbin absorbs unused slots.

Args:
    config:        DustbinSCOTConfig
    embedding_dim: must match GAT output_dim

Forward:
    embeddings:  [N, D]  L2-normalised joint embeddings from GAT
    joint_types: [N]     COCO joint type indices (0-16)
    k:           int|None  if provided, used as-is; if None, uses k_max

Returns:
    logits:      [N, K_eff]  assignment logits (active persons only)
    T:           [N, K_used*17+1] full transport plan including dustbin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DustbinSCOTConfig

NUM_JOINT_TYPES = 17


class DustbinSCOTHead(nn.Module):

    def __init__(self, config: DustbinSCOTConfig, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.sinkhorn_iters = config.sinkhorn_iters
        self.sinkhorn_tau = config.sinkhorn_tau
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

        # Learned dustbin cost — how expensive it is to reject a keypoint
        # Higher = harder to reject (more keypoints assigned to persons)
        # Lower = easier to reject (more keypoints go to dustbin)
        self.dustbin_cost = nn.Parameter(torch.tensor(config.dustbin_cost_init))

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
            k:           number of people (if known). None = use k_max.

        Returns:
            logits: [N, K_eff] person-level assignment logits
            T:      [N, K_used*17+1] full transport plan with dustbin
        """
        n = embeddings.size(0)
        device = embeddings.device

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

        # ── Add dustbin column ─────────────────────────────────────────
        # Dustbin cost is the same for all keypoints — learned scalar
        dustbin_col = self.dustbin_cost.abs().expand(n, 1)  # [N, 1]
        cost_ext = torch.cat([cost, dustbin_col], dim=1)  # [N, K_used*17 + 1]

        # ── Sinkhorn with dustbin ──────────────────────────────────────
        T = self._sinkhorn_dustbin(cost_ext, n, k_used)  # [N, K_used*17 + 1]

        # ── Person-level logits (exclude dustbin) ──────────────────────
        T_persons = T[:, :-1]  # [N, K_used*17]
        T_reshaped = T_persons.view(n, k_used, NUM_JOINT_TYPES)  # [N, K_used, 17]
        person_mass = T_reshaped.sum(dim=2)  # [N, K_used]

        # Determine active persons
        total_person_mass = T_reshaped.sum(dim=(0, 2))  # [K_used]
        active = total_person_mass > self.person_threshold

        if active.sum() == 0:
            active[total_person_mass.argmax()] = True

        if k is not None:
            # Training: use all K slots
            logits = (person_mass + 1e-8).log()
        else:
            # Inference: only active persons
            logits = (person_mass[:, active] + 1e-8).log()

        return logits, T

    def _sinkhorn_dustbin(
        self,
        cost: torch.Tensor,
        n: int,
        k: int,
    ) -> torch.Tensor:
        """
        Standard balanced Sinkhorn with dustbin column.

        The dustbin column competes with person slots. Keypoints whose
        best person assignment costs more than the dustbin cost will
        be absorbed by the dustbin instead.

        Target marginal:
          - Each person-type slot: 1 (at most one keypoint)
          - Dustbin: n (can absorb all keypoints if needed)
        """
        n_slots = k * NUM_JOINT_TYPES
        n_cols = n_slots + 1  # +1 for dustbin
        device = cost.device

        log_K = -cost / self.sinkhorn_tau  # [N, n_cols]

        # Target marginal: 1 per person slot, n for dustbin
        target_marginal = torch.ones(n_cols, device=device)
        target_marginal[-1] = n  # dustbin can absorb everything

        log_v = torch.zeros(n_cols, device=device)

        for _ in range(self.sinkhorn_iters):
            # Row normalisation (each keypoint fully assigned)
            log_u = -torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)

            # Column normalisation (each slot matches target marginal)
            log_sum = torch.logsumexp(
                log_K + log_u.unsqueeze(1), dim=0,
            )
            log_v = target_marginal.log() - log_sum

        T = torch.exp(log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0))

        return T  # [N, n_cols] — includes dustbin
