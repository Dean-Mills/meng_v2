import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SlotAttentionConfig


class SlotAttention(nn.Module):
    """
    Slot Attention grouping head.

    Takes GAT embeddings and assigns each joint to one of K person slots
    via iterative competitive attention. Slots compete for joints — if two
    slots both want the same joint, the one that matches it better wins.
    This natural competition causes slots to specialise onto different people.

    Unlike DEC this is trained end-to-end with ground truth labels via
    Hungarian-matched cross entropy, so it actually learns from the data.

    Args:
        config:        SlotAttentionConfig
        embedding_dim: must match GAT output_dim

    Forward:
        embeddings:    [N, D]  L2-normalised joint embeddings from GAT
        k:             int     number of people in this scene (from person_labels at train time)

    Returns:
        logits:  [N, K]  unnormalised assignment scores per joint per slot
        slots:   [K, D]  final slot vectors after iteration
    """

    def __init__(self, config: SlotAttentionConfig, embedding_dim: int):
        super().__init__()
        self.num_iterations = config.num_iterations
        self.embedding_dim  = embedding_dim
        d = embedding_dim

        # Slots are initialised by sampling from a learned Gaussian
        self.slot_mu    = nn.Parameter(torch.randn(1, 1, d))
        self.slot_sigma = nn.Parameter(torch.ones(1, 1, d))

        # LayerNorms
        self.norm_inputs = nn.LayerNorm(d)
        self.norm_slots  = nn.LayerNorm(d)
        self.norm_ff     = nn.LayerNorm(d)

        # Attention projections
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)

        # Slot update GRU — slots carry state across iterations
        self.gru = nn.GRUCell(d, d)

        # Feed-forward refinement after each update
        self.ff = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.ReLU(),
            nn.Linear(d * 2, d),
        )

        # Final projection from slot vectors to per-joint assignment logits
        self.to_logits = nn.Linear(d, d, bias=False)

        self.scale = d ** -0.5

    def _init_slots(self, k: int, device: torch.device) -> torch.Tensor:
        """
        Sample K slot vectors from the learned Gaussian.
        Shape: [K, D]
        """
        mu    = self.slot_mu.expand(1, k, -1)       # [1, K, D]
        sigma = self.slot_sigma.expand(1, k, -1)    # [1, K, D]
        slots = mu + sigma * torch.randn_like(mu)   # [1, K, D]
        return slots.squeeze(0)                      # [K, D]

    def forward(
        self,
        embeddings: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: [N, D]
            k:          number of people / slots

        Returns:
            logits: [N, K]
            slots:  [K, D]
        """
        n, d   = embeddings.shape
        device = embeddings.device

        # Normalise inputs once
        inputs = self.norm_inputs(embeddings)   # [N, D]
        k_proj = self.to_k(inputs)              # [N, D]
        v_proj = self.to_v(inputs)              # [N, D]

        # Initialise slots
        slots = self._init_slots(k, device)     # [K, D]

        for _ in range(self.num_iterations):
            slots_prev = slots
            slots_n    = self.norm_slots(slots)     # [K, D]

            # Attention: slots query, joints are keys/values
            q = self.to_q(slots_n)                  # [K, D]

            # Dot product attention scores [K, N]
            attn = torch.einsum("kd,nd->kn", q, k_proj) * self.scale

            # Softmax over SLOTS (not joints) — this is the competition
            # Each joint's attention sums to 1 across all slots
            attn = F.softmax(attn, dim=0)           # [K, N]

            # Normalise within each slot (weighted mean)
            attn_norm = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)  # [K, N]

            # Weighted sum of values per slot
            updates = torch.einsum("kn,nd->kd", attn_norm, v_proj)     # [K, D]

            # Update slots via GRU
            slots = self.gru(
                updates.reshape(k, d),
                slots_prev.reshape(k, d),
            )                                       # [K, D]

            # Feed-forward refinement
            slots = slots + self.ff(self.norm_ff(slots))

        # Project slots back to embedding space for dot-product scoring
        slot_keys = self.to_logits(slots)           # [K, D]

        # Per-joint logits: similarity of each joint to each slot
        logits = torch.einsum("nd,kd->nk", embeddings, slot_keys)  # [N, K]

        return logits, slots