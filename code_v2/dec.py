import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DECConfig


class DEC(nn.Module):
    """
    Deep Embedded Clustering.

    Takes GAT embeddings and assigns each joint to one of K clusters
    using a soft assignment based on distance to learned cluster centres.

    At training time K comes from ground truth person count.
    At inference time K must be provided explicitly.

    The target distribution P is updated every `update_interval` steps rather
    than every step. Updating P every step causes cluster collapse — the model
    chases a moving target and eventually all joints pile into one cluster.
    Periodic updates let the centres stabilise between target refreshes.

    Args:
        config:           DECConfig
        embedding_dim:    int  must match GAT output_dim
        update_interval:  int  how many steps between target distribution updates
    """

    def __init__(self, config: DECConfig, embedding_dim: int, update_interval: int = 100):
        super().__init__()
        self.alpha           = config.alpha
        self.embedding_dim   = embedding_dim
        self.update_interval = update_interval

        self._current_k: int = 0
        self._step: int      = 0
        self._p: torch.Tensor | None = None   # cached target distribution

        self.cluster_centres: nn.Parameter = nn.Parameter(
            torch.empty(0, embedding_dim)
        )

    def initialise_centres(self, embeddings: torch.Tensor, k: int) -> None:
        """
        Initialise cluster centres using k-means++ on the provided embeddings.
        Also resets the step counter and clears the cached target distribution.
        Call this once before training on each new graph/scene.

        Args:
            embeddings: [N, D]
            k:          number of clusters
        """
        centres = self._kmeans_plusplus(embeddings.detach(), k)
        self.cluster_centres = nn.Parameter(centres)
        self._current_k = k
        self._step      = 0
        self._p         = None

    @torch.no_grad()
    def _kmeans_plusplus(self, embeddings: torch.Tensor, k: int) -> torch.Tensor:
        """
        k-means++ initialisation — spreads initial centres out rather than
        picking randomly, which leads to faster and more stable convergence.
        """
        n = embeddings.size(0)

        idx = int(torch.randint(n, (1,)).item())
        centres = [embeddings[idx]]

        for _ in range(k - 1):
            stacked   = torch.stack(centres)
            dists     = torch.cdist(embeddings, stacked)
            min_dists = dists.min(dim=1).values
            probs     = (min_dists ** 2)
            probs     = probs / probs.sum()
            idx       = int(torch.multinomial(probs, 1).item())
            centres.append(embeddings[idx])

        return torch.stack(centres)   # [K, D]

    def soft_assignment(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute soft assignment Q using Student's t-distribution.

        q_ij = (1 + ||z_i - μ_j||² / α)^(-(α+1)/2)
               ─────────────────────────────────────
               Σ_j (1 + ||z_i - μ_j||² / α)^(-(α+1)/2)

        Args:
            embeddings: [N, D]

        Returns:
            q: [N, K]
        """
        sq_dists  = torch.cdist(embeddings, self.cluster_centres) ** 2
        numerator = (1.0 + sq_dists / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        q         = numerator / numerator.sum(dim=1, keepdim=True)
        return q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """
        Compute sharpened target distribution P from soft assignments Q.

        p_ij = q_ij² / f_j
               ──────────────
               Σ_j q_ij² / f_j

        where f_j = Σ_i q_ij  (soft cluster frequency)

        P is sharper than Q — it pushes the model toward more confident assignments.

        Args:
            q: [N, K]

        Returns:
            p: [N, K]
        """
        f         = q.sum(dim=0, keepdim=True)
        numerator = (q ** 2) / f
        p         = numerator / numerator.sum(dim=1, keepdim=True)
        return p

    def forward(
        self,
        embeddings: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: [N, D]  GAT embeddings
            k:          int     number of people in this scene

        Returns:
            q: [N, K]  soft assignments
            p: [N, K]  cached target distribution (updated every update_interval steps)
        """
        if k != self._current_k or self.cluster_centres.shape[0] == 0:
            self.initialise_centres(embeddings, k)

        q = self.soft_assignment(embeddings)

        # Update target distribution periodically to prevent cluster collapse
        if self._p is None or self._step % self.update_interval == 0:
            with torch.no_grad():
                self._p = self.target_distribution(q)

        self._step += 1

        assert self._p is not None
        return q, self._p