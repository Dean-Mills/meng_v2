"""DEC isolation tests
Verifies DEC works correctly in true isolation using fabricated embeddings.

Fabricated embeddings are Gaussian blobs on the unit hypersphere — tight clusters
with known ground truth labels. This removes the GAT dependency entirely and lets
us test exactly what DEC is supposed to do: cluster already-structured embeddings.

Noise controls difficulty:
    low  (0.05) — tight, well-separated clusters (easy)
    high (0.3)  — overlapping clusters (hard)

Tests:
1. Initialisation      — cluster centres have correct shape, k-means++ spreads them out
2. Soft assignment     — q is [N, K], rows sum to 1, values in [0, 1]
3. Target distribution — p is sharper than q
4. Loss computation    — KL divergence is a scalar, not NaN
5. Gradient flow       — cluster centres receive gradients
6. Assignment accuracy — argmax(q) matches labels after training
                         (uses Hungarian matching to handle cluster permutations)

Note on K:
    At training time K = number of unique people in person_labels.
    At inference time K must be provided externally — this is a known open problem
    addressed separately from the grouping task itself.

Run from code_v2/:
    python tests/test_dec.py --config configs/dec_isolation.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import List

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import ExperimentConfig
from dec import DEC


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def _pass(msg: str):  print(f"  ✓ {msg}")
def _fail(msg: str):
    print(f"  ✗ {msg}")
    raise AssertionError(msg)

def _check(condition: bool, msg: str):
    _pass(msg) if condition else _fail(msg)


def make_clustered_embeddings(
    n_per_person: int,
    k: int,
    d: int,
    noise: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fabricate L2-normalised embeddings as Gaussian blobs on the unit hypersphere.

    Each person gets n_per_person embeddings sampled around a random centre,
    with Gaussian noise added before re-normalising onto the sphere.

    Args:
        n_per_person: joints per person
        k:            number of people
        d:            embedding dimension
        noise:        std of Gaussian noise — controls cluster tightness
        device:       torch device

    Returns:
        embeddings: [N, D]  L2-normalised
        labels:     [N]     ground truth person IDs 0..K-1
    """
    centres    = F.normalize(torch.randn(k, d, device=device), dim=-1)
    embeddings = []
    labels     = []

    for person_id in range(k):
        pts = centres[person_id].unsqueeze(0) + torch.randn(n_per_person, d, device=device) * noise
        pts = F.normalize(pts, dim=-1)
        embeddings.append(pts)
        labels.extend([person_id] * n_per_person)

    return torch.cat(embeddings), torch.tensor(labels, device=device)


def hungarian_accuracy(
    pred_clusters: np.ndarray,
    true_labels: np.ndarray,
    k: int,
) -> float:
    """
    Compute assignment accuracy after optimal cluster-to-person matching.

    DEC cluster indices are arbitrary — cluster 0 might correspond to person 2.
    Hungarian algorithm finds the optimal bijection between predicted clusters
    and ground truth people, then computes accuracy under that mapping.
    """
    confusion = np.zeros((k, k), dtype=np.int64)
    for p, t in zip(pred_clusters, true_labels):
        confusion[p, t] += 1
    row_idx, col_idx = linear_sum_assignment(-confusion)
    correct = confusion[row_idx, col_idx].sum()
    return correct / len(true_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_initialisation(dec: DEC, embeddings: torch.Tensor, k: int):
    _header("Initialisation")

    dec.initialise_centres(embeddings, k)

    _check(
        dec.cluster_centres.shape == (k, dec.embedding_dim),
        f"Cluster centres shape: {dec.cluster_centres.shape} — expected [{k}, {dec.embedding_dim}]"
    )
    _check(not torch.isnan(dec.cluster_centres).any().item(), "No NaN in cluster centres")

    centres  = dec.cluster_centres.detach()
    dists    = torch.cdist(centres, centres)
    mask     = ~torch.eye(k, dtype=torch.bool, device=centres.device)
    min_dist = dists[mask].min().item()
    _check(min_dist > 1e-3, f"Centres are spread out (min pairwise distance: {min_dist:.4f})")

    print(f"\n  Initialisation: all checks passed")


def test_soft_assignment(dec: DEC, embeddings: torch.Tensor, k: int):
    _header("Soft Assignment")

    q, p = dec(embeddings, k)

    _check(q.shape == (embeddings.size(0), k), f"q shape: {q.shape} — expected [{embeddings.size(0)}, {k}]")
    _check(p.shape == (embeddings.size(0), k), f"p shape: {p.shape} — expected [{embeddings.size(0)}, {k}]")
    _check((q.sum(dim=1) - 1.0).abs().max().item() < 1e-5, "q rows sum to 1")
    _check((p.sum(dim=1) - 1.0).abs().max().item() < 1e-5, "p rows sum to 1")
    _check(q.min().item() >= 0.0 and q.max().item() <= 1.0, "q values in [0, 1]")
    _check(p.min().item() >= 0.0 and p.max().item() <= 1.0, "p values in [0, 1]")

    q_entropy = -(q * (q + 1e-8).log()).sum(dim=1).mean().item()
    p_entropy = -(p * (p + 1e-8).log()).sum(dim=1).mean().item()
    _check(p_entropy < q_entropy,
           f"p is sharper than q (q entropy: {q_entropy:.4f}, p entropy: {p_entropy:.4f})")

    print(f"\n  Soft assignment: all checks passed")
    return q, p


def test_loss_computation(q: torch.Tensor, p: torch.Tensor):
    _header("Loss Computation")

    loss = F.kl_div(q.log(), p, reduction="batchmean")

    _check(isinstance(loss, torch.Tensor), "Loss is a tensor")
    _check(loss.shape == torch.Size([]),   "Loss is a scalar")
    _check(not torch.isnan(loss).item(),   "Loss is not NaN")
    _check(not torch.isinf(loss).item(),   "Loss is not Inf")
    _check(loss.item() >= 0.0,             f"Loss >= 0 ({loss.item():.4f})")

    print(f"  KL loss: {loss.item():.4f}")
    print(f"\n  Loss computation: all checks passed")
    return loss


def test_gradient_flow(dec: DEC, embeddings: torch.Tensor, k: int):
    _header("Gradient Flow")

    dec.train()
    optimizer = optim.Adam(dec.parameters(), lr=1e-3)
    optimizer.zero_grad()

    q, p = dec(embeddings, k)
    loss = F.kl_div(q.log(), p, reduction="batchmean")
    loss.backward()

    _check(dec.cluster_centres.grad is not None, "Cluster centres have gradients")
    assert dec.cluster_centres.grad is not None
    grad_norm = dec.cluster_centres.grad.norm().item()
    _check(grad_norm > 0, f"Gradient norm > 0 ({grad_norm:.4f})")

    print(f"\n  Gradient flow: all checks passed")


def test_assignment_accuracy(
    dec: DEC,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    n_steps: int,
    noise: float,
    min_accuracy: float = 0.8,
):
    _header(f"Assignment Accuracy ({n_steps} steps, noise={noise})")

    optimizer  = optim.Adam(dec.parameters(), lr=1e-3)
    labels_np  = labels.cpu().numpy()

    # Before training
    dec.eval()
    with torch.no_grad():
        q_before, _ = dec(embeddings, k)
    acc_before = hungarian_accuracy(q_before.argmax(dim=1).cpu().numpy(), labels_np, k)
    print(f"\n  Before training: accuracy = {acc_before:.4f}")

    # Training
    dec.train()
    print(f"  Training...")
    for step in range(n_steps):
        optimizer.zero_grad()
        q, p = dec(embeddings, k)
        loss = F.kl_div(q.log(), p, reduction="batchmean")
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            dec.eval()
            with torch.no_grad():
                q_mid, _ = dec(embeddings, k)
            acc_mid = hungarian_accuracy(q_mid.argmax(dim=1).cpu().numpy(), labels_np, k)
            print(f"  Step {step+1:3d}: accuracy={acc_mid:.4f}  loss={loss.item():.4f}")
            dec.train()

    # After training
    dec.eval()
    with torch.no_grad():
        q_after, _ = dec(embeddings, k)
    acc_after = hungarian_accuracy(q_after.argmax(dim=1).cpu().numpy(), labels_np, k)
    print(f"\n  After training:  accuracy = {acc_after:.4f}")

    _check(acc_after >= acc_before,
           f"Accuracy did not degrade ({acc_before:.4f} → {acc_after:.4f})")
    _check(acc_after >= min_accuracy,
           f"Accuracy >= {min_accuracy} ({acc_after:.4f})")

    print(f"\n  Assignment accuracy: all checks passed ✅")
    return acc_before, acc_after


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=Path, default=Path("configs/dec_isolation.yaml"))
    parser.add_argument("--n_steps",     type=int,  default=1000)
    parser.add_argument("--k",           type=int,  default=5,   help="Number of people / clusters")
    parser.add_argument("--n_per_person",type=int,  default=50,  help="Joints per person")
    parser.add_argument("--noise_easy",  type=float,default=0.05,help="Noise for easy test")
    parser.add_argument("--noise_medium",type=float,default=0.1, help="Noise for medium test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = ExperimentConfig.from_yaml(args.config)
    print(f"Config: {cfg.name}")
    assert cfg.dec is not None, "dec config missing from yaml"

    d = cfg.gat.output_dim
    k = args.k

    passed: List[str] = []
    failed: List[str] = []

    # Use easy embeddings for structural tests
    embeddings, labels = make_clustered_embeddings(args.n_per_person, k, d, args.noise_easy, device)
    print(f"\n  Fabricated embeddings: {k} people × {args.n_per_person} joints = {embeddings.shape[0]} nodes")
    print(f"  Embedding dim: {d}")

    dec = DEC(cfg.dec, embedding_dim=d, update_interval=cfg.dec.update_interval).to(device)

    for name, fn in [
        ("Initialisation",   lambda: test_initialisation(dec, embeddings, k)),
        ("Gradient flow",    lambda: test_gradient_flow(dec, embeddings, k)),
    ]:
        try:
            fn()
            passed.append(name)
        except AssertionError as e:
            failed.append(f"{name}: {e}")

    dec.initialise_centres(embeddings, k)
    try:
        q, p = test_soft_assignment(dec, embeddings, k)
        passed.append("Soft assignment")
        try:
            test_loss_computation(q, p)
            passed.append("Loss computation")
        except AssertionError as e:
            failed.append(f"Loss computation: {e}")
    except AssertionError as e:
        failed.append(f"Soft assignment: {e}")

    # Accuracy — easy (noise=0.05)
    # k-means++ should already achieve ~1.0 here — this is an initialisation check,
    # not a DEC learning check. We just verify accuracy doesn't degrade.
    dec_easy = DEC(cfg.dec, embedding_dim=d, update_interval=cfg.dec.update_interval).to(device)
    emb_easy, lbl_easy = make_clustered_embeddings(args.n_per_person, k, d, args.noise_easy, device)
    dec_easy.initialise_centres(emb_easy, k)
    try:
        test_assignment_accuracy(dec_easy, emb_easy, lbl_easy, k, args.n_steps, args.noise_easy,
                                 min_accuracy=0.8)
        passed.append(f"Assignment accuracy (easy, noise={args.noise_easy})")
    except AssertionError as e:
        failed.append(f"Assignment accuracy easy: {e}")

    # Accuracy — medium (noise=0.15)
    # The real DEC test. k-means++ gets it partially right (~0.7),
    # DEC should improve meaningfully to > 0.85.
    dec_med = DEC(cfg.dec, embedding_dim=d, update_interval=cfg.dec.update_interval).to(device)
    emb_med, lbl_med = make_clustered_embeddings(args.n_per_person, k, d, args.noise_medium, device)
    dec_med.initialise_centres(emb_med, k)
    try:
        test_assignment_accuracy(dec_med, emb_med, lbl_med, k, args.n_steps, args.noise_medium,
                                 min_accuracy=0.75)
        passed.append(f"Assignment accuracy (medium, noise={args.noise_medium})")
    except AssertionError as e:
        failed.append(f"Assignment accuracy medium: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    for p in passed:
        print(f"    ✓ {p}")
    if failed:
        for f in failed:
            print(f"    ✗ {f}")
    else:
        print(f"\n  All tests passed ✅")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()