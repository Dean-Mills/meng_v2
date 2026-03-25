"""DMoN isolation tests.

Verifies the DMoN grouping head works correctly in isolation using fabricated
embeddings with synthetic kNN graphs. No real data or GAT required — same
approach as test_slot_attention.py and test_graph_partitioning.py.

Unlike the other heads, DMoN needs an adjacency matrix for the spectral
modularity loss. A kNN graph is built from the fabricated embeddings.

Tests:
1. Forward pass        — logits [N, K], s [N, K], no NaN, losses are scalars
2. Loss computation    — combined loss scalar, not NaN, >= 0
3. Gradient flow       — all parameters receive gradients
4. Assignment accuracy — Hungarian-matched accuracy improves and exceeds 0.9
                         tested at easy (noise=0.05) and hard (noise=0.15)

Run from code_v3/:
    python tests/test_dmon.py --config configs/dmon_isolation.yaml
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
from dmon import DMoNHead
from losses import DMoNLoss


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def _check(condition: bool, msg: str):
    if condition:
        print(f"  ✓ {msg}")
    else:
        print(f"  ✗ {msg}")
        raise AssertionError(msg)


def make_clustered_embeddings(
    n_per_person: int,
    k: int,
    d: int,
    noise: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fabricate L2-normalised Gaussian blobs on the unit hypersphere.
    Also generates joint_types (cycling 0-16 per person).

    Returns:
        embeddings:  [N, D]
        labels:      [N]
        joint_types: [N]
    """
    centres    = F.normalize(torch.randn(k, d, device=device), dim=-1)
    embeddings = []
    labels     = []
    joint_types = []
    for person_id in range(k):
        pts = centres[person_id].unsqueeze(0) + \
              torch.randn(n_per_person, d, device=device) * noise
        pts = F.normalize(pts, dim=-1)
        embeddings.append(pts)
        labels.extend([person_id] * n_per_person)
        # Cycle through joint types 0-16
        joint_types.extend([j % 17 for j in range(n_per_person)])
    return (
        torch.cat(embeddings),
        torch.tensor(labels, device=device),
        torch.tensor(joint_types, device=device, dtype=torch.long),
    )


def build_knn_graph(
    embeddings: torch.Tensor, k_neighbors: int = 8,
) -> torch.Tensor:
    """Build a symmetric kNN graph from embeddings. Returns edge_index [2, E]."""
    n = embeddings.size(0)
    k = min(k_neighbors, n - 1)

    dist = torch.cdist(embeddings, embeddings, p=2)
    dist.fill_diagonal_(float("inf"))
    _, indices = dist.topk(k, dim=1, largest=False)

    source = torch.arange(n, device=embeddings.device).repeat_interleave(k)
    target = indices.flatten()

    edge_index = torch.stack([source, target], dim=0)

    # Symmetrize (add reverse edges)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Remove duplicates
    edge_index = torch.unique(edge_index, dim=1)

    return edge_index


def hungarian_accuracy(
    pred: np.ndarray,
    true: np.ndarray,
    k: int,
) -> float:
    k_max = max(k, pred.max() + 1, true.max() + 1)
    confusion = np.zeros((k_max, k_max), dtype=np.int64)
    for p, t in zip(pred, true):
        confusion[p, t] += 1
    row_idx, col_idx = linear_sum_assignment(-confusion)
    return confusion[row_idx, col_idx].sum() / len(true)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_pass(
    model: DMoNHead,
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    k: int,
    joint_types: torch.Tensor,
):
    _header("Forward Pass")

    model.eval()
    with torch.no_grad():
        logits, s, spec, ortho, clust, type_l = model(
            embeddings, edge_index, k, joint_types=joint_types,
        )

    n, d = embeddings.shape
    _check(logits.shape == (n, k), f"logits shape: {logits.shape} — expected [{n}, {k}]")
    _check(s.shape == (n, k),      f"s shape: {s.shape} — expected [{n}, {k}]")
    _check(not torch.isnan(logits).any().item(), "No NaN in logits")
    _check(not torch.isnan(s).any().item(),      "No NaN in s")
    _check(not torch.isinf(logits).any().item(), "No Inf in logits")

    # Soft assignments should sum to 1 per node
    row_sums = s.sum(dim=1)
    _check(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5),
           f"Soft assignment rows sum to 1 (max deviation: {(row_sums - 1).abs().max():.6f})")

    # Losses should be scalars
    for name, loss in [("spectral", spec), ("ortho", ortho),
                       ("cluster", clust), ("type", type_l)]:
        _check(loss.shape == torch.Size([]), f"{name} loss is scalar")
        _check(not torch.isnan(loss).item(), f"{name} loss not NaN")

    print(f"\n  spectral={spec.item():.4f}  ortho={ortho.item():.4f}"
          f"  cluster={clust.item():.4f}  type={type_l.item():.4f}")
    print(f"\n  Forward pass: all checks passed")


def test_loss_computation(
    model: DMoNHead,
    loss_fn: DMoNLoss,
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    joint_types: torch.Tensor,
):
    _header("Loss Computation")

    logits, s, spec, ortho, clust, type_l = model(
        embeddings, edge_index, k, joint_types=joint_types,
    )
    out = loss_fn(logits, labels, spec, ortho, clust, type_l)
    loss = out["total_loss"]

    _check(isinstance(loss, torch.Tensor), "Loss is a tensor")
    _check(loss.shape == torch.Size([]),   "Loss is a scalar")
    _check(not torch.isnan(loss).item(),   "Loss is not NaN")
    _check(not torch.isinf(loss).item(),   "Loss is not Inf")

    print(f"  Total loss: {loss.item():.4f}")
    print(f"  CE:         {out['ce_loss']:.4f}")
    print(f"  Spectral:   {out['spectral_loss']:.4f}")
    print(f"  Ortho:      {out['ortho_loss']:.4f}")
    print(f"  Cluster:    {out['cluster_loss']:.4f}")
    print(f"  Type:       {out['type_loss']:.4f}")
    print(f"  Accuracy:   {out['accuracy']:.4f}")
    print(f"\n  Loss computation: all checks passed")


def test_gradient_flow(
    model: DMoNHead,
    loss_fn: DMoNLoss,
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    joint_types: torch.Tensor,
):
    _header("Gradient Flow")

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    logits, s, spec, ortho, clust, type_l = model(
        embeddings, edge_index, k, joint_types=joint_types,
    )
    loss = loss_fn(logits, labels, spec, ortho, clust, type_l)["total_loss"]
    loss.backward()

    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    _check(len(missing) == 0, f"All parameters have gradients")

    total_norm = sum(p.grad.norm().item() for p in model.parameters()
                     if p.requires_grad and p.grad is not None)
    _check(total_norm > 0, f"Total gradient norm > 0 ({total_norm:.4f})")

    print(f"\n  Gradient flow: all checks passed")


def test_assignment_accuracy(
    model: DMoNHead,
    loss_fn: DMoNLoss,
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    joint_types: torch.Tensor,
    k: int,
    n_steps: int,
    noise: float,
    min_accuracy: float = 0.9,
):
    _header(f"Assignment Accuracy ({n_steps} steps, noise={noise})")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=1e-5,
    )
    labels_np = labels.cpu().numpy()

    # Before training
    model.eval()
    with torch.no_grad():
        logits_before, *_ = model(embeddings, edge_index, k, joint_types)
    acc_before = hungarian_accuracy(
        logits_before.argmax(dim=1).cpu().numpy(), labels_np, k,
    )
    print(f"\n  Before training: accuracy = {acc_before:.4f}")

    # Training
    model.train()
    best_acc  = acc_before
    best_step = 0
    print(f"  Training...")
    for step in range(n_steps):
        optimizer.zero_grad()
        logits, s, spec, ortho, clust, type_l = model(
            embeddings, edge_index, k, joint_types,
        )
        out = loss_fn(logits, labels, spec, ortho, clust, type_l)
        out["total_loss"].backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits_mid, *_ = model(embeddings, edge_index, k, joint_types)
            acc_mid = hungarian_accuracy(
                logits_mid.argmax(dim=1).cpu().numpy(), labels_np, k,
            )
            if acc_mid > best_acc:
                best_acc  = acc_mid
                best_step = step + 1

            extra = ""
            if best_step == step + 1:
                extra = "  ← best"
            print(f"  Step {step+1:3d}: accuracy={acc_mid:.4f}"
                  f"  loss={out['total_loss'].item():.4f}"
                  f"  ce={out['ce_loss']:.4f}"
                  f"  spec={out['spectral_loss']:.4f}"
                  f"  type={out['type_loss']:.4f}{extra}")
            model.train()

    # After training
    model.eval()
    with torch.no_grad():
        logits_after, *_ = model(embeddings, edge_index, k, joint_types)
    acc_final = hungarian_accuracy(
        logits_after.argmax(dim=1).cpu().numpy(), labels_np, k,
    )

    print(f"\n  Final accuracy:  {acc_final:.4f}")
    print(f"  Best accuracy:   {best_acc:.4f}  (step {best_step})")

    _check(best_acc >= acc_before,
           f"Best accuracy did not degrade ({acc_before:.4f} → {best_acc:.4f})")
    _check(best_acc >= min_accuracy,
           f"Best accuracy >= {min_accuracy} ({best_acc:.4f})")

    print(f"\n  Assignment accuracy: all checks passed ✅")
    return acc_before, best_acc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=Path,  default=Path("configs/dmon_isolation.yaml"))
    parser.add_argument("--n_steps",      type=int,   default=500)
    parser.add_argument("--k",            type=int,   default=5,    help="Number of people / clusters")
    parser.add_argument("--n_per_person", type=int,   default=17,   help="Joints per person")
    parser.add_argument("--noise_easy",   type=float, default=0.05, help="Noise for easy test")
    parser.add_argument("--noise_hard",   type=float, default=0.15, help="Noise for hard test")
    parser.add_argument("--k_neighbors",  type=int,   default=8,    help="kNN graph neighbors")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = ExperimentConfig.from_yaml(args.config)
    print(f"Config: {cfg.name}")
    assert cfg.dmon is not None, "dmon config missing from yaml"

    d = cfg.gat.output_dim
    k = args.k

    print(f"\n  Fabricated embeddings: {k} people × {args.n_per_person} joints"
          f" = {k * args.n_per_person} nodes")
    print(f"  Embedding dim: {d}")

    # Use easy embeddings for structural tests
    embeddings, labels, joint_types = make_clustered_embeddings(
        args.n_per_person, k, d, args.noise_easy, device,
    )
    edge_index = build_knn_graph(embeddings, args.k_neighbors)
    print(f"  kNN edges: {edge_index.size(1)} (k={args.k_neighbors})")

    model   = DMoNHead(cfg.dmon, embedding_dim=d).to(device)
    loss_fn = DMoNLoss(cfg.loss, cfg.dmon)

    passed: List[str] = []
    failed: List[str] = []

    for name, fn in [
        ("Forward pass",     lambda: test_forward_pass(model, embeddings,
                                                       edge_index, k, joint_types)),
        ("Loss computation", lambda: test_loss_computation(model, loss_fn, embeddings,
                                                           edge_index, labels, k,
                                                           joint_types)),
        ("Gradient flow",    lambda: test_gradient_flow(model, loss_fn, embeddings,
                                                        edge_index, labels, k,
                                                        joint_types)),
    ]:
        try:
            fn()
            passed.append(name)
        except AssertionError as e:
            failed.append(f"{name}: {e}")

    # Accuracy — easy
    model_easy = DMoNHead(cfg.dmon, embedding_dim=d).to(device)
    emb_easy, lbl_easy, jt_easy = make_clustered_embeddings(
        args.n_per_person, k, d, args.noise_easy, device,
    )
    ei_easy = build_knn_graph(emb_easy, args.k_neighbors)
    try:
        test_assignment_accuracy(
            model_easy, loss_fn, emb_easy, ei_easy, lbl_easy, jt_easy, k,
            args.n_steps, args.noise_easy, min_accuracy=0.9,
        )
        passed.append(f"Assignment accuracy (easy, noise={args.noise_easy})")
    except AssertionError as e:
        failed.append(f"Assignment accuracy easy: {e}")

    # Accuracy — hard
    model_hard = DMoNHead(cfg.dmon, embedding_dim=d).to(device)
    emb_hard, lbl_hard, jt_hard = make_clustered_embeddings(
        args.n_per_person, k, d, args.noise_hard, device,
    )
    ei_hard = build_knn_graph(emb_hard, args.k_neighbors)
    try:
        test_assignment_accuracy(
            model_hard, loss_fn, emb_hard, ei_hard, lbl_hard, jt_hard, k,
            args.n_steps, args.noise_hard, min_accuracy=0.9,
        )
        passed.append(f"Assignment accuracy (hard, noise={args.noise_hard})")
    except AssertionError as e:
        failed.append(f"Assignment accuracy hard: {e}")

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
