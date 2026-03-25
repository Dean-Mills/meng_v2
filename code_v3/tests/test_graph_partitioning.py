"""Graph partitioning isolation tests.

Verifies the edge classifier works correctly in isolation using fabricated
embeddings. No real data or GAT required — same approach as test_dec.py
and test_slot_attention.py.

The edge classifier predicts for every pair of joints whether they belong
to the same person. Groups are recovered by thresholding predictions and
finding connected components.

Tests:
1. Forward pass       — logits [E], pairs [E, 2], no NaN
2. Loss computation   — scalar, not NaN, >= 0, reports edge F1 and grouping accuracy
3. Gradient flow      — all parameters receive gradients
4. Edge F1            — easy (noise=0.05) and hard (noise=0.15), both at 17 joints/person
5. Grouping accuracy  — connected components + Hungarian matching >= 0.9

Run from code_v2/:
    python tests/test_graph_partitioning.py \\
        --config configs/graph_partitioning_isolation.yaml
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

from config import ExperimentConfig
from graph_partitioning import EdgeClassifier
from losses import GraphPartitioningLoss


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fabricate L2-normalised Gaussian blobs on the unit hypersphere."""
    centres    = F.normalize(torch.randn(k, d, device=device), dim=-1)
    embeddings = []
    labels     = []
    for person_id in range(k):
        pts = centres[person_id].unsqueeze(0) + \
              torch.randn(n_per_person, d, device=device) * noise
        pts = F.normalize(pts, dim=-1)
        embeddings.append(pts)
        labels.extend([person_id] * n_per_person)
    return torch.cat(embeddings), torch.tensor(labels, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_pass(model: EdgeClassifier, embeddings: torch.Tensor):
    _header("Forward Pass")

    n, d = embeddings.shape
    e_pairs = n * (n - 1) // 2   # expected number of pairs

    model.eval()
    with torch.no_grad():
        logits, pairs = model(embeddings)

    _check(logits.shape == (e_pairs,),    f"logits shape: {logits.shape} — expected [{e_pairs}]")
    _check(pairs.shape  == (e_pairs, 2),  f"pairs shape:  {pairs.shape}  — expected [{e_pairs}, 2]")
    _check(not torch.isnan(logits).any().item(), "No NaN in logits")
    _check(not torch.isinf(logits).any().item(), "No Inf in logits")
    _check((pairs[:, 0] < pairs[:, 1]).all().item(), "All pairs have i < j")

    print(f"  Pairs: {e_pairs}  ({n} nodes)")
    print(f"\n  Forward pass: all checks passed")
    return logits, pairs


def test_loss_computation(
    model:   EdgeClassifier,
    loss_fn: GraphPartitioningLoss,
    embeddings:    torch.Tensor,
    labels:        torch.Tensor,
):
    _header("Loss Computation")

    logits, pairs = model(embeddings)
    out  = loss_fn(logits, pairs, labels)
    loss = out["total_loss"]

    _check(isinstance(loss, torch.Tensor),  "Loss is a tensor")
    _check(loss.shape == torch.Size([]),     "Loss is a scalar")
    _check(not torch.isnan(loss).item(),     "Loss is not NaN")
    _check(not torch.isinf(loss).item(),     "Loss is not Inf")
    _check(loss.item() >= 0.0,               f"Loss >= 0 ({loss.item():.4f})")

    print(f"  Loss:             {loss.item():.4f}")
    print(f"  Edge F1:          {out['edge_f1']:.4f}")
    print(f"  Edge accuracy:    {out['edge_accuracy']:.4f}")
    print(f"  Grouping accuracy:{out['grouping_accuracy']:.4f}")
    print(f"\n  Loss computation: all checks passed")


def test_gradient_flow(
    model:   EdgeClassifier,
    loss_fn: GraphPartitioningLoss,
    embeddings: torch.Tensor,
    labels:     torch.Tensor,
):
    _header("Gradient Flow")

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    logits, pairs = model(embeddings)
    loss = loss_fn(logits, pairs, labels)["total_loss"]
    loss.backward()

    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    _check(len(missing) == 0, "All parameters have gradients")

    total_norm = sum(p.grad.norm().item() for p in model.parameters()
                     if p.requires_grad and p.grad is not None)
    _check(total_norm > 0, f"Total gradient norm > 0 ({total_norm:.4f})")

    print(f"\n  Gradient flow: all checks passed")


def test_assignment_accuracy(
    model:   EdgeClassifier,
    loss_fn: GraphPartitioningLoss,
    embeddings: torch.Tensor,
    labels:     torch.Tensor,
    n_steps:    int,
    noise:      float,
    min_edge_f1:        float = 0.9,
    min_grouping_acc:   float = 0.9,
):
    _header(f"Assignment Accuracy ({n_steps} steps, noise={noise})")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=1e-5
    )

    # Before training
    model.eval()
    with torch.no_grad():
        logits_before, pairs = model(embeddings)
    out_before = loss_fn(logits_before, pairs, labels)
    f1_before  = out_before["edge_f1"]
    ga_before  = out_before["grouping_accuracy"]
    print(f"\n  Before training: edge F1 = {f1_before:.4f}  grouping acc = {ga_before:.4f}")

    # Training
    best_f1 = f1_before
    best_ga = ga_before
    best_f1_step = 0
    best_ga_step = 0

    model.train()
    print(f"  Training...")
    for step in range(n_steps):
        optimizer.zero_grad()
        logits, pairs = model(embeddings)
        loss = loss_fn(logits, pairs, labels)["total_loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits_mid, pairs_mid = model(embeddings)
            out_mid = loss_fn(logits_mid, pairs_mid, labels)
            f1_mid  = out_mid["edge_f1"]
            ga_mid  = out_mid["grouping_accuracy"]

            tags = []
            if f1_mid > best_f1:
                best_f1      = f1_mid
                best_f1_step = step + 1
                tags.append("best F1")
            if ga_mid > best_ga:
                best_ga      = ga_mid
                best_ga_step = step + 1
                tags.append("best grouping")

            tag_str = f"  ← {', '.join(tags)}" if tags else ""
            print(f"  Step {step+1:3d}: F1={f1_mid:.4f}  grouping={ga_mid:.4f}"
                  f"  loss={loss.item():.4f}{tag_str}")
            model.train()

    # After training
    model.eval()
    with torch.no_grad():
        logits_after, pairs_after = model(embeddings)
    out_after = loss_fn(logits_after, pairs_after, labels)
    f1_final  = out_after["edge_f1"]
    ga_final  = out_after["grouping_accuracy"]

    print(f"\n  Final:  edge F1 = {f1_final:.4f}  grouping acc = {ga_final:.4f}")
    print(f"  Best:   edge F1 = {best_f1:.4f}  (step {best_f1_step})"
          f"    grouping acc = {best_ga:.4f}  (step {best_ga_step})")

    _check(best_f1 >= min_edge_f1,
           f"Best edge F1 >= {min_edge_f1} ({best_f1:.4f})")
    _check(best_ga >= min_grouping_acc,
           f"Best grouping accuracy >= {min_grouping_acc} ({best_ga:.4f})")

    print(f"\n  Assignment accuracy: all checks passed ✅")
    return best_f1, best_ga


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       type=Path,  default=Path("configs/graph_partitioning_isolation.yaml"))
    parser.add_argument("--n_steps",      type=int,   default=500)
    parser.add_argument("--k",            type=int,   default=5,    help="Number of people")
    parser.add_argument("--n_per_person", type=int,   default=17,   help="Joints per person")
    parser.add_argument("--noise_easy",   type=float, default=0.05)
    parser.add_argument("--noise_hard",   type=float, default=0.15)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = ExperimentConfig.from_yaml(args.config)
    print(f"Config: {cfg.name}")
    assert cfg.graph_partitioning is not None, "graph_partitioning config missing from yaml"

    d = cfg.gat.output_dim
    k = args.k
    n = k * args.n_per_person

    print(f"\n  Fabricated embeddings: {k} people × {args.n_per_person} joints = {n} nodes")
    print(f"  Embedding dim: {d}")
    print(f"  Edge pairs:    {n * (n - 1) // 2}")

    embeddings, labels = make_clustered_embeddings(
        args.n_per_person, k, d, args.noise_easy, device
    )

    model   = EdgeClassifier(cfg.graph_partitioning, embedding_dim=d).to(device)
    loss_fn = GraphPartitioningLoss(cfg.loss)

    passed: List[str] = []
    failed: List[str] = []

    for name, fn in [
        ("Forward pass",     lambda: test_forward_pass(model, embeddings)),
        ("Gradient flow",    lambda: test_gradient_flow(model, loss_fn, embeddings, labels)),
        ("Loss computation", lambda: test_loss_computation(model, loss_fn, embeddings, labels)),
    ]:
        try:
            fn()
            passed.append(name)
        except AssertionError as e:
            failed.append(f"{name}: {e}")

    # Easy accuracy
    model_easy = EdgeClassifier(cfg.graph_partitioning, embedding_dim=d).to(device)
    emb_easy, lbl_easy = make_clustered_embeddings(
        args.n_per_person, k, d, args.noise_easy, device
    )
    try:
        test_assignment_accuracy(model_easy, loss_fn, emb_easy, lbl_easy,
                                 args.n_steps, args.noise_easy)
        passed.append(f"Assignment accuracy (easy, noise={args.noise_easy})")
    except AssertionError as e:
        failed.append(f"Assignment accuracy easy: {e}")

    # Hard accuracy
    model_hard = EdgeClassifier(cfg.graph_partitioning, embedding_dim=d).to(device)
    emb_hard, lbl_hard = make_clustered_embeddings(
        args.n_per_person, k, d, args.noise_hard, device
    )
    try:
        test_assignment_accuracy(model_hard, loss_fn, emb_hard, lbl_hard,
                                 args.n_steps, args.noise_hard)
        passed.append(f"Assignment accuracy (hard, noise={args.noise_hard})")
    except AssertionError as e:
        failed.append(f"Assignment accuracy hard: {e}")

    # Summary
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