"""Slot attention isolation tests.

Verifies slot attention works correctly in isolation using fabricated embeddings.
No real data or GAT required — same approach as test_dec.py.

Unlike DEC, slot attention is trained with ground truth labels so it should
handle the 17 joints per person / 128D case that DEC could not.

Tests:
1. Forward pass     — logits [N, K], slots [K, D], no NaN
2. Loss computation — scalar, not NaN, >= 0
3. Gradient flow    — all parameters receive gradients
4. Assignment accuracy — Hungarian-matched accuracy improves and exceeds 0.9
                         tested at easy (noise=0.05) and hard (noise=0.15)

Run from code_v2/:
    python tests/test_slot_attention.py --config configs/slot_attention_isolation.yaml
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
from slot_attention import SlotAttention
from losses import SlotAttentionLoss


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
        pts = centres[person_id].unsqueeze(0) + torch.randn(n_per_person, d, device=device) * noise
        pts = F.normalize(pts, dim=-1)
        embeddings.append(pts)
        labels.extend([person_id] * n_per_person)
    return torch.cat(embeddings), torch.tensor(labels, device=device)


def hungarian_accuracy(
    pred: np.ndarray,
    true: np.ndarray,
    k: int,
) -> float:
    confusion        = np.zeros((k, k), dtype=np.int64)
    for p, t in zip(pred, true):
        confusion[p, t] += 1
    row_idx, col_idx = linear_sum_assignment(-confusion)
    return confusion[row_idx, col_idx].sum() / len(true)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_pass(model: SlotAttention, embeddings: torch.Tensor, k: int):
    _header("Forward Pass")

    model.eval()
    with torch.no_grad():
        logits, slots = model(embeddings, k)

    n, d = embeddings.shape
    _check(logits.shape == (n, k),  f"logits shape: {logits.shape} — expected [{n}, {k}]")
    _check(slots.shape  == (k, d),  f"slots shape:  {slots.shape}  — expected [{k}, {d}]")
    _check(not torch.isnan(logits).any().item(), "No NaN in logits")
    _check(not torch.isnan(slots).any().item(),  "No NaN in slots")
    _check(not torch.isinf(logits).any().item(), "No Inf in logits")

    print(f"\n  Forward pass: all checks passed")
    return logits, slots


def test_loss_computation(
    model: SlotAttention,
    loss_fn: SlotAttentionLoss,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int,
):
    _header("Loss Computation")

    logits, _ = model(embeddings, k)
    out       = loss_fn(logits, labels)
    loss      = out["total_loss"]

    _check(isinstance(loss, torch.Tensor),      "Loss is a tensor")
    _check(loss.shape == torch.Size([]),         "Loss is a scalar")
    _check(not torch.isnan(loss).item(),         "Loss is not NaN")
    _check(not torch.isinf(loss).item(),         "Loss is not Inf")
    _check(loss.item() >= 0.0,                   f"Loss >= 0 ({loss.item():.4f})")

    print(f"  Loss:     {loss.item():.4f}")
    print(f"  Accuracy: {out['accuracy']:.4f}")
    print(f"\n  Loss computation: all checks passed")
    return loss


def test_gradient_flow(
    model: SlotAttention,
    loss_fn: SlotAttentionLoss,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int,
):
    _header("Gradient Flow")

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    logits, _ = model(embeddings, k)
    loss      = loss_fn(logits, labels)["total_loss"]
    loss.backward()

    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    _check(len(missing) == 0, f"All parameters have gradients")

    total_norm = sum(p.grad.norm().item() for p in model.parameters()
                     if p.requires_grad and p.grad is not None)
    _check(total_norm > 0, f"Total gradient norm > 0 ({total_norm:.4f})")

    print(f"\n  Gradient flow: all checks passed")


def test_assignment_accuracy(
    model: SlotAttention,
    loss_fn: SlotAttentionLoss,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    n_steps: int,
    noise: float,
    min_accuracy: float = 0.9,
):
    _header(f"Assignment Accuracy ({n_steps} steps, noise={noise})")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)
    labels_np = labels.cpu().numpy()

    # Before training
    model.eval()
    with torch.no_grad():
        logits_before, _ = model(embeddings, k)
    acc_before = hungarian_accuracy(
        logits_before.argmax(dim=1).cpu().numpy(), labels_np, k
    )
    print(f"\n  Before training: accuracy = {acc_before:.4f}")

    # Training
    model.train()
    best_acc  = acc_before
    best_step = 0
    print(f"  Training...")
    for step in range(n_steps):
        optimizer.zero_grad()
        logits, _ = model(embeddings, k)
        loss      = loss_fn(logits, labels)["total_loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                logits_mid, _ = model(embeddings, k)
            acc_mid = hungarian_accuracy(
                logits_mid.argmax(dim=1).cpu().numpy(), labels_np, k
            )
            if acc_mid > best_acc:
                best_acc  = acc_mid
                best_step = step + 1
            print(f"  Step {step+1:3d}: accuracy={acc_mid:.4f}  loss={loss.item():.4f}"
                  + ("  ← best" if best_step == step + 1 else ""))
            model.train()

    # After training
    model.eval()
    with torch.no_grad():
        logits_after, _ = model(embeddings, k)
    acc_final = hungarian_accuracy(
        logits_after.argmax(dim=1).cpu().numpy(), labels_np, k
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
    parser.add_argument("--config",       type=Path,  default=Path("configs/slot_attention_isolation.yaml"))
    parser.add_argument("--n_steps",      type=int,   default=500)
    parser.add_argument("--k",            type=int,   default=5,    help="Number of people / slots")
    parser.add_argument("--n_per_person", type=int,   default=17,   help="Joints per person")
    parser.add_argument("--noise_easy",   type=float, default=0.05, help="Noise for easy test")
    parser.add_argument("--noise_hard",   type=float, default=0.15, help="Noise for hard test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = ExperimentConfig.from_yaml(args.config)
    print(f"Config: {cfg.name}")
    assert cfg.slot_attention is not None, "slot_attention config missing from yaml"

    d = cfg.gat.output_dim
    k = args.k

    print(f"\n  Fabricated embeddings: {k} people × {args.n_per_person} joints = {k * args.n_per_person} nodes")
    print(f"  Embedding dim: {d}")

    # Use easy embeddings for structural tests
    embeddings, labels = make_clustered_embeddings(args.n_per_person, k, d, args.noise_easy, device)

    model   = SlotAttention(cfg.slot_attention, embedding_dim=d).to(device)
    loss_fn = SlotAttentionLoss(cfg.loss)

    passed: List[str] = []
    failed: List[str] = []

    for name, fn in [
        ("Forward pass",     lambda: test_forward_pass(model, embeddings, k)),
        ("Gradient flow",    lambda: test_gradient_flow(model, loss_fn, embeddings, labels, k)),
        ("Loss computation", lambda: test_loss_computation(model, loss_fn, embeddings, labels, k)),
    ]:
        try:
            fn()
            passed.append(name)
        except AssertionError as e:
            failed.append(f"{name}: {e}")

    # Accuracy — easy
    model_easy = SlotAttention(cfg.slot_attention, embedding_dim=d).to(device)
    emb_easy, lbl_easy = make_clustered_embeddings(args.n_per_person, k, d, args.noise_easy, device)
    try:
        test_assignment_accuracy(model_easy, loss_fn, emb_easy, lbl_easy, k,
                                 args.n_steps, args.noise_easy, min_accuracy=0.9)
        passed.append(f"Assignment accuracy (easy, noise={args.noise_easy})")
    except AssertionError as e:
        failed.append(f"Assignment accuracy easy: {e}")

    # Accuracy — hard
    model_hard = SlotAttention(cfg.slot_attention, embedding_dim=d).to(device)
    emb_hard, lbl_hard = make_clustered_embeddings(args.n_per_person, k, d, args.noise_hard, device)
    try:
        test_assignment_accuracy(model_hard, loss_fn, emb_hard, lbl_hard, k,
                                 args.n_steps, args.noise_hard, min_accuracy=0.9)
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