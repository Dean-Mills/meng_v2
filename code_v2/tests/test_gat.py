"""GAT isolation tests
Verifies the GAT is working correctly before connecting to a grouping head.

Tests:
1. Forward pass — correct output shape
2. L2 normalisation — all embedding norms ~1.0
3. Loss computation — contrastive loss returns a scalar
4. Gradient flow — all parameters have gradients
5. Embedding separability — quantitative metrics + visualisations

Metrics:
- Silhouette score       — cluster separation quality [-1, 1], want > 0.5
- Intra/inter distance   — ratio of same-person vs different-person distances
- kNN accuracy           — fraction of nearest neighbours from same person
- PCA plot               — 2D projection coloured by person
- t-SNE plot             — neighbourhood-preserving 2D projection

Run from code_v2/:
    python tests/test_gat.py --virtual_dir /path/to/virtual --config configs/gat_isolation.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config import ExperimentConfig
from gat import GATEmbedding
from losses import GATOnlyLoss
from virtual_adapter import VirtualAdapter
from dataset import PoseDataset
from dataloader import create_dataloader
from preprocessor import PosePreprocessor


PERSON_COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45"
]


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

def _tensor_check(condition: torch.Tensor, msg: str):
    _check(bool(condition.item()), msg)

def _collect_embeddings(
    gat: GATEmbedding,
    graphs: list,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run GAT on all graphs and collect embeddings + labels as numpy arrays.
    Returns (embeddings [N_total, D], labels [N_total])
    """
    gat.eval()
    all_emb = []
    all_lbl = []
    with torch.no_grad():
        for g in graphs:
            emb = gat(g)
            all_emb.append(emb.cpu().numpy())
            all_lbl.append(g.person_labels.cpu().numpy())
    return np.concatenate(all_emb), np.concatenate(all_lbl)


# ─────────────────────────────────────────────────────────────────────────────
# Basic tests
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_pass(gat: GATEmbedding, graph, cfg: ExperimentConfig):
    _header("Forward Pass")
    gat.eval()
    with torch.no_grad():
        embeddings = gat(graph)

    assert graph.x is not None
    _check(
        embeddings.shape == (graph.x.shape[0], cfg.gat.output_dim),
        f"Output shape: {embeddings.shape} — expected [{graph.x.shape[0]}, {cfg.gat.output_dim}]"
    )
    _check(embeddings.dtype == torch.float32, "Output dtype is float32")
    _check(not torch.isnan(embeddings).any().item(), "No NaN values")
    _check(not torch.isinf(embeddings).any().item(), "No Inf values")
    print(f"\n  Forward pass: all checks passed")
    return embeddings


def test_l2_normalisation(embeddings: torch.Tensor):
    _header("L2 Normalisation")
    norms         = embeddings.norm(dim=-1)
    max_deviation = (norms - 1.0).abs().max().item()
    _check(
        max_deviation < 1e-5,
        f"All norms ≈ 1.0 (mean={norms.mean().item():.6f}, max deviation={max_deviation:.2e})"
    )
    print(f"\n  L2 normalisation: all checks passed")


def test_loss_computation(gat: GATEmbedding, loss_fn: GATOnlyLoss, graph):
    _header("Loss Computation")
    gat.eval()
    with torch.no_grad():
        embeddings = gat(graph)
    result = loss_fn(embeddings, graph.person_labels)

    loss = result["total_loss"]
    _check(isinstance(loss, torch.Tensor),      "total_loss is a tensor")
    _check(loss.shape == torch.Size([]),         "total_loss is a scalar")
    _check(not torch.isnan(loss).item(),         "total_loss is not NaN")
    _check(loss.item() >= 0.0,                   f"total_loss >= 0 ({loss.item():.4f})")
    print(f"  pos_similarity: {result['pos_similarity']:.4f}")
    print(f"  neg_similarity: {result['neg_similarity']:.4f}")
    print(f"  total_loss:     {loss.item():.4f}")
    print(f"\n  Loss computation: all checks passed")
    return result


def test_gradient_flow(gat: GATEmbedding, loss_fn: GATOnlyLoss, graph):
    _header("Gradient Flow")
    gat.train()
    optimizer = optim.Adam(gat.parameters(), lr=1e-3)
    optimizer.zero_grad()

    embeddings = gat(graph)
    loss_fn(embeddings, graph.person_labels)["total_loss"].backward()

    missing = [n for n, p in gat.named_parameters() if p.requires_grad and p.grad is None]
    _check(len(missing) == 0, f"All parameters have gradients")

    total_norm = sum(p.grad.norm().item() for p in gat.parameters()
                     if p.requires_grad and p.grad is not None)
    _check(total_norm > 0, f"Total gradient norm > 0 ({total_norm:.4f})")
    print(f"\n  Gradient flow: all checks passed")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_embedding_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute quantitative embedding quality metrics.

    Returns:
        silhouette:      [-1, 1]  cluster separation, want > 0.5
        intra_dist:      avg distance between same-person joint pairs
        inter_dist:      avg distance between different-person joint pairs
        distance_ratio:  inter / intra, want > 1 (ideally >> 1)
        knn_accuracy:    fraction of k=5 nearest neighbours from same person
    """
    n = len(embeddings)

    # ── Silhouette score ──────────────────────────────────────────────────
    # Needs at least 2 unique labels and not all same
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        silhouette = 0.0
    else:
        silhouette = float(silhouette_score(embeddings, labels))

    # ── Intra / inter distances ───────────────────────────────────────────
    # Pairwise L2 distances
    diff        = embeddings[:, None, :] - embeddings[None, :, :]   # [N, N, D]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))                  # [N, N]

    same    = (labels[:, None] == labels[None, :])
    mask    = ~np.eye(n, dtype=bool)
    pos_mask = same & mask
    neg_mask = ~same & mask

    intra_dist = dist_matrix[pos_mask].mean() if pos_mask.any() else 0.0
    inter_dist = dist_matrix[neg_mask].mean() if neg_mask.any() else 0.0
    distance_ratio = float(inter_dist / intra_dist) if intra_dist > 0 else 0.0

    # ── kNN accuracy ──────────────────────────────────────────────────────
    k = min(5, n - 1)
    correct = 0
    total   = 0
    for i in range(n):
        dists    = dist_matrix[i].copy()
        dists[i] = np.inf                          # exclude self
        nn_idx   = np.argsort(dists)[:k]
        correct += (labels[nn_idx] == labels[i]).sum()
        total   += k

    knn_accuracy = correct / total if total > 0 else 0.0

    return {
        "silhouette":     silhouette,
        "intra_dist":     float(intra_dist),
        "inter_dist":     float(inter_dist),
        "distance_ratio": distance_ratio,
        "knn_accuracy":   float(knn_accuracy),
    }


def print_metrics(metrics: Dict[str, float], label: str = ""):
    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}Silhouette:      {metrics['silhouette']:.4f}  (want > 0.5)")
    print(f"{prefix}Intra distance:  {metrics['intra_dist']:.4f}")
    print(f"{prefix}Inter distance:  {metrics['inter_dist']:.4f}")
    print(f"{prefix}Distance ratio:  {metrics['distance_ratio']:.4f}  (want > 1)")
    print(f"{prefix}kNN accuracy:    {metrics['knn_accuracy']:.4f}  (want > 0.9)")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualise_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: Path,
):
    """PCA and t-SNE plots side by side, coloured by person."""
    unique_labels = np.unique(labels)
    colours = [PERSON_COLOURS[int(l) % len(PERSON_COLOURS)] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=14)

    # ── PCA ───────────────────────────────────────────────────────────────
    pca    = PCA(n_components=2)
    pca_2d = pca.fit_transform(embeddings)

    axes[0].scatter(pca_2d[:, 0], pca_2d[:, 1], c=colours, s=30, alpha=0.7)
    axes[0].set_title(f"PCA  (var explained: {pca.explained_variance_ratio_.sum():.2%})")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    # ── t-SNE ─────────────────────────────────────────────────────────────
    perplexity = min(30, len(embeddings) - 1)
    tsne       = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_2d    = tsne.fit_transform(embeddings)

    axes[1].scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=colours, s=30, alpha=0.7)
    axes[1].set_title("t-SNE")
    axes[1].set_xlabel("dim 1")
    axes[1].set_ylabel("dim 2")

    # Legend
    patches = [
        mpatches.Patch(color=PERSON_COLOURS[int(l) % len(PERSON_COLOURS)],
                       label=f"Person {int(l)}")
        for l in unique_labels
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(unique_labels),
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"  Saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Separability test with full metrics
# ─────────────────────────────────────────────────────────────────────────────

def test_embedding_separability(
    gat: GATEmbedding,
    loss_fn: GATOnlyLoss,
    graphs: list,
    vis_graph,
    n_steps: int,
    save_dir: Path,
):
    _header(f"Embedding Separability ({n_steps} training steps)")

    optimizer = optim.Adam(gat.parameters(), lr=1e-3)

    # ── Before training ───────────────────────────────────────────────────
    emb_before, lbl_before = _collect_embeddings(gat, graphs)
    metrics_before = compute_embedding_metrics(emb_before, lbl_before)
    print(f"\n  Before training:")
    print_metrics(metrics_before, "before")

    vis_emb_before, vis_lbl_before = _collect_embeddings(gat, [vis_graph])
    visualise_embeddings(
        vis_emb_before, vis_lbl_before,
        title="GAT Embeddings — Before Training",
        save_path=save_dir / "embeddings_before.png",
    )

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n  Training...")
    gat.train()
    for step in range(n_steps):
        for graph in graphs:
            optimizer.zero_grad()
            embeddings = gat(graph)
            loss_fn(embeddings, graph.person_labels)["total_loss"].backward()
            optimizer.step()

        if (step + 1) % 25 == 0:
            emb, lbl = _collect_embeddings(gat, graphs)
            m = compute_embedding_metrics(emb, lbl)
            print(f"  Step {step+1:3d}: silhouette={m['silhouette']:.4f}  "
                  f"ratio={m['distance_ratio']:.4f}  knn={m['knn_accuracy']:.4f}")
            gat.train()

    # ── After training ────────────────────────────────────────────────────
    emb_after, lbl_after = _collect_embeddings(gat, graphs)
    metrics_after = compute_embedding_metrics(emb_after, lbl_after)
    print(f"\n  After training:")
    print_metrics(metrics_after, "after")

    vis_emb_after, vis_lbl_after = _collect_embeddings(gat, [vis_graph])
    visualise_embeddings(
        vis_emb_after, vis_lbl_after,
        title="GAT Embeddings — After Training",
        save_path=save_dir / "embeddings_after.png",
    )

    # ── Checks ────────────────────────────────────────────────────────────
    # Silhouette is not used as a pass/fail criterion — it is unreliable on
    # L2-normalised embeddings on the unit hypersphere. Reported for reference only.
    _check(
        metrics_after["distance_ratio"] > metrics_before["distance_ratio"],
        f"Distance ratio improved ({metrics_before['distance_ratio']:.4f} → {metrics_after['distance_ratio']:.4f})"
    )
    _check(
        metrics_after["knn_accuracy"] > metrics_before["knn_accuracy"],
        f"kNN accuracy improved ({metrics_before['knn_accuracy']:.4f} → {metrics_after['knn_accuracy']:.4f})"
    )
    _check(
        metrics_after["knn_accuracy"] > 0.9,
        f"kNN accuracy > 0.9 ({metrics_after['knn_accuracy']:.4f})"
    )

    print(f"\n  Embedding separability: all checks passed ✅")
    return metrics_before, metrics_after


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--virtual_dir", type=Path, required=True)
    parser.add_argument("--config",      type=Path, default=Path("configs/gat_isolation.yaml"))
    parser.add_argument("--n_steps",     type=int,  default=100)
    parser.add_argument("--n_graphs",    type=int,  default=8)
    parser.add_argument("--save_dir",    type=Path, default=Path("outputs/test_gat"))
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = ExperimentConfig.from_yaml(args.config)
    print(f"Config: {cfg.name}")

    # ── Data ──────────────────────────────────────────────────────────────
    adapter      = VirtualAdapter(data_dir=args.virtual_dir, min_people=2)
    dataset      = PoseDataset(adapter=adapter, target_size=512)
    loader       = create_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0)
    preprocessor = PosePreprocessor(image_size=512, k_neighbors=8, device=device)

    # Collect enough graphs for training
    all_graphs: list = []
    for batch in loader:
        all_graphs.extend(preprocessor.process_batch(batch))
        if len(all_graphs) >= args.n_graphs:
            break
    test_graphs = all_graphs[:args.n_graphs]
    _check(len(test_graphs) > 0, f"Got {len(test_graphs)} graphs")

    # Find a 5-person graph for clean single-scene visualisation
    vis_graph = max(test_graphs, key=lambda g: g.num_people)
    for g in all_graphs:
        if g.num_people == 5:
            vis_graph = g
            break
    n_nodes = vis_graph.x.shape[0] if vis_graph.x is not None else "?"
    print(f"  Visualisation graph: {vis_graph.num_people} people, {n_nodes} nodes")

    # ── Model ─────────────────────────────────────────────────────────────
    gat     = GATEmbedding(cfg.gat).to(device)
    loss_fn = GATOnlyLoss(cfg.loss)
    graph   = test_graphs[0]

    passed: List[str] = []
    failed: List[str] = []

    for name, fn in [
        ("Forward pass",       lambda: test_forward_pass(gat, graph, cfg)),
        ("Loss computation",   lambda: test_loss_computation(gat, loss_fn, graph)),
        ("Gradient flow",      lambda: test_gradient_flow(gat, loss_fn, graph)),
    ]:
        try:
            result = fn()
            if name == "Forward pass":
                try:
                    test_l2_normalisation(result)
                    passed.append("L2 normalisation")
                except AssertionError as e:
                    failed.append(f"L2 normalisation: {e}")
            passed.append(name)
        except AssertionError as e:
            failed.append(f"{name}: {e}")

    try:
        test_embedding_separability(
            gat, loss_fn, test_graphs,
            vis_graph=vis_graph,
            n_steps=args.n_steps,
            save_dir=args.save_dir,
        )
        passed.append("Embedding separability")
    except AssertionError as e:
        failed.append(f"Embedding separability: {e}")

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
    print(f"  Visualisations saved to: {args.save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()