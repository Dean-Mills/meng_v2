"""
THA-MR — Type-wise Hungarian Assignment with Multi-Restart selection.

Test-time compute extension of THA (eval_hungarian_grouping.py). Standard
THA runs a single deterministic k-means initialisation; COP-Kmeans gets 10
random restarts with best-inertia selection. THA-MR levels the playing
field: run THA from n_restarts different centroid initialisations and keep
the solution with the lowest constrained inertia.

Restart 0 reproduces the baseline THA initialisation exactly (best-of-10
k-means, random_state=42), so THA-MR is never worse than THA in the
selection objective. Note the honest caveat: lowest inertia does not
guarantee highest PGA.

This file is a pure addition — it imports the existing predict_tha /
predict_cop_kmeans / predict_knn unchanged and reports them as baseline
columns next to THA-MR. No existing module is modified.

Usage:
    python eval_tha_multirestart.py \
        --checkpoint outputs/pg_gat_finetune_sweep/3362f752/best.pt \
        --virtual_dir data/virtual

    python eval_tha_multirestart.py \
        --checkpoint outputs/pg_gat_finetune_sweep/3362f752/best.pt \
        --coco_img_dir data/coco2017/val2017 \
        --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from config import ExperimentConfig
from gat import GATEmbedding
from preprocessor import PosePreprocessor
from dataset import PoseDataset
from dataloader import create_dataloader
from virtual_adapter import VirtualAdapter
from evaluator import compute_pga, predict_knn
from eval_cop_kmeans import predict_cop_kmeans
from eval_hungarian_grouping import predict_tha

NUM_JOINT_TYPES = 17


# ─────────────────────────────────────────────────────────────────────────────
# Core algorithm
# ─────────────────────────────────────────────────────────────────────────────

def _tha_from_init(
    emb_np: np.ndarray,
    k: int,
    jt_np: np.ndarray,
    centroids: np.ndarray,
    n_iters: int = 10,
) -> tuple[np.ndarray, float]:
    """
    Run the THA loop (per-type Hungarian assignment + centroid updates)
    from a given centroid initialisation.

    Assignment logic mirrors predict_tha in eval_hungarian_grouping.py
    exactly — only the initialisation is externalised.

    Args:
        emb_np:    [N, D] L2-normalised embeddings
        k:         number of people
        jt_np:     [N] joint type indices 0-16
        centroids: [K, D] initial centroids
        n_iters:   number of centroid update iterations

    Returns:
        labels:  [N] person assignment labels
        inertia: sum of squared distances to assigned final centroids
    """
    n = len(emb_np)
    centroids = centroids.copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(n_iters):
        # ── Per-type Hungarian assignment ─────────────────────────────
        for jt in range(NUM_JOINT_TYPES):
            mask = jt_np == jt
            if mask.sum() == 0:
                continue

            indices = np.where(mask)[0]
            type_embs = emb_np[indices]  # [M, D]

            # Cost matrix: distance from each joint of this type to each
            # centroid [M, K]
            cost = np.linalg.norm(
                type_embs[:, None] - centroids[None, :], axis=2
            )

            m = len(indices)
            if m <= k:
                # Fewer joints than people — standard Hungarian
                row, col = linear_sum_assignment(cost)
                for r, c in zip(row, col):
                    labels[indices[r]] = c
            else:
                # More joints than people — match people to joints,
                # leftovers go to nearest centroid
                cost_t = cost.T  # [K, M]
                row, col = linear_sum_assignment(cost_t)
                assigned = set()
                for r, c in zip(row, col):
                    labels[indices[c]] = r
                    assigned.add(c)
                for j in range(m):
                    if j not in assigned:
                        labels[indices[j]] = np.argmin(cost[j])

        # ── Update centroids ──────────────────────────────────────────
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k)
        for i in range(n):
            new_centroids[labels[i]] += emb_np[i]
            counts[labels[i]] += 1

        for c in range(k):
            if counts[c] > 0:
                new_centroids[c] /= counts[c]
            else:
                new_centroids[c] = centroids[c]  # keep old if empty

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    inertia = float(
        np.sum((emb_np - centroids[labels]) ** 2)
    )
    return labels, inertia


def predict_tha_multirestart(
    embeddings: torch.Tensor,
    k: int,
    joint_types: torch.Tensor,
    n_restarts: int = 10,
    n_iters: int = 10,
) -> torch.Tensor:
    """
    THA with multi-restart selection.

    Restart 0 uses the baseline THA initialisation (deterministic
    best-of-10 k-means, random_state=42). Restarts 1..n-1 use single
    k-means++ initialisations with seeds 1..n-1. The solution with the
    lowest constrained inertia is returned.

    Args:
        embeddings:  [N, D] L2-normalised joint embeddings
        k:           number of people
        joint_types: [N] joint type indices 0-16
        n_restarts:  total restarts including the baseline init
        n_iters:     THA centroid update iterations per restart

    Returns:
        labels: [N] person assignment labels
    """
    emb_np = embeddings.cpu().numpy()
    jt_np = joint_types.cpu().numpy()

    best_labels = None
    best_inertia = float("inf")

    for r in range(n_restarts):
        if r == 0:
            # Baseline THA initialisation — identical to predict_tha
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
        else:
            km = KMeans(n_clusters=k, random_state=r, n_init=1)
        km.fit(emb_np)

        labels, inertia = _tha_from_init(
            emb_np, k, jt_np, km.cluster_centers_, n_iters=n_iters,
        )

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels

    return torch.tensor(best_labels, device=embeddings.device, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    checkpoint_path: Path,
    device: str,
    virtual_dir: Optional[Path] = None,
    split: str = "test",
    coco_img_dir: Optional[Path] = None,
    coco_ann_file: Optional[Path] = None,
    max_images: Optional[int] = None,
    n_restarts: int = 10,
):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ExperimentConfig(**ckpt["config"])

    is_hyperbolic = cfg.sa_gat_hyperbolic is not None
    if is_hyperbolic:
        from sa_gat_hyperbolic import SAGATHyperbolicEmbedding
        gat = SAGATHyperbolicEmbedding(cfg.sa_gat_hyperbolic).to(device)
        embedding_dim = cfg.sa_gat_hyperbolic.output_dim
        use_depth = cfg.sa_gat_hyperbolic.use_depth
    elif cfg.sa_gat is not None:
        from sa_gat import SAGATEmbedding
        gat = SAGATEmbedding(cfg.sa_gat).to(device)
        embedding_dim = cfg.sa_gat.output_dim
        use_depth = cfg.sa_gat.use_depth
    else:
        gat = GATEmbedding(cfg.gat).to(device)
        embedding_dim = cfg.gat.output_dim
        use_depth = cfg.gat.use_depth
    gat.load_state_dict(ckpt["gat_state"])
    gat.eval()

    k_neighbors = 16 if embedding_dim >= 256 else 8
    preprocessor = PosePreprocessor(
        device=device, k_neighbors=k_neighbors, use_depth=use_depth,
    )

    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    if coco_img_dir is not None and coco_ann_file is not None:
        from coco_adapter import CocoAdapter
        adapter = CocoAdapter(
            img_dir=coco_img_dir, ann_file=coco_ann_file, device=device,
            use_depth=use_depth,
        )
        print(f"Evaluating on COCO: {coco_ann_file.name}")
    elif virtual_dir is not None:
        adapter = VirtualAdapter(virtual_dir / split)
        print(f"Evaluating on virtual/{split}")
    else:
        raise ValueError("Provide --virtual_dir or --coco_img_dir + --coco_ann_file")

    dataset = PoseDataset(adapter)
    loader = create_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)

    results = {"knn": [], "cop_kmeans": [], "tha": [], "tha_mr": []}

    with torch.no_grad():
        for batch in loader:
            graphs = preprocessor.process_batch(batch)
            for graph in graphs:
                if max_images and len(results["knn"]) >= max_images:
                    break
                graph = graph.to(device)
                embeddings = gat(graph)
                if is_hyperbolic:
                    import torch.nn.functional as F
                    logmap = gat.manifold.logmap0(embeddings)
                    embeddings = F.normalize(logmap[..., 1:], p=2, dim=-1)
                k = int(graph.num_people)
                gt = graph.person_labels

                knn_pred = predict_knn(embeddings, k)
                cop_pred = predict_cop_kmeans(embeddings, k, graph.joint_types)
                tha_pred = predict_tha(embeddings, k, graph.joint_types)
                tha_mr_pred = predict_tha_multirestart(
                    embeddings, k, graph.joint_types, n_restarts=n_restarts,
                )

                results["knn"].append(compute_pga(knn_pred, gt))
                results["cop_kmeans"].append(compute_pga(cop_pred, gt))
                results["tha"].append(compute_pga(tha_pred, gt))
                results["tha_mr"].append(compute_pga(tha_mr_pred, gt))

    n = len(results["knn"])
    print(f"\n{'='*60}")
    print(f"THA MULTI-RESTART ({n} images, {n_restarts} restarts)")
    print(f"{'='*60}")
    print(f"{'Method':<25}{'PGA':>10}{'Std':>10}")
    print("-" * 45)
    for method, label in [
        ("knn", "kNN"),
        ("cop_kmeans", "COP-Kmeans"),
        ("tha", "THA (single init)"),
        ("tha_mr", f"THA-MR ({n_restarts} restarts)"),
    ]:
        vals = results[method]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {label:<23}{mean:>10.4f}{std:>10.4f}")
    print(f"{'='*60}")

    # Save
    suffix = "coco" if coco_img_dir is not None else "virtual"
    save_path = checkpoint_path.parent / f"eval_tha_multirestart_{suffix}.json"
    save_data = {
        "n_images": n,
        "n_restarts": n_restarts,
        "pga": {
            m: {"mean": sum(v)/len(v),
                "std": (sum((x - sum(v)/len(v))**2 for x in v)/len(v))**0.5}
            for m, v in results.items()
        },
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {save_path}")

    return {
        f"{m}_pga": sum(v) / len(v)
        for m, v in results.items()
    } | {"n_images": n, "n_restarts": n_restarts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--virtual_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--coco_img_dir", type=Path, default=None)
    parser.add_argument("--coco_ann_file", type=Path, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--n_restarts", type=int, default=10)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Attach eval metrics to this W&B run id. "
                             "Default: read from checkpoint metadata. "
                             "Pass 'none' to skip.")
    args = parser.parse_args()

    metrics = evaluate(
        args.checkpoint, args.device,
        virtual_dir=args.virtual_dir, split=args.split,
        coco_img_dir=args.coco_img_dir, coco_ann_file=args.coco_ann_file,
        max_images=args.max_images, n_restarts=args.n_restarts,
    )

    run_id = args.wandb_run_id
    if run_id is None:
        from wandb_helpers import get_wandb_run_id_from_ckpt
        run_id = get_wandb_run_id_from_ckpt(args.checkpoint)
    if run_id and run_id.lower() != "none":
        from wandb_helpers import attach_eval_metrics
        dataset = "coco" if args.coco_img_dir is not None else "synth"
        url = attach_eval_metrics(run_id, f"eval/tha_multirestart/{dataset}", metrics)
        if url: print(f"W&B summary updated: {url}")


if __name__ == "__main__":
    main()
