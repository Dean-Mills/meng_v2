"""
COP-Kmeans evaluation — constrained k-means with must-not-link
constraints for same-type keypoints.

No training required — uses existing contrastive GAT checkpoint.
The only change from standard k-means is that during assignment,
a keypoint is assigned to the nearest centroid that doesn't violate
the constraint: no two keypoints of the same type in one cluster.

Usage:
    python eval_cop_kmeans.py --checkpoint outputs/checkpoints/knn_no_depth/best.pt \
                              --virtual_dir data/virtual

    python eval_cop_kmeans.py --checkpoint outputs/checkpoints/knn_no_depth/best.pt \
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
from sklearn.cluster import KMeans

from config import ExperimentConfig
from gat import GATEmbedding
from preprocessor import PosePreprocessor
from dataset import PoseDataset
from dataloader import create_dataloader
from virtual_adapter import VirtualAdapter
from evaluator import compute_pga, predict_knn, COCO_JOINT_NAMES


def predict_cop_kmeans(
    embeddings: torch.Tensor,
    k: int,
    joint_types: torch.Tensor,
    n_init: int = 10,
) -> torch.Tensor:
    """
    COP-Kmeans: k-means with must-not-link constraints.

    Same-type keypoints cannot be in the same cluster.

    Args:
        embeddings:  [N, D] L2-normalised
        k:           number of clusters
        joint_types: [N] joint type indices 0-16
        n_init:      number of random restarts (best kept)

    Returns:
        labels: [N] cluster assignments
    """
    emb_np = embeddings.cpu().numpy()
    jt_np = joint_types.cpu().numpy()
    n, d = emb_np.shape

    # Build must-not-link pairs: all pairs of keypoints with the same type
    mnl_pairs = []
    type_indices = {}
    for i in range(n):
        t = jt_np[i]
        if t not in type_indices:
            type_indices[t] = []
        type_indices[t].append(i)

    for t, indices in type_indices.items():
        for a_idx in range(len(indices)):
            for b_idx in range(a_idx + 1, len(indices)):
                mnl_pairs.append((indices[a_idx], indices[b_idx]))

    best_labels = None
    best_inertia = float("inf")

    for _ in range(n_init):
        # Initialise with k-means++
        km = KMeans(n_clusters=k, n_init=1, random_state=None)
        km.fit(emb_np)
        centroids = km.cluster_centers_.copy()

        # Run constrained Lloyd iterations
        for _ in range(50):
            # Assignment step with constraints
            labels = _constrained_assign(emb_np, centroids, jt_np, k)

            # Update step
            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    new_centroids[c] = emb_np[mask].mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        # Compute inertia
        inertia = sum(
            np.sum((emb_np[labels == c] - centroids[c]) ** 2)
            for c in range(k)
        )

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return torch.tensor(best_labels, device=embeddings.device, dtype=torch.long)


def _constrained_assign(
    emb: np.ndarray,
    centroids: np.ndarray,
    joint_types: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Assign each point to nearest centroid without violating
    must-not-link constraints (no duplicate types per cluster).
    """
    n = emb.shape[0]

    # Distances to all centroids [N, K]
    dists = np.linalg.norm(
        emb[:, None, :] - centroids[None, :, :], axis=2
    )

    # Sort points by their best distance (assign most confident first)
    best_dists = dists.min(axis=1)
    order = np.argsort(best_dists)

    labels = np.full(n, -1, dtype=np.int64)
    # Track which types are used in each cluster
    cluster_types = [set() for _ in range(k)]

    for i in order:
        t = joint_types[i]
        # Try centroids in order of distance
        sorted_clusters = np.argsort(dists[i])

        assigned = False
        for c in sorted_clusters:
            if t not in cluster_types[c]:
                labels[i] = c
                cluster_types[c].add(t)
                assigned = True
                break

        if not assigned:
            # All clusters have this type — assign to nearest anyway
            # (this shouldn't happen if K >= actual number of people)
            labels[i] = sorted_clusters[0]

    return labels


def evaluate(
    checkpoint_path: Path,
    device: str,
    virtual_dir: Optional[Path] = None,
    split: str = "test",
    coco_img_dir: Optional[Path] = None,
    coco_ann_file: Optional[Path] = None,
    max_images: Optional[int] = None,
):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ExperimentConfig(**ckpt["config"])

    if cfg.sa_gat is not None:
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

    results = {"knn": [], "cop_kmeans": []}
    per_joint = {"knn": {j: [] for j in range(17)},
                 "cop_kmeans": {j: [] for j in range(17)}}

    n_constrained = 0  # how often the constraint changed an assignment
    n_total = 0

    with torch.no_grad():
        for batch in loader:
            graphs = preprocessor.process_batch(batch)
            for graph in graphs:
                if max_images and sum(len(v) for v in results.values()) // 2 >= max_images:
                    break
                graph = graph.to(device)
                embeddings = gat(graph)
                k = int(graph.num_people)
                gt = graph.person_labels

                knn_pred = predict_knn(embeddings, k)
                cop_pred = predict_cop_kmeans(embeddings, k, graph.joint_types)

                results["knn"].append(compute_pga(knn_pred, gt))
                results["cop_kmeans"].append(compute_pga(cop_pred, gt))

                # Count constraint activations
                n_total += len(knn_pred)
                n_constrained += (knn_pred != cop_pred).sum().item()

                # Per-joint
                jt_np = graph.joint_types.cpu().numpy()
                for method, pred in [("knn", knn_pred), ("cop_kmeans", cop_pred)]:
                    pred_np = pred.cpu().numpy()
                    gt_np = gt.cpu().numpy()
                    from scipy.optimize import linear_sum_assignment
                    unique_pred = np.unique(pred_np)
                    unique_gt = np.unique(gt_np)
                    sz = max(len(unique_pred), len(unique_gt))
                    conf = np.zeros((sz, sz), dtype=np.int64)
                    pr = {v: i for i, v in enumerate(unique_pred)}
                    gr = {v: i for i, v in enumerate(unique_gt)}
                    for p, g in zip(pred_np, gt_np):
                        conf[pr[p], gr[g]] += 1
                    row, col = linear_sum_assignment(-conf)
                    pred_to_gt = {unique_pred[r]: unique_gt[c]
                                  for r, c in zip(row, col)
                                  if r < len(unique_pred) and c < len(unique_gt)}
                    for jt_val, p, g in zip(jt_np, pred_np, gt_np):
                        correct = 1.0 if pred_to_gt.get(p) == g else 0.0
                        per_joint[method][jt_val].append(correct)

    # Print results
    knn_mean = sum(results["knn"]) / len(results["knn"])
    cop_mean = sum(results["cop_kmeans"]) / len(results["cop_kmeans"])

    print(f"\n{'='*60}")
    print(f"COP-KMEANS EVALUATION")
    print(f"{'='*60}")
    print(f"kNN PGA:        {knn_mean:.4f}")
    print(f"COP-Kmeans PGA: {cop_mean:.4f}")
    print(f"Difference:     {cop_mean - knn_mean:+.4f}")
    print(f"\nConstraint changed {n_constrained}/{n_total} assignments "
          f"({n_constrained/n_total:.3f})")

    print(f"\n{'Joint':<20}{'kNN':>10}{'COP-KM':>10}{'Diff':>10}")
    print("-" * 50)
    for j in range(17):
        name = COCO_JOINT_NAMES[j]
        knn_acc = sum(per_joint["knn"][j]) / max(len(per_joint["knn"][j]), 1)
        cop_acc = sum(per_joint["cop_kmeans"][j]) / max(len(per_joint["cop_kmeans"][j]), 1)
        diff = cop_acc - knn_acc
        print(f"  {name:<18}{knn_acc:>10.4f}{cop_acc:>10.4f}{diff:>+10.4f}")
    print(f"{'='*60}")

    # Save results next to checkpoint with dataset suffix
    suffix = "coco" if coco_img_dir is not None else "virtual"
    save_path = checkpoint_path.parent / f"eval_cop_kmeans_{suffix}.json"
    per_joint_summary = {}
    for j in range(17):
        name = COCO_JOINT_NAMES[j]
        per_joint_summary[name] = {
            "knn": sum(per_joint["knn"][j]) / max(len(per_joint["knn"][j]), 1),
            "cop_kmeans": sum(per_joint["cop_kmeans"][j]) / max(len(per_joint["cop_kmeans"][j]), 1),
        }
    save_data = {
        "knn_pga": knn_mean, "cop_kmeans_pga": cop_mean,
        "constraint_changed": n_constrained, "total_assignments": n_total,
        "per_joint": per_joint_summary,
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--virtual_dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--coco_img_dir", type=Path, default=None)
    parser.add_argument("--coco_ann_file", type=Path, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate(
        args.checkpoint, args.device,
        virtual_dir=args.virtual_dir, split=args.split,
        coco_img_dir=args.coco_img_dir, coco_ann_file=args.coco_ann_file,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
