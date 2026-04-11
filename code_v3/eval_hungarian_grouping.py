"""
Type-wise Hungarian Assignment (THA) for pose grouping.

For each joint type independently, runs Hungarian (optimal bipartite)
matching between that type's detected joints and K person centroids.
Centroids are initialised from k-means and updated iteratively.

This gives globally optimal per-type assignment with type constraints
by construction, avoiding COP-Kmeans' greedy cascade errors.

Usage:
    python eval_hungarian_grouping.py \
        --checkpoint outputs/finetune_sa_gat_coco/latest/best.pt \
        --virtual_dir data/virtual

    python eval_hungarian_grouping.py \
        --checkpoint outputs/finetune_sa_gat_coco/latest/best.pt \
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

COCO_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def predict_tha(
    embeddings: torch.Tensor,
    k: int,
    joint_types: torch.Tensor,
    n_iters: int = 10,
) -> torch.Tensor:
    """
    Type-wise Hungarian Assignment.

    Args:
        embeddings: [N, D] L2-normalised joint embeddings
        k: number of people
        joint_types: [N] joint type indices 0-16
        n_iters: number of centroid update iterations

    Returns:
        labels: [N] person assignment labels
    """
    emb_np = embeddings.cpu().numpy()
    jt_np = joint_types.cpu().numpy()
    n = len(emb_np)

    # Initialise centroids from k-means
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(emb_np)
    centroids = km.cluster_centers_.copy()  # [K, D]

    labels = np.zeros(n, dtype=int)

    for iteration in range(n_iters):
        # ── Per-type Hungarian assignment ─────────────────────────────
        for jt in range(17):
            mask = jt_np == jt
            if mask.sum() == 0:
                continue

            indices = np.where(mask)[0]
            type_embs = emb_np[indices]  # [M, D]

            # Cost matrix: distance from each joint of this type to each centroid
            # [M, K]
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
                # More joints than people — extend cost matrix
                # Add dummy columns so it's square, then take first K cols
                # Or: transpose and match people to joints
                cost_t = cost.T  # [K, M]
                row, col = linear_sum_assignment(cost_t)
                assigned = set()
                for r, c in zip(row, col):
                    labels[indices[c]] = r
                    assigned.add(c)
                # Unassigned joints go to nearest centroid
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

        # Check convergence
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return torch.tensor(labels, device=embeddings.device, dtype=torch.long)


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

    results = {"knn": [], "cop_kmeans": [], "tha": []}

    with torch.no_grad():
        for batch in loader:
            graphs = preprocessor.process_batch(batch)
            for graph in graphs:
                if max_images and len(results["knn"]) >= max_images:
                    break
                graph = graph.to(device)
                embeddings = gat(graph)
                k = int(graph.num_people)
                gt = graph.person_labels

                knn_pred = predict_knn(embeddings, k)
                cop_pred = predict_cop_kmeans(embeddings, k, graph.joint_types)
                tha_pred = predict_tha(embeddings, k, graph.joint_types)

                results["knn"].append(compute_pga(knn_pred, gt))
                results["cop_kmeans"].append(compute_pga(cop_pred, gt))
                results["tha"].append(compute_pga(tha_pred, gt))

    n = len(results["knn"])
    print(f"\n{'='*60}")
    print(f"TYPE-WISE HUNGARIAN ASSIGNMENT ({n} images)")
    print(f"{'='*60}")
    print(f"{'Method':<25}{'PGA':>10}{'Std':>10}")
    print("-" * 45)
    for method, label in [
        ("knn", "kNN"),
        ("cop_kmeans", "COP-Kmeans"),
        ("tha", "THA"),
    ]:
        vals = results[method]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {label:<23}{mean:>10.4f}{std:>10.4f}")
    print(f"{'='*60}")

    # Save
    suffix = "coco" if coco_img_dir is not None else "virtual"
    save_path = checkpoint_path.parent / f"eval_tha_{suffix}.json"
    save_data = {
        "n_images": n,
        "pga": {
            m: {"mean": sum(v)/len(v),
                "std": (sum((x - sum(v)/len(v))**2 for x in v)/len(v))**0.5}
            for m, v in results.items()
        },
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
