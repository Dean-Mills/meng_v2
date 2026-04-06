"""
Type-Constrained Agglomerative Clustering evaluation.

Bottom-up clustering: start with each keypoint as its own cluster.
At each step, merge the two closest clusters — but never merge two
clusters that would create duplicate joint types. Stop when K clusters
remain.

No training required — uses existing contrastive GAT checkpoint.

Usage:
    python eval_agglomerative.py --checkpoint outputs/checkpoints/knn_no_depth/best.pt \
                                 --virtual_dir data/virtual

    python eval_agglomerative.py --checkpoint outputs/checkpoints/knn_no_depth/best.pt \
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

from config import ExperimentConfig
from gat import GATEmbedding
from preprocessor import PosePreprocessor
from dataset import PoseDataset
from dataloader import create_dataloader
from virtual_adapter import VirtualAdapter
from evaluator import compute_pga, predict_knn, COCO_JOINT_NAMES


def predict_agglomerative(
    embeddings: torch.Tensor,
    k: int,
    joint_types: torch.Tensor,
) -> torch.Tensor:
    """
    Type-constrained agglomerative clustering.

    Bottom-up: start with N clusters (one per keypoint), merge closest
    pair at each step unless merging would create duplicate joint types.
    Stop at K clusters.

    Uses average linkage (distance between clusters = mean pairwise
    distance between their members).

    Args:
        embeddings:  [N, D] L2-normalised
        k:           target number of clusters
        joint_types: [N] joint type indices 0-16

    Returns:
        labels: [N] cluster assignments
    """
    emb_np = embeddings.cpu().numpy()
    jt_np = joint_types.cpu().numpy()
    n = emb_np.shape[0]

    # Pairwise distance matrix
    dist_matrix = np.linalg.norm(
        emb_np[:, None, :] - emb_np[None, :, :], axis=2
    )

    # Initialise: each point is its own cluster
    # cluster_members[c] = list of point indices in cluster c
    cluster_members = {i: [i] for i in range(n)}
    # cluster_types[c] = set of joint types in cluster c
    cluster_types = {i: {jt_np[i]} for i in range(n)}
    # active clusters
    active = set(range(n))

    # Precompute inter-cluster distances (average linkage)
    # For efficiency, maintain a distance cache
    cluster_dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            cluster_dist[(i, j)] = dist_matrix[i, j]

    n_clusters = n

    while n_clusters > k:
        # Find closest mergeable pair
        best_dist = float("inf")
        best_pair = None

        for ci in active:
            for cj in active:
                if ci >= cj:
                    continue

                # Check type constraint: no duplicate types after merge
                merged_types = cluster_types[ci] | cluster_types[cj]
                if len(merged_types) < len(cluster_types[ci]) + len(cluster_types[cj]):
                    # There's a type overlap — skip this pair
                    continue

                # Get distance
                key = (ci, cj) if ci < cj else (cj, ci)
                d = cluster_dist.get(key, float("inf"))

                if d < best_dist:
                    best_dist = d
                    best_pair = (ci, cj)

        if best_pair is None:
            # No valid merges possible — all remaining merges violate constraints
            break

        ci, cj = best_pair

        # Merge cj into ci
        new_members = cluster_members[ci] + cluster_members[cj]
        cluster_members[ci] = new_members
        cluster_types[ci] = cluster_types[ci] | cluster_types[cj]

        # Update distances to merged cluster (average linkage)
        for ck in active:
            if ck == ci or ck == cj:
                continue
            key_i = (min(ci, ck), max(ci, ck))
            key_j = (min(cj, ck), max(cj, ck))

            # Average linkage: mean of all pairwise distances
            total_dist = 0.0
            count = 0
            for mi in cluster_members[ci]:
                for mk in cluster_members[ck]:
                    total_dist += dist_matrix[mi, mk]
                    count += 1
            new_dist = total_dist / count if count > 0 else float("inf")

            new_key = (min(ci, ck), max(ci, ck))
            cluster_dist[new_key] = new_dist

            # Remove old cj distances
            old_key = (min(cj, ck), max(cj, ck))
            cluster_dist.pop(old_key, None)

        # Remove cj
        del cluster_members[cj]
        del cluster_types[cj]
        active.remove(cj)
        n_clusters -= 1

    # Build labels
    labels = np.zeros(n, dtype=np.int64)
    for label_idx, (cluster_id, members) in enumerate(cluster_members.items()):
        for m in members:
            labels[m] = label_idx

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
        )
        print(f"Evaluating on COCO: {coco_ann_file.name}")
    elif virtual_dir is not None:
        adapter = VirtualAdapter(virtual_dir / split)
        print(f"Evaluating on virtual/{split}")
    else:
        raise ValueError("Provide --virtual_dir or --coco_img_dir + --coco_ann_file")

    dataset = PoseDataset(adapter)
    loader = create_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)

    results = {"knn": [], "agglom": []}
    per_joint = {"knn": {j: [] for j in range(17)},
                 "agglom": {j: [] for j in range(17)}}

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
                agglom_pred = predict_agglomerative(embeddings, k, graph.joint_types)

                results["knn"].append(compute_pga(knn_pred, gt))
                results["agglom"].append(compute_pga(agglom_pred, gt))

                # Per-joint
                jt_np = graph.joint_types.cpu().numpy()
                for method, pred in [("knn", knn_pred), ("agglom", agglom_pred)]:
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
    agglom_mean = sum(results["agglom"]) / len(results["agglom"])

    print(f"\n{'='*60}")
    print(f"TYPE-CONSTRAINED AGGLOMERATIVE CLUSTERING")
    print(f"{'='*60}")
    print(f"kNN PGA:         {knn_mean:.4f}")
    print(f"Agglomerative PGA: {agglom_mean:.4f}")
    print(f"Difference:      {agglom_mean - knn_mean:+.4f}")

    print(f"\n{'Joint':<20}{'kNN':>10}{'Agglom':>10}{'Diff':>10}")
    print("-" * 50)
    for j in range(17):
        name = COCO_JOINT_NAMES[j]
        knn_acc = sum(per_joint["knn"][j]) / max(len(per_joint["knn"][j]), 1)
        agglom_acc = sum(per_joint["agglom"][j]) / max(len(per_joint["agglom"][j]), 1)
        diff = agglom_acc - knn_acc
        print(f"  {name:<18}{knn_acc:>10.4f}{agglom_acc:>10.4f}{diff:>+10.4f}")
    print(f"{'='*60}")

    # Save results next to checkpoint with dataset suffix
    suffix = "coco" if coco_img_dir is not None else "virtual"
    save_path = checkpoint_path.parent / f"eval_agglomerative_{suffix}.json"
    per_joint_summary = {}
    for j in range(17):
        name = COCO_JOINT_NAMES[j]
        per_joint_summary[name] = {
            "knn": sum(per_joint["knn"][j]) / max(len(per_joint["knn"][j]), 1),
            "agglom": sum(per_joint["agglom"][j]) / max(len(per_joint["agglom"][j]), 1),
        }
    save_data = {"knn_pga": knn_mean, "agglom_pga": agglom_mean, "per_joint": per_joint_summary}
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
