"""
Skeleton-Aware Spectral Clustering evaluation.

Build an affinity matrix from GAT embedding similarity, then zero out
affinities between same-type keypoints (they can never be in the same
cluster). Run spectral clustering on the modified affinity.

The type constraint lives in the graph structure — the Laplacian
encodes that same-type keypoints are disconnected.

No training required — uses existing contrastive GAT checkpoint.

Usage:
    python eval_spectral.py --checkpoint outputs/checkpoints/knn_no_depth/best.pt \
                            --virtual_dir data/virtual

    python eval_spectral.py --checkpoint outputs/checkpoints/knn_no_depth/best.pt \
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
from sklearn.cluster import SpectralClustering, KMeans

from config import ExperimentConfig
from gat import GATEmbedding
from preprocessor import PosePreprocessor
from dataset import PoseDataset
from dataloader import create_dataloader
from virtual_adapter import VirtualAdapter
from evaluator import compute_pga, predict_knn, COCO_JOINT_NAMES


def predict_spectral(
    embeddings: torch.Tensor,
    k: int,
    joint_types: torch.Tensor,
) -> torch.Tensor:
    """
    Skeleton-aware spectral clustering.

    1. Build affinity from embedding cosine similarity (RBF kernel)
    2. Zero out affinities between same-type keypoints
    3. Run spectral clustering on the modified affinity

    Args:
        embeddings:  [N, D] L2-normalised
        k:           number of clusters
        joint_types: [N] joint type indices 0-16

    Returns:
        labels: [N] cluster assignments
    """
    emb_np = embeddings.cpu().numpy()
    jt_np = joint_types.cpu().numpy()
    n = emb_np.shape[0]

    if n <= k:
        return torch.arange(n, device=embeddings.device, dtype=torch.long)

    # Cosine similarity (embeddings are L2-normalised, so dot product = cosine)
    sim = emb_np @ emb_np.T  # [N, N]

    # Convert to non-negative affinity via RBF on distance
    # For L2-normalised vectors: ||a-b||^2 = 2 - 2*cos(a,b)
    sq_dist = 2.0 - 2.0 * sim
    sq_dist = np.clip(sq_dist, 0, None)

    # RBF kernel with median heuristic for gamma
    triu_idx = np.triu_indices(n, k=1)
    median_dist = np.median(sq_dist[triu_idx])
    gamma = 1.0 / (median_dist + 1e-8)

    affinity = np.exp(-gamma * sq_dist)

    # Zero out self-affinities
    np.fill_diagonal(affinity, 0)

    # Type constraint: zero out affinities between same-type keypoints
    for i in range(n):
        for j in range(i + 1, n):
            if jt_np[i] == jt_np[j]:
                affinity[i, j] = 0.0
                affinity[j, i] = 0.0

    # Run spectral clustering on modified affinity
    try:
        sc = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            random_state=42,
            n_init=10,
            assign_labels="kmeans",
        )
        labels = sc.fit_predict(affinity)
    except Exception:
        # Fall back to standard k-means if spectral fails
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(emb_np)

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

    results = {"knn": [], "spectral": []}
    per_joint = {"knn": {j: [] for j in range(17)},
                 "spectral": {j: [] for j in range(17)}}

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
                spectral_pred = predict_spectral(embeddings, k, graph.joint_types)

                results["knn"].append(compute_pga(knn_pred, gt))
                results["spectral"].append(compute_pga(spectral_pred, gt))

                # Per-joint
                jt_np = graph.joint_types.cpu().numpy()
                for method, pred in [("knn", knn_pred), ("spectral", spectral_pred)]:
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
    spectral_mean = sum(results["spectral"]) / len(results["spectral"])

    print(f"\n{'='*60}")
    print(f"SKELETON-AWARE SPECTRAL CLUSTERING")
    print(f"{'='*60}")
    print(f"kNN PGA:      {knn_mean:.4f}")
    print(f"Spectral PGA: {spectral_mean:.4f}")
    print(f"Difference:   {spectral_mean - knn_mean:+.4f}")

    print(f"\n{'Joint':<20}{'kNN':>10}{'Spectral':>10}{'Diff':>10}")
    print("-" * 50)
    for j in range(17):
        name = COCO_JOINT_NAMES[j]
        knn_acc = sum(per_joint["knn"][j]) / max(len(per_joint["knn"][j]), 1)
        spec_acc = sum(per_joint["spectral"][j]) / max(len(per_joint["spectral"][j]), 1)
        diff = spec_acc - knn_acc
        print(f"  {name:<18}{knn_acc:>10.4f}{spec_acc:>10.4f}{diff:>+10.4f}")
    print(f"{'='*60}")

    # Save results next to checkpoint with dataset suffix
    suffix = "coco" if coco_img_dir is not None else "virtual"
    save_path = checkpoint_path.parent / f"eval_spectral_{suffix}.json"
    per_joint_summary = {}
    for j in range(17):
        name = COCO_JOINT_NAMES[j]
        per_joint_summary[name] = {
            "knn": sum(per_joint["knn"][j]) / max(len(per_joint["knn"][j]), 1),
            "spectral": sum(per_joint["spectral"][j]) / max(len(per_joint["spectral"][j]), 1),
        }
    save_data = {"knn_pga": knn_mean, "spectral_pga": spectral_mean, "per_joint": per_joint_summary}
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
