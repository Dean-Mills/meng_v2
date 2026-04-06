"""
K estimation via eigenvalue gap on SA-GAT embeddings.

Computes the affinity matrix from GAT embeddings, builds the graph
Laplacian, and estimates K from the largest gap between consecutive
eigenvalues. The number of near-zero eigenvalues equals the number
of connected components (people).

Then evaluates grouping with predicted K vs ground truth K using
both COP-Kmeans and (optionally) SCOT.

Usage:
    python eval_k_estimation.py --checkpoint outputs/checkpoints/sa_gat_full/best.pt \
                                --virtual_dir data/virtual

    python eval_k_estimation.py --checkpoint outputs/checkpoints/sa_gat_full/best.pt \
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
from eval_cop_kmeans import predict_cop_kmeans


def estimate_k_eigengap(
    embeddings: torch.Tensor,
    k_max: int = 15,
) -> int:
    """
    Estimate number of clusters from the eigenvalue gap of the
    graph Laplacian built from embedding similarity.

    Args:
        embeddings: [N, D] L2-normalised
        k_max:      maximum K to consider

    Returns:
        k_est: estimated number of clusters
    """
    emb_np = embeddings.cpu().numpy()
    n = emb_np.shape[0]

    if n <= 2:
        return 1

    # Cosine similarity (L2-normalised → dot product)
    sim = emb_np @ emb_np.T

    # RBF affinity with median heuristic
    sq_dist = 2.0 - 2.0 * sim
    sq_dist = np.clip(sq_dist, 0, None)

    triu_idx = np.triu_indices(n, k=1)
    median_dist = np.median(sq_dist[triu_idx])
    gamma = 1.0 / (median_dist + 1e-8)

    W = np.exp(-gamma * sq_dist)
    np.fill_diagonal(W, 0)

    # Graph Laplacian: L = D - W
    D = np.diag(W.sum(axis=1))
    L = D - W

    # Compute smallest eigenvalues
    k_compute = min(k_max + 1, n - 1)
    try:
        from scipy.sparse.linalg import eigsh
        eigenvalues = eigsh(L, k=k_compute, which='SM', return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)
    except Exception:
        # Fallback to dense eigendecomposition
        eigenvalues = np.linalg.eigvalsh(L)
        eigenvalues = np.sort(eigenvalues)[:k_compute]

    # Find the largest gap between consecutive eigenvalues
    # Skip the first eigenvalue (always ~0 for connected graph)
    # K = index of the largest gap
    if len(eigenvalues) < 3:
        return 1

    gaps = np.diff(eigenvalues[1:])  # gaps between eigenvalue 1-2, 2-3, etc.

    if len(gaps) == 0:
        return 1

    # K = position of largest gap + 1 (because we skipped eigenvalue 0)
    # eigenvalue 0 is trivial, eigenvalues 1..K should be near-zero,
    # then a big gap at position K
    k_est = np.argmax(gaps) + 2  # +2 because: +1 for skipping ev0, +1 for 0-indexing

    # Clamp to reasonable range
    k_est = max(1, min(k_est, k_max))

    return int(k_est)


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

    # Load K head if present
    k_head_model = None
    if ckpt.get("k_head_state") is not None:
        from k_head import KEstimationHead
        k_head_model = KEstimationHead(embedding_dim=embedding_dim).to(device)
        k_head_model.load_state_dict(ckpt["k_head_state"])
        k_head_model.eval()
        print("Loaded learned K estimation head")

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

    # Track K estimation accuracy
    k_exact = 0
    k_off_by_one = 0
    k_total = 0
    k_errors = []  # (predicted, actual)

    # Track PGA with GT K vs predicted K
    results_gt = {"knn": [], "cop_km": []}
    results_pred = {"knn": [], "cop_km": []}

    with torch.no_grad():
        for batch in loader:
            graphs = preprocessor.process_batch(batch)
            for graph in graphs:
                if max_images and k_total >= max_images:
                    break
                graph = graph.to(device)
                embeddings = gat(graph)
                k_gt = int(graph.num_people)
                gt = graph.person_labels

                # Estimate K
                if k_head_model is not None:
                    k_pred = k_head_model.predict(embeddings)
                else:
                    k_pred = estimate_k_eigengap(embeddings)

                # Track K accuracy
                k_total += 1
                if k_pred == k_gt:
                    k_exact += 1
                if abs(k_pred - k_gt) <= 1:
                    k_off_by_one += 1
                k_errors.append((k_pred, k_gt))

                # Grouping with GT K
                knn_gt = predict_knn(embeddings, k_gt)
                cop_gt = predict_cop_kmeans(embeddings, k_gt, graph.joint_types)
                results_gt["knn"].append(compute_pga(knn_gt, gt))
                results_gt["cop_km"].append(compute_pga(cop_gt, gt))

                # Grouping with predicted K
                k_use = max(1, k_pred)
                knn_pred = predict_knn(embeddings, k_use)
                cop_pred = predict_cop_kmeans(embeddings, k_use, graph.joint_types)
                results_pred["knn"].append(compute_pga(knn_pred, gt))
                results_pred["cop_km"].append(compute_pga(cop_pred, gt))

    # Results
    print(f"\n{'='*65}")
    print(f"K ESTIMATION EVALUATION")
    print(f"{'='*65}")

    print(f"\nK estimation accuracy ({k_total} scenes):")
    print(f"  Exact match:   {k_exact}/{k_total} ({k_exact/k_total:.3f})")
    print(f"  Off by <=1:    {k_off_by_one}/{k_total} ({k_off_by_one/k_total:.3f})")

    # K distribution
    from collections import Counter
    gt_counts = Counter(gt_k for _, gt_k in k_errors)
    print(f"\n  Per GT K breakdown:")
    for k_val in sorted(gt_counts.keys()):
        scenes = [(p, g) for p, g in k_errors if g == k_val]
        exact = sum(1 for p, g in scenes if p == g)
        print(f"    K={k_val}: {exact}/{len(scenes)} exact "
              f"({exact/len(scenes):.2f}), "
              f"predictions: {Counter(p for p, _ in scenes).most_common(3)}")

    # PGA comparison
    print(f"\nGrouping PGA:")
    print(f"  {'Method':<25}{'GT K':>10}{'Pred K':>10}{'Diff':>10}")
    print(f"  {'-'*55}")

    for method in ["knn", "cop_km"]:
        gt_mean = sum(results_gt[method]) / len(results_gt[method])
        pred_mean = sum(results_pred[method]) / len(results_pred[method])
        diff = pred_mean - gt_mean
        label = "kNN" if method == "knn" else "COP-Kmeans"
        print(f"  {label:<25}{gt_mean:>10.4f}{pred_mean:>10.4f}{diff:>+10.4f}")

    print(f"{'='*65}")

    # Save results next to checkpoint with dataset suffix
    suffix = "coco" if coco_img_dir is not None else "virtual"
    save_path = checkpoint_path.parent / f"eval_k_estimation_{suffix}.json"
    save_data = {
        "k_exact": k_exact, "k_off_by_one": k_off_by_one, "k_total": k_total,
        "pga": {},
    }
    for method in ["knn", "cop_km"]:
        save_data["pga"][method] = {
            "gt_k": sum(results_gt[method]) / len(results_gt[method]),
            "pred_k": sum(results_pred[method]) / len(results_pred[method]),
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
