"""
Evaluator for TriGAT (triplet-aware GAT).

Clusters triplet embeddings with kNN / COP-Kmeans / THA, then votes
joint-level cluster assignments and computes PGA against GT joint labels.

Usage:
    python eval_trigat.py --checkpoint outputs/train_trigat/latest/best.pt --virtual_dir data/virtual
    python eval_trigat.py --checkpoint outputs/train_trigat/latest/best.pt --coco_img_dir data/coco2017/val2017 --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
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
from preprocessor import PosePreprocessor
from dataset import PoseDataset
from dataloader import create_dataloader
from virtual_adapter import VirtualAdapter
from evaluator import compute_pga
from trigat import TriGATEmbedding, build_triplet_graph, vote_joint_labels


def predict_knn_on_embeddings(embeddings: torch.Tensor, k: int) -> torch.Tensor:
    emb_np = embeddings.cpu().numpy()
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(emb_np)
    return torch.tensor(labels, device=embeddings.device, dtype=torch.long)


def predict_cop_kmeans_triplets(
    embeddings: torch.Tensor,
    k: int,
    triplet_types: torch.Tensor,
) -> torch.Tensor:
    """
    COP-Kmeans adapted for triplets: must-not-link constraint is that no
    two triplets of the same type can belong to the same cluster (since
    each person has exactly one triplet of each of the 19 types).
    """
    from eval_cop_kmeans import predict_cop_kmeans as _predict
    # Reuse the same constrained k-means — it uses the `joint_types` arg
    # as an opaque "type" used to build must-not-link pairs.
    return _predict(embeddings, k, triplet_types)


def predict_tha_triplets(
    embeddings: torch.Tensor,
    k: int,
    triplet_types: torch.Tensor,
) -> torch.Tensor:
    """THA adapted for triplets — per-type Hungarian matching to K centroids."""
    from eval_hungarian_grouping import predict_tha as _predict
    return _predict(embeddings, k, triplet_types)


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

    if cfg.trigat is None:
        raise ValueError("Checkpoint is not a TriGAT model")

    gat = TriGATEmbedding(cfg.trigat).to(device)
    gat.load_state_dict(ckpt["gat_state"])
    gat.eval()

    # Preprocessor still builds the joint graph (TriGAT derives the
    # triplet graph from it at forward time).
    preprocessor = PosePreprocessor(
        device=device, k_neighbors=8, use_depth=False,
    )

    print(f"Loaded TriGAT checkpoint (epoch {ckpt.get('epoch', '?')})")

    if coco_img_dir is not None and coco_ann_file is not None:
        from coco_adapter import CocoAdapter
        adapter = CocoAdapter(
            img_dir=coco_img_dir, ann_file=coco_ann_file, device=device,
            use_depth=False,
        )
        print(f"Evaluating on COCO: {coco_ann_file.name}")
    elif virtual_dir is not None:
        adapter = VirtualAdapter(virtual_dir / split)
        print(f"Evaluating on virtual/{split}")
    else:
        raise ValueError("Provide --virtual_dir or --coco_img_dir+--coco_ann_file")

    dataset = PoseDataset(adapter)
    loader = create_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)

    results = {"knn": [], "cop_kmeans": [], "tha": []}
    n_eval = 0

    with torch.no_grad():
        for batch in loader:
            joint_graphs = preprocessor.process_batch(batch)
            for graph in joint_graphs:
                if max_images is not None and n_eval >= max_images:
                    break
                graph = graph.to(device)
                k = int(graph.num_people)

                tri_graph = build_triplet_graph(graph, k_neighbors=8)
                if tri_graph is None:
                    continue
                tri_graph = tri_graph.to(device)

                # Skip if fewer triplets than clusters
                if tri_graph.x.size(0) < k:
                    continue

                tri_emb = gat(tri_graph)

                # Three grouping methods on triplet embeddings
                for method_name, predictor in [
                    ("knn",        lambda e, kk, tt: predict_knn_on_embeddings(e, kk)),
                    ("cop_kmeans", predict_cop_kmeans_triplets),
                    ("tha",        predict_tha_triplets),
                ]:
                    tri_labels = predictor(tri_emb, k, tri_graph.triplet_types)
                    joint_labels = vote_joint_labels(
                        tri_labels, tri_graph.joint_pos_in_triplet,
                        tri_graph.num_joints,
                    )
                    pga = compute_pga(joint_labels, graph.person_labels)
                    results[method_name].append(pga)

                n_eval += 1
            if max_images is not None and n_eval >= max_images:
                break

    # Summarise
    n = len(results["knn"])
    print(f"\n{'='*60}")
    print(f"TRIGAT EVALUATION ({n} images)")
    print(f"{'='*60}")
    print(f"{'Method':<25}{'PGA':>10}{'Std':>10}")
    print("-" * 45)
    for method, label in [
        ("knn",        "Triplets + kNN"),
        ("cop_kmeans", "Triplets + COP-Kmeans"),
        ("tha",        "Triplets + THA"),
    ]:
        vals = results[method]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {label:<23}{mean:>10.4f}{std:>10.4f}")
    print(f"{'='*60}")

    suffix = "coco" if coco_img_dir is not None else "virtual"
    save_path = checkpoint_path.parent / f"eval_trigat_{suffix}.json"
    save_data = {
        "n_images": n,
        "pga": {
            m: {"mean": sum(v)/len(v),
                "std": (sum((x - sum(v)/len(v))**2 for x in v)/len(v))**0.5}
            for m, v in results.items() if v
        },
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {save_path}")

    out = {f"{m}_pga": sum(v) / len(v) for m, v in results.items() if v}
    out["n_images"] = n
    return out


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
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Attach eval metrics to this W&B run id. "
                             "Default: read from checkpoint metadata. "
                             "Pass 'none' to skip.")
    args = parser.parse_args()

    metrics = evaluate(
        args.checkpoint, args.device,
        virtual_dir=args.virtual_dir, split=args.split,
        coco_img_dir=args.coco_img_dir, coco_ann_file=args.coco_ann_file,
        max_images=args.max_images,
    )

    # ── Attach to W&B run summary (optional) ──────────────────────────────
    run_id = args.wandb_run_id
    if run_id is None:
        from wandb_helpers import get_wandb_run_id_from_ckpt
        run_id = get_wandb_run_id_from_ckpt(args.checkpoint)
    if run_id and run_id.lower() != "none":
        from wandb_helpers import attach_eval_metrics
        dataset = "coco" if args.coco_img_dir is not None else "synth"
        url = attach_eval_metrics(run_id, f"eval/trigat/{dataset}", metrics)
        if url: print(f"W&B summary updated: {url}")


if __name__ == "__main__":
    main()
