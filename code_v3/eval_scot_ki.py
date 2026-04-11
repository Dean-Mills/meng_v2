"""
Evaluate SCOT with k-means initialised prototypes (SCOT-KI).

Uses a trained SA-GAT checkpoint but does NOT need a trained SCOT head.
The node encoder can be randomly initialised or loaded from a trained
SCOT checkpoint.

Usage:
    # With random node encoder (just to test the OT assignment):
    python eval_scot_ki.py \
        --checkpoint outputs/finetune_sa_gat_coco/latest/best.pt \
        --virtual_dir data/virtual

    # On COCO:
    python eval_scot_ki.py \
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

from config import ExperimentConfig, SCOTConfig
from gat import GATEmbedding
from preprocessor import PosePreprocessor
from dataset import PoseDataset
from dataloader import create_dataloader
from virtual_adapter import VirtualAdapter
from evaluator import compute_pga, predict_knn
from eval_cop_kmeans import predict_cop_kmeans
from ot_head_kmeans_init import SCOTKmeansInitHead

COCO_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def predict_scot_ki(
    head: SCOTKmeansInitHead,
    embeddings: torch.Tensor,
    k: int,
    joint_types: torch.Tensor,
) -> torch.Tensor:
    logits, T = head(embeddings, k, joint_types)
    return logits.argmax(dim=1)


def evaluate(
    checkpoint_path: Path,
    device: str,
    virtual_dir: Optional[Path] = None,
    split: str = "test",
    coco_img_dir: Optional[Path] = None,
    coco_ann_file: Optional[Path] = None,
    max_images: Optional[int] = None,
    skip_encoder: bool = False,
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

    # SCOT-KI head — node encoder can be identity if skip_encoder
    scot_cfg = SCOTConfig(
        hidden_dim=embedding_dim if skip_encoder else 256,
        k_max=20,
        sinkhorn_iters=10,
        sinkhorn_tau=0.1,
    )
    scot_ki = SCOTKmeansInitHead(scot_cfg, embedding_dim=embedding_dim).to(device)

    if skip_encoder:
        # Replace node encoder with identity — use raw embeddings
        scot_ki.node_encoder = torch.nn.Identity()

    scot_ki.eval()

    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
    print(f"Node encoder: {'identity (skip)' if skip_encoder else 'random init'}")

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

    results = {"knn": [], "cop_kmeans": [], "scot_ki": []}

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
                ki_pred = predict_scot_ki(scot_ki, embeddings, k, graph.joint_types)

                results["knn"].append(compute_pga(knn_pred, gt))
                results["cop_kmeans"].append(compute_pga(cop_pred, gt))
                results["scot_ki"].append(compute_pga(ki_pred, gt))

    n = len(results["knn"])
    print(f"\n{'='*60}")
    print(f"SCOT-KI EVALUATION ({n} images)")
    print(f"{'='*60}")
    print(f"{'Method':<25}{'PGA':>10}{'Std':>10}")
    print("-" * 45)
    for method, label in [
        ("knn", "kNN"),
        ("cop_kmeans", "COP-Kmeans"),
        ("scot_ki", "SCOT-KI"),
    ]:
        vals = results[method]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {label:<23}{mean:>10.4f}{std:>10.4f}")
    print(f"{'='*60}")

    # Save
    suffix = "coco" if coco_img_dir is not None else "virtual"
    save_path = checkpoint_path.parent / f"eval_scot_ki_{suffix}.json"
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
    parser.add_argument("--skip_encoder", action="store_true",
                        help="Use identity encoder (raw embeddings → k-means → Sinkhorn)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate(
        args.checkpoint, args.device,
        virtual_dir=args.virtual_dir, split=args.split,
        coco_img_dir=args.coco_img_dir, coco_ann_file=args.coco_ann_file,
        max_images=args.max_images, skip_encoder=args.skip_encoder,
    )


if __name__ == "__main__":
    main()
