"""
End-to-end demo: load a COCO image, run SCOT, draw predicted skeletons.

Produces a side-by-side image:
  Left:  predicted person grouping (SCOT)
  Right: ground truth person grouping

Usage:
    python demo.py --checkpoint outputs/checkpoints/scot_no_depth/best.pt \
                   --coco_img_dir data/coco2017/val2017 \
                   --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json \
                   --output outputs/demo.png

    # Specific image:
    python demo.py --checkpoint ... --img_id 123456

    # Random image with N people:
    python demo.py --checkpoint ... --min_people 3 --max_people 5
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from torchvision.io import read_image

from config import ExperimentConfig
from gat import GATEmbedding
from preprocessor import PosePreprocessor
from dataset import PoseDataset
from dataloader import create_dataloader
from coco_adapter import CocoAdapter
from evaluator import predict_scot, predict_knn, compute_pga


COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Distinct colours for up to 10 people
PERSON_COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def load_model(checkpoint_path: Path, device: str):
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

    head = None
    if ckpt.get("head_state") is not None and cfg.scot is not None:
        from ot_head import SCOTHead
        head = SCOTHead(cfg.scot, embedding_dim=embedding_dim).to(device)
        head.load_state_dict(ckpt["head_state"])
        head.eval()

    k_neighbors = 16 if embedding_dim >= 256 else 8
    preprocessor = PosePreprocessor(
        device=device, k_neighbors=k_neighbors, use_depth=use_depth,
    )

    return gat, head, preprocessor, cfg


def draw_skeleton(ax, keypoints_list, labels, title, image, alpha=0.8):
    """
    Draw skeletons on an axis.

    Args:
        ax: matplotlib axis
        keypoints_list: list of [17, 4] tensors (one per person, in original coords)
        labels: [N] tensor — person label per valid joint (matches preprocessor output)
        title: axis title
        image: [3, H, W] tensor (uint8)
        alpha: skeleton line alpha
    """
    img_np = image.permute(1, 2, 0).cpu().numpy()
    ax.imshow(img_np)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    # Rebuild the mapping from (person_idx, joint_idx) → node index
    # Must match preprocessor logic exactly
    node_idx = 0
    person_joint_to_node = {}
    for person_idx, kps in enumerate(keypoints_list):
        for joint_idx in range(17):
            v = kps[joint_idx, 3].item()
            if v == 0:
                continue
            person_joint_to_node[(person_idx, joint_idx)] = node_idx
            node_idx += 1

    n_people = len(keypoints_list)
    labels_np = labels.cpu().numpy()

    # Assign colours based on labels
    unique_labels = np.unique(labels_np)
    label_to_colour = {
        l: PERSON_COLOURS[i % len(PERSON_COLOURS)]
        for i, l in enumerate(unique_labels)
    }

    # Draw each person's skeleton
    for person_idx, kps in enumerate(keypoints_list):
        kps_np = kps.cpu().numpy()  # [17, 4]

        # Get this person's joints and their predicted labels
        for joint_idx in range(17):
            x, y, z, v = kps_np[joint_idx]
            if v == 0:
                continue

            node = person_joint_to_node.get((person_idx, joint_idx))
            if node is None:
                continue

            colour = label_to_colour[labels_np[node]]
            ax.plot(x, y, "o", color=colour, markersize=4, markeredgecolor="white",
                    markeredgewidth=0.5, zorder=5)

        # Draw skeleton edges
        for j1, j2 in COCO_SKELETON:
            v1 = kps_np[j1, 3]
            v2 = kps_np[j2, 3]
            if v1 == 0 or v2 == 0:
                continue

            node1 = person_joint_to_node.get((person_idx, j1))
            node2 = person_joint_to_node.get((person_idx, j2))
            if node1 is None or node2 is None:
                continue

            # Use the label of the first joint for edge colour
            colour = label_to_colour[labels_np[node1]]
            x1, y1 = kps_np[j1, :2]
            x2, y2 = kps_np[j2, :2]
            ax.plot([x1, x2], [y1, y2], "-", color=colour, linewidth=1.5,
                    alpha=alpha, zorder=4)


def main():
    parser = argparse.ArgumentParser(description="SCOT end-to-end demo")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--coco_img_dir", type=Path, required=True)
    parser.add_argument("--coco_ann_file", type=Path, required=True)
    parser.add_argument("--img_id", type=int, default=None,
                        help="Specific COCO image ID (random if not set)")
    parser.add_argument("--min_people", type=int, default=2)
    parser.add_argument("--max_people", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo.png"))
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── Load model ─────────────────────────────────────────────────────
    print("Loading model...")
    gat, head, preprocessor, cfg = load_model(args.checkpoint, args.device)

    if head is None:
        raise ValueError("Checkpoint does not contain a SCOT head")

    # ── Load COCO adapter ──────────────────────────────────────────────
    adapter = CocoAdapter(
        img_dir=args.coco_img_dir,
        ann_file=args.coco_ann_file,
        min_people=args.min_people,
        max_people=args.max_people,
        device=args.device,
    )
    dataset = PoseDataset(adapter)

    # ── Pick an image ──────────────────────────────────────────────────
    if args.img_id is not None:
        # Find the index for this image ID
        idx = None
        for i, img_id in enumerate(adapter.img_ids):
            if img_id == args.img_id:
                idx = i
                break
        if idx is None:
            raise ValueError(f"Image ID {args.img_id} not found in dataset")
    else:
        idx = random.randint(0, len(dataset) - 1)

    sample = dataset[idx]
    img_id = sample["img_id"]
    n_people = sample["num_people"]
    print(f"Image: {img_id} ({n_people} people)")

    # ── Build graph ────────────────────────────────────────────────────
    # Process single sample through the preprocessor
    keypoints_list = sample["keypoints"]
    graph = preprocessor.create_graph(keypoints_list)

    if graph is None:
        print("Not enough valid joints in this image. Try another.")
        return

    graph = graph.to(args.device)

    # ── Run model ──────────────────────────────────────────────────────
    with torch.no_grad():
        embeddings = gat(graph)
        k = int(graph.num_people)

        # SCOT prediction
        scot_labels = predict_scot(head, embeddings, k, graph.joint_types)

        # kNN prediction for comparison
        knn_labels = predict_knn(embeddings, k)

        # Ground truth
        gt_labels = graph.person_labels

        # Compute PGA
        scot_pga = compute_pga(scot_labels, gt_labels)
        knn_pga = compute_pga(knn_labels, gt_labels)

    print(f"SCOT PGA: {scot_pga:.4f}")
    print(f"kNN  PGA: {knn_pga:.4f}")

    # ── Draw ───────────────────────────────────────────────────────────
    # Use the original (transformed) image for display
    image = sample["image"]  # [3, H, W] uint8

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    draw_skeleton(axes[0], keypoints_list, gt_labels,
                  f"Ground Truth ({n_people} people)", image)
    draw_skeleton(axes[1], keypoints_list, scot_labels,
                  f"SCOT (PGA={scot_pga:.3f})", image)
    draw_skeleton(axes[2], keypoints_list, knn_labels,
                  f"kNN (PGA={knn_pga:.3f})", image)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
