"""
End-to-end demo: raw image → HigherHRNet detection → SA-GAT → COP-Kmeans → skeleton visualisation.

No ground truth used at any point. This is what a deployed system would look like.

Interactive: press any key for next image, 'q' to quit, 's' to save current image.

Shows a 2-panel comparison:
  Left:  HigherHRNet's own AE grouping
  Right: SA-GAT + COP-Kmeans re-grouping

Usage:
    python demo_end_to_end.py \
        --checkpoint outputs/finetune_sa_gat_coco/latest/best.pt \
        --coco_img_dir data/coco2017/val2017 \
        --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json \
        --min_people 3 --max_people 5
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from config import ExperimentConfig
from gat import GATEmbedding
from preprocessor import PosePreprocessor
from evaluator import predict_knn
from eval_cop_kmeans import predict_cop_kmeans
from eval_end_to_end import (
    hrnet_joints_to_detections,
    build_graph_from_detections,
)

sys.path.insert(0, str(Path(__file__).parent / "vendors" / "simple-HigherHRNet"))

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

PERSON_COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def draw_detections(ax, image_rgb, positions, types, labels, title):
    """
    Draw detected skeletons coloured by person assignment.

    Args:
        ax: matplotlib axis
        image_rgb: [H, W, 3] numpy array (RGB)
        positions: [N, 2] (x, y) pixel coords
        types: [N] joint type indices
        labels: [N] person labels
        title: axis title
    """
    ax.imshow(image_rgb)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

    if len(positions) == 0:
        return

    labels_np = labels if isinstance(labels, np.ndarray) else labels.cpu().numpy()
    unique_labels = np.unique(labels_np)
    label_to_colour = {
        l: PERSON_COLOURS[i % len(PERSON_COLOURS)]
        for i, l in enumerate(unique_labels)
    }

    # Build a lookup: (person_label, joint_type) → position
    joint_lookup = {}
    for i in range(len(positions)):
        key = (labels_np[i], int(types[i]))
        joint_lookup[key] = positions[i]

    # Draw joints
    for i in range(len(positions)):
        x, y = positions[i]
        colour = label_to_colour[labels_np[i]]
        ax.plot(x, y, "o", color=colour, markersize=5,
                markeredgecolor="white", markeredgewidth=0.5, zorder=5)

    # Draw skeleton edges
    for person_label in unique_labels:
        colour = label_to_colour[person_label]
        for j1, j2 in COCO_SKELETON:
            pos1 = joint_lookup.get((person_label, j1))
            pos2 = joint_lookup.get((person_label, j2))
            if pos1 is None or pos2 is None:
                continue
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], "-",
                    color=colour, linewidth=2, alpha=0.8, zorder=4)


def process_image(image_bgr, hrnet, gat, preprocessor, device):
    """Run the full pipeline on a single BGR image. Returns None if not enough detections."""
    joints = hrnet.predict(image_bgr)
    det_pos, det_types, hrnet_labels = hrnet_joints_to_detections(joints)

    if len(det_pos) < 2:
        return None

    n_people = len(np.unique(hrnet_labels))

    with torch.no_grad():
        graph = build_graph_from_detections(det_pos, det_types, preprocessor, device)
        if graph is None:
            return None
        graph = graph.to(device)
        embeddings = gat(graph)
        cop_labels = predict_cop_kmeans(embeddings, n_people, graph.joint_types)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return {
        "image_rgb": image_rgb,
        "det_pos": det_pos,
        "det_types": det_types,
        "hrnet_labels": hrnet_labels,
        "cop_labels": cop_labels.cpu().numpy(),
        "n_people": n_people,
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-end pose grouping demo")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="SA-GAT checkpoint")
    parser.add_argument("--coco_img_dir", type=Path, required=True)
    parser.add_argument("--coco_ann_file", type=Path, required=True)
    parser.add_argument("--min_people", type=int, default=2)
    parser.add_argument("--max_people", type=int, default=6)
    parser.add_argument("--hrnet_weights", type=Path,
                        default=Path("vendors/simple-HigherHRNet/weights/pose_higher_hrnet_w32_512.pth"))
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ── Load SA-GAT ───────────────────────────────────────────────────────
    print("Loading SA-GAT...")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ExperimentConfig(**ckpt["config"])

    if cfg.sa_gat is not None:
        from sa_gat import SAGATEmbedding
        gat = SAGATEmbedding(cfg.sa_gat).to(args.device)
        embedding_dim = cfg.sa_gat.output_dim
        use_depth = cfg.sa_gat.use_depth
    else:
        gat = GATEmbedding(cfg.gat).to(args.device)
        embedding_dim = cfg.gat.output_dim
        use_depth = cfg.gat.use_depth
    gat.load_state_dict(ckpt["gat_state"])
    gat.eval()

    k_neighbors = 16 if embedding_dim >= 256 else 8
    preprocessor = PosePreprocessor(
        device=args.device, k_neighbors=k_neighbors, use_depth=use_depth,
    )

    # ── Load HigherHRNet ─────────────────────────────────────────────────
    print("Loading HigherHRNet...")
    from SimpleHigherHRNet import SimpleHigherHRNet
    hrnet = SimpleHigherHRNet(
        c=32, nof_joints=17,
        checkpoint_path=str(args.hrnet_weights),
        resolution=512,
        device=torch.device(args.device),
    )

    # ── Build image list ─────────────────────────────────────────────────
    from pycocotools.coco import COCO
    print("Loading COCO annotations...")
    coco = COCO(str(args.coco_ann_file))
    cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = sorted(coco.getImgIds(catIds=cat_ids))

    valid = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        n = sum(1 for a in anns
                if sum(1 for v in a["keypoints"][2::3] if v > 0) >= 3)
        if args.min_people <= n <= args.max_people:
            valid.append(img_id)

    random.shuffle(valid)
    print(f"Found {len(valid)} images with {args.min_people}-{args.max_people} people")
    print("Controls: any key = next | s = save | q = quit\n")

    # ── Interactive loop ─────────────────────────────────────────────────
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    save_count = 0
    last_key = [None]

    def on_key(event):
        last_key[0] = event.key

    fig.canvas.mpl_connect("key_press_event", on_key)

    for img_id in valid:
        img_info = coco.loadImgs(img_id)[0]
        image_path = args.coco_img_dir / img_info["file_name"]
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        result = process_image(image_bgr, hrnet, gat, preprocessor, args.device)
        if result is None:
            continue

        for ax in axes:
            ax.clear()

        draw_detections(
            axes[0], result["image_rgb"], result["det_pos"], result["det_types"],
            result["hrnet_labels"],
            f"HigherHRNet AE ({result['n_people']} people)"
        )
        draw_detections(
            axes[1], result["image_rgb"], result["det_pos"], result["det_types"],
            result["cop_labels"],
            f"SA-GAT + COP-Kmeans ({result['n_people']} people)"
        )

        fig.suptitle(
            f"{img_info['file_name']}  (id={img_id})    [any key=next | s=save | q=quit]",
            fontsize=10, color="gray",
        )
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show(block=False)

        # Wait for keypress
        last_key[0] = None
        while last_key[0] is None:
            if not plt.fignum_exists(fig.number):
                print("Window closed.")
                return
            plt.pause(0.05)

        if last_key[0] == "q":
            print("Quit.")
            break
        elif last_key[0] == "s":
            save_count += 1
            save_path = Path("outputs") / f"demo_e2e_{save_count:03d}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
            # Wait for another key to advance
            last_key[0] = None
            while last_key[0] is None:
                if not plt.fignum_exists(fig.number):
                    return
                plt.pause(0.05)
            if last_key[0] == "q":
                break

    plt.ioff()
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
