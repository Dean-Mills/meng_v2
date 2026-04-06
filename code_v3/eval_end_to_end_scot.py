"""
End-to-end evaluation: HigherHRNet → SA-GAT → K-head → SCOT → PGA.

The full autonomous pipeline with no GT K dependency.

Uses two checkpoints:
  1. SA-GAT + SCOT checkpoint (sa_gat_scot_no_depth)
  2. K-head checkpoint (k_head_frozen) — trained on sa_gat_full embeddings,
     so K predictions may be approximate on sa_gat_scot embeddings.

Compares grouping methods:
  1. HigherHRNet's own AE grouping
  2. SA-GAT + SCOT (with GT K, for reference)
  3. SA-GAT + SCOT (with predicted K)
  4. SA-GAT + COP-Kmeans (with predicted K)

Usage:
    python eval_end_to_end_scot.py \
        --scot_checkpoint outputs/frozen_checkpoints/sa_gat_scot_no_depth/best.pt \
        --k_head_checkpoint outputs/frozen_checkpoints/k_head_frozen/best.pt \
        --coco_img_dir data/coco2017/val2017 \
        --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from config import ExperimentConfig
from gat import GATEmbedding
from preprocessor import PosePreprocessor
from evaluator import compute_pga, predict_knn, predict_scot

sys.path.insert(0, str(Path(__file__).parent / "vendors" / "simple-HigherHRNet"))

MATCH_THRESHOLD = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Reuse helpers from eval_end_to_end
# ─────────────────────────────────────────────────────────────────────────────

from eval_end_to_end import (
    match_detections_to_gt,
    hrnet_joints_to_detections,
    build_graph_from_detections,
)
from eval_cop_kmeans import predict_cop_kmeans


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    scot_checkpoint_path: Path,
    k_head_checkpoint_path: Path,
    coco_img_dir: Path,
    coco_ann_file: Path,
    hrnet_weights: Path,
    device: str,
    max_images: Optional[int] = None,
    hrnet_device: str = "cpu",
):
    # ── Load SA-GAT + SCOT ───────────────────────────────────────────────
    ckpt = torch.load(scot_checkpoint_path, map_location=device, weights_only=False)
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

    # Load SCOT head
    assert cfg.scot is not None, "SCOT checkpoint must contain a SCOT config"
    from ot_head import SCOTHead
    scot_head = SCOTHead(cfg.scot, embedding_dim=embedding_dim).to(device)
    scot_head.load_state_dict(ckpt["head_state"])
    scot_head.eval()

    print(f"Loaded SA-GAT + SCOT (epoch {ckpt.get('epoch', '?')}, "
          f"val PGA {ckpt.get('val_pga', 0):.4f})")

    # ── Load K-head ──────────────────────────────────────────────────────
    k_ckpt = torch.load(k_head_checkpoint_path, map_location=device, weights_only=False)
    from k_head import KEstimationHead
    k_head = KEstimationHead(embedding_dim=embedding_dim).to(device)
    k_head.load_state_dict(k_ckpt["k_head_state"])
    k_head.eval()

    print(f"Loaded K-head (val accuracy {k_ckpt.get('val_k_accuracy', '?')})")
    print(f"K-head embedding dim: {embedding_dim}")

    k_neighbors = 16 if embedding_dim >= 256 else 8
    preprocessor = PosePreprocessor(
        device=device, k_neighbors=k_neighbors, use_depth=use_depth,
    )

    # ── Load HigherHRNet ─────────────────────────────────────────────────
    from SimpleHigherHRNet import SimpleHigherHRNet

    hrnet = SimpleHigherHRNet(
        c=32, nof_joints=17,
        checkpoint_path=str(hrnet_weights),
        resolution=512,
        device=torch.device(hrnet_device),
    )
    print(f"Loaded HigherHRNet w32-512 on {hrnet_device}")

    # ── Load COCO annotations ────────────────────────────────────────────
    from pycocotools.coco import COCO
    print("Loading COCO annotations...")
    coco = COCO(str(coco_ann_file))

    cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = sorted(coco.getImgIds(catIds=cat_ids))

    valid_img_ids = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        people_with_kps = []
        for ann in anns:
            kps = np.array(ann["keypoints"]).reshape(17, 3)
            if (kps[:, 2] > 0).sum() >= 3:
                people_with_kps.append(ann)
        if len(people_with_kps) >= 1:
            valid_img_ids.append(img_id)

    print(f"COCO: {len(valid_img_ids)} valid images")

    if max_images:
        valid_img_ids = valid_img_ids[:max_images]

    # ── Evaluate ─────────────────────────────────────────────────────────
    results = {
        "hrnet_ae": [],
        "scot_gt_k": [],
        "scot_pred_k": [],
        "cop_km_pred_k": [],
    }
    k_stats = {"exact": 0, "off_by_one": 0, "total": 0}
    detection_stats = {"total_gt": 0, "total_matched": 0, "total_detected": 0}

    with torch.no_grad():
        for i, img_id in enumerate(valid_img_ids):
            img_info = coco.loadImgs(img_id)[0]
            img_path = coco_img_dir / img_info["file_name"]
            image = cv2.imread(str(img_path))

            if image is None:
                continue

            # ── HigherHRNet detection ─────────────────────────────────
            joints = hrnet.predict(image)
            det_pos, det_types, hrnet_person_ids = hrnet_joints_to_detections(joints)

            if len(det_pos) < 2:
                continue

            # ── Get GT keypoints ──────────────────────────────────────
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
            anns = coco.loadAnns(ann_ids)

            gt_kps_list = []
            for ann in anns:
                kps = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)
                kps_4 = np.zeros((17, 4), dtype=np.float32)
                kps_4[:, 0] = kps[:, 0]
                kps_4[:, 1] = kps[:, 1]
                kps_4[:, 2] = 0.0
                kps_4[:, 3] = kps[:, 2]
                if (kps[:, 2] > 0).sum() >= 3:
                    gt_kps_list.append(kps_4)

            if len(gt_kps_list) < 1:
                continue

            # ── Match detections to GT ────────────────────────────────
            matched_det_idx, matched_gt_person, matched_gt_type = \
                match_detections_to_gt(det_pos, det_types, gt_kps_list)

            if len(matched_det_idx) < 2:
                continue

            n_gt_kps = sum((kps[:, 3] > 0).sum() for kps in gt_kps_list)
            detection_stats["total_gt"] += n_gt_kps
            detection_stats["total_matched"] += len(matched_det_idx)
            detection_stats["total_detected"] += len(det_pos)

            n_gt_people = len(gt_kps_list)
            gt_labels = torch.tensor(matched_gt_person, dtype=torch.long)

            # ── 1. HigherHRNet AE grouping ────────────────────────────
            hrnet_labels = torch.tensor(
                hrnet_person_ids[matched_det_idx], dtype=torch.long
            )
            results["hrnet_ae"].append(compute_pga(hrnet_labels, gt_labels))

            # ── Build graph and get embeddings ────────────────────────
            matched_pos = det_pos[matched_det_idx]
            matched_types = det_types[matched_det_idx]

            graph = build_graph_from_detections(
                matched_pos, matched_types, preprocessor, device,
            )
            if graph is None:
                continue

            graph = graph.to(device)
            embeddings = gat(graph)

            if len(matched_det_idx) < n_gt_people:
                continue

            # ── 2. SCOT with GT K ─────────────────────────────────────
            scot_pred = predict_scot(
                scot_head, embeddings, n_gt_people, graph.joint_types,
            )
            results["scot_gt_k"].append(
                compute_pga(scot_pred, gt_labels.to(device))
            )

            # ── 3. Predict K ──────────────────────────────────────────
            k_pred = k_head.predict(embeddings)
            k_pred = max(1, k_pred)  # at least 1 person

            k_stats["total"] += 1
            if k_pred == n_gt_people:
                k_stats["exact"] += 1
            if abs(k_pred - n_gt_people) <= 1:
                k_stats["off_by_one"] += 1

            # ── 4. SCOT with predicted K ──────────────────────────────
            k_for_scot = min(k_pred, len(matched_det_idx) // 17 + 1)
            k_for_scot = max(1, k_for_scot)
            scot_pred_k = predict_scot(
                scot_head, embeddings, k_for_scot, graph.joint_types,
            )
            results["scot_pred_k"].append(
                compute_pga(scot_pred_k, gt_labels.to(device))
            )

            # ── 5. COP-Kmeans with predicted K ───────────────────────
            k_for_cop = min(k_pred, len(matched_det_idx))
            k_for_cop = max(1, k_for_cop)
            cop_pred_k = predict_cop_kmeans(
                embeddings, k_for_cop, graph.joint_types,
            )
            results["cop_km_pred_k"].append(
                compute_pga(cop_pred_k, gt_labels.to(device))
            )

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(valid_img_ids)} images processed...")

    # ── Print results ────────────────────────────────────────────────────
    n = len(results["hrnet_ae"])
    if n == 0:
        print("No valid images processed!")
        return

    print(f"\n{'='*65}")
    print(f"END-TO-END EVALUATION — FULL PIPELINE ({n} images)")
    print(f"{'='*65}")

    recall = detection_stats["total_matched"] / max(detection_stats["total_gt"], 1)
    precision = detection_stats["total_matched"] / max(detection_stats["total_detected"], 1)
    print(f"\nDetection stats:")
    print(f"  GT keypoints:       {detection_stats['total_gt']}")
    print(f"  Detected keypoints: {detection_stats['total_detected']}")
    print(f"  Matched keypoints:  {detection_stats['total_matched']}")
    print(f"  Recall:             {recall:.3f}")
    print(f"  Precision:          {precision:.3f}")

    if k_stats["total"] > 0:
        print(f"\nK estimation:")
        print(f"  Exact match:  {k_stats['exact']}/{k_stats['total']} "
              f"({k_stats['exact']/k_stats['total']:.3f})")
        print(f"  Off by <=1:   {k_stats['off_by_one']}/{k_stats['total']} "
              f"({k_stats['off_by_one']/k_stats['total']:.3f})")

    print(f"\n{'Method':<35}{'PGA':>10}{'Std':>10}")
    print("-" * 55)
    for method, label in [
        ("hrnet_ae", "HigherHRNet AE"),
        ("scot_gt_k", "SA-GAT + SCOT (GT K)"),
        ("scot_pred_k", "SA-GAT + SCOT (predicted K)"),
        ("cop_km_pred_k", "SA-GAT + COP-KM (predicted K)"),
    ]:
        vals = results[method]
        if len(vals) == 0:
            continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {label:<33}{mean:>10.4f}{std:>10.4f}")
    print(f"{'='*65}")

    # Save results
    save_path = scot_checkpoint_path.parent / "eval_end_to_end_scot_coco.json"
    save_data = {
        "n_images": n,
        "detection": {k: int(v) for k, v in detection_stats.items()},
        "k_estimation": {k: int(v) for k, v in k_stats.items()},
        "pga": {
            method: {
                "mean": sum(vals) / len(vals),
                "std": (sum((v - sum(vals)/len(vals)) ** 2 for v in vals) / len(vals)) ** 0.5,
            }
            for method, vals in results.items()
            if len(vals) > 0
        },
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end: HigherHRNet → SA-GAT → K-head → SCOT"
    )
    parser.add_argument("--scot_checkpoint", type=Path, required=True,
                        help="SA-GAT + SCOT checkpoint")
    parser.add_argument("--k_head_checkpoint", type=Path, required=True,
                        help="Frozen K-head checkpoint")
    parser.add_argument("--coco_img_dir", type=Path, required=True)
    parser.add_argument("--coco_ann_file", type=Path, required=True)
    parser.add_argument("--hrnet_weights", type=Path,
                        default=Path("vendors/simple-HigherHRNet/weights/pose_higher_hrnet_w32_512.pth"))
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hrnet_device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate(
        scot_checkpoint_path=args.scot_checkpoint,
        k_head_checkpoint_path=args.k_head_checkpoint,
        coco_img_dir=args.coco_img_dir,
        coco_ann_file=args.coco_ann_file,
        hrnet_weights=args.hrnet_weights,
        device=args.device,
        max_images=args.max_images,
        hrnet_device=args.hrnet_device,
    )


if __name__ == "__main__":
    main()
