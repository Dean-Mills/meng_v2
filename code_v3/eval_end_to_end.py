"""
End-to-end evaluation: HigherHRNet detection → SA-GAT/GAT → COP-Kmeans → PGA.

Compares three grouping methods on detected (not GT) keypoints:
  1. HigherHRNet's own associative embedding grouping
  2. kNN on SA-GAT/GAT embeddings
  3. COP-Kmeans on SA-GAT/GAT embeddings

Also reports detection recall (how many GT keypoints were matched).

Usage:
    python eval_end_to_end.py \
        --checkpoint outputs/frozen_checkpoints/sa_gat_full/best.pt \
        --coco_img_dir data/coco2017/val2017 \
        --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json \
        --hrnet_weights vendors/simple-HigherHRNet/weights/pose_higher_hrnet_w32_512.pth
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
from evaluator import compute_pga

# HigherHRNet imports
sys.path.insert(0, str(Path(__file__).parent / "vendors" / "simple-HigherHRNet"))

COCO_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

MATCH_THRESHOLD = 10.0  # pixels — max distance to match detected kp to GT kp


# ─────────────────────────────────────────────────────────────────────────────
# Keypoint matching: detected ↔ GT
# ─────────────────────────────────────────────────────────────────────────────

def match_detections_to_gt(
    det_kps: np.ndarray,
    det_types: np.ndarray,
    gt_kps_list: List[np.ndarray],
    threshold: float = MATCH_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Match detected keypoints to GT keypoints by type and proximity.

    Args:
        det_kps: [N_det, 2] detected keypoint positions (x, y)
        det_types: [N_det] joint type indices
        gt_kps_list: list of [17, 4] arrays (x, y, z, v) per GT person
        threshold: max pixel distance for a valid match

    Returns:
        matched_det_idx: indices into det_kps that were matched
        matched_gt_person: GT person label for each matched detection
        matched_gt_type: GT joint type for each matched detection
    """
    # Build flat GT keypoint list
    gt_positions = []  # (x, y)
    gt_person_ids = []
    gt_types = []

    for person_idx, kps in enumerate(gt_kps_list):
        for j in range(17):
            v = kps[j, 3] if kps.shape[1] == 4 else kps[j, 2]
            if v > 0:  # visible or occluded
                gt_positions.append([kps[j, 0], kps[j, 1]])
                gt_person_ids.append(person_idx)
                gt_types.append(j)

    if len(gt_positions) == 0 or len(det_kps) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)

    gt_positions = np.array(gt_positions)
    gt_person_ids = np.array(gt_person_ids)
    gt_types = np.array(gt_types)

    matched_det_idx = []
    matched_gt_person = []
    matched_gt_type = []

    used_gt = set()

    # For each joint type, match detected to GT using Hungarian assignment
    for jt in range(17):
        det_mask = det_types == jt
        gt_mask = gt_types == jt

        det_indices = np.where(det_mask)[0]
        gt_indices = np.where(gt_mask)[0]

        if len(det_indices) == 0 or len(gt_indices) == 0:
            continue

        # Cost matrix: pairwise distances
        det_pos = det_kps[det_indices]
        gt_pos = gt_positions[gt_indices]
        cost = np.linalg.norm(det_pos[:, None] - gt_pos[None, :], axis=2)

        row, col = linear_sum_assignment(cost)

        for r, c in zip(row, col):
            if cost[r, c] < threshold and gt_indices[c] not in used_gt:
                matched_det_idx.append(det_indices[r])
                matched_gt_person.append(gt_person_ids[gt_indices[c]])
                matched_gt_type.append(jt)
                used_gt.add(gt_indices[c])

    return (
        np.array(matched_det_idx, dtype=int),
        np.array(matched_gt_person, dtype=int),
        np.array(matched_gt_type, dtype=int),
    )


def hrnet_joints_to_detections(
    joints: np.ndarray,
    conf_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert HigherHRNet output to flat detection arrays.

    Args:
        joints: [n_people, 17, 3] array with (y, x, confidence)
        conf_threshold: minimum confidence to keep a detection

    Returns:
        positions: [N, 2] (x, y) pixel positions
        types: [N] joint type indices
        person_ids: [N] HigherHRNet's person assignment
    """
    if len(joints) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    positions = []
    types = []
    person_ids = []

    for person_idx in range(joints.shape[0]):
        for j in range(17):
            y, x, conf = joints[person_idx, j]
            if conf > conf_threshold:
                positions.append([x, y])
                types.append(j)
                person_ids.append(person_idx)

    if len(positions) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    return np.array(positions), np.array(types, dtype=int), np.array(person_ids, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Build graph from detected keypoints (no GT needed)
# ─────────────────────────────────────────────────────────────────────────────

def detections_to_keypoints_list(
    positions: np.ndarray,
    types: np.ndarray,
    gt_person_labels: np.ndarray,
    image_size: int = 512,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Convert matched detections into the format expected by PosePreprocessor.

    We create a fake keypoints_list where each "person" has up to 17 joints,
    but we put ALL keypoints into a single flat structure that create_graph
    can process. The person_labels come from GT matching.

    Returns:
        keypoints_list: list of [17, 4] tensors (one per fake person, just for
                       preprocessor compatibility)
        Actually, we return the raw data needed to call create_graph directly.
    """
    # We can't use the standard create_graph because it expects per-person
    # keypoints. Instead we'll build the graph manually.
    pass


def build_graph_from_detections(
    positions: np.ndarray,
    types: np.ndarray,
    preprocessor: PosePreprocessor,
    device: str,
) -> Optional[torch.Tensor]:
    """
    Build a PyG graph from unassigned detected keypoints.

    Args:
        positions: [N, 2] (x, y) in pixel coords
        types: [N] joint type indices
        preprocessor: PosePreprocessor instance

    Returns:
        PyG Data object with x, edge_index, joint_types (no person_labels)
    """
    from torch_geometric.data import Data

    if len(positions) < 2:
        return None

    # Normalise positions
    x_norm = torch.tensor(positions[:, 0], dtype=torch.float32) / preprocessor.image_size
    y_norm = torch.tensor(positions[:, 1], dtype=torch.float32) / preprocessor.image_size

    if preprocessor.use_depth:
        z_norm = torch.zeros(len(positions), dtype=torch.float32)
        v_norm = torch.ones(len(positions), dtype=torch.float32)  # all detected = visible
        node_x = torch.stack([x_norm, y_norm, z_norm, v_norm], dim=1)
    else:
        v_norm = torch.ones(len(positions), dtype=torch.float32)
        node_x = torch.stack([x_norm, y_norm, v_norm], dim=1)

    node_x = node_x.to(device)
    jt = torch.tensor(types, dtype=torch.long).to(device)

    # kNN edges
    if preprocessor.use_depth:
        spatial = node_x[:, :3]
    else:
        spatial = node_x[:, :2]
    edge_index = preprocessor._knn_edges(spatial, k=preprocessor.k_neighbors)

    return Data(x=node_x, edge_index=edge_index, joint_types=jt)


# ─────────────────────────────────────────────────────────────────────────────
# COP-Kmeans (copied from eval_cop_kmeans to avoid circular imports)
# ─────────────────────────────────────────────────────────────────────────────

def predict_cop_kmeans(
    embeddings: torch.Tensor,
    k: int,
    joint_types: torch.Tensor,
) -> torch.Tensor:
    from eval_cop_kmeans import predict_cop_kmeans as _predict_cop_kmeans
    return _predict_cop_kmeans(embeddings, k, joint_types)


def predict_knn(embeddings: torch.Tensor, k: int) -> torch.Tensor:
    from evaluator import predict_knn as _predict_knn
    return _predict_knn(embeddings, k)


def predict_tha(
    embeddings: torch.Tensor,
    k: int,
    joint_types: torch.Tensor,
) -> torch.Tensor:
    from eval_hungarian_grouping import predict_tha as _predict_tha
    return _predict_tha(embeddings, k, joint_types)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    checkpoint_path: Path,
    coco_img_dir: Path,
    coco_ann_file: Path,
    hrnet_weights: Path,
    device: str,
    max_images: Optional[int] = None,
    hrnet_device: str = "cpu",
):
    # ── Load SA-GAT/GAT ──────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ExperimentConfig(**ckpt["config"])

    needs_mobilenet = False
    if cfg.sa_gat_v2 is not None:
        from sa_gat_v2 import SAGATV2Embedding
        gat = SAGATV2Embedding(cfg.sa_gat_v2).to(device)
        embedding_dim = cfg.sa_gat_v2.output_dim
        use_depth = cfg.sa_gat_v2.use_depth
        needs_mobilenet = True
    elif cfg.sa_gat is not None:
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

    print(f"Loaded SA-GAT/GAT checkpoint (epoch {ckpt.get('epoch', '?')})")

    # ── Load MobileNetV2 extractor (only if needed) ───────────────────────
    mobilenet_extractor = None
    if needs_mobilenet:
        from cache_mobilenet_features import MobileNetExtractor, sample_features_at_keypoints
        mobilenet_extractor = MobileNetExtractor(device)
        print(f"Loaded MobileNetV2 extractor for SA-GAT v2 visual features")

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

    # Get images with at least 2 people (same filter as our eval)
    cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = sorted(coco.getImgIds(catIds=cat_ids))

    # Filter to images with >= 2 people with enough keypoints
    valid_img_ids = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        people_with_kps = []
        for ann in anns:
            kps = np.array(ann["keypoints"]).reshape(17, 3)
            if (kps[:, 2] > 0).sum() >= 3:  # at least 3 visible keypoints
                people_with_kps.append(ann)
        if len(people_with_kps) >= 1:
            valid_img_ids.append(img_id)

    print(f"COCO: {len(valid_img_ids)} valid images")

    if max_images:
        valid_img_ids = valid_img_ids[:max_images]

    # ── Evaluate ─────────────────────────────────────────────────────────
    results = {"hrnet_ae": [], "knn": [], "cop_kmeans": [], "tha": []}
    detection_stats = {"total_gt": 0, "total_matched": 0, "total_detected": 0}

    with torch.no_grad():
        for i, img_id in enumerate(valid_img_ids):
            img_info = coco.loadImgs(img_id)[0]
            img_path = coco_img_dir / img_info["file_name"]
            image = cv2.imread(str(img_path))

            if image is None:
                continue

            # ── HigherHRNet detection ─────────────────────────────────
            joints = hrnet.predict(image)  # [n_people, 17, 3] (y, x, conf)

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
                kps_4[:, 0] = kps[:, 0]  # x
                kps_4[:, 1] = kps[:, 1]  # y
                kps_4[:, 2] = 0.0        # z (unused)
                kps_4[:, 3] = kps[:, 2]  # visibility
                if (kps[:, 2] > 0).sum() >= 3:
                    gt_kps_list.append(kps_4)

            if len(gt_kps_list) < 1:
                continue

            # ── Match detections to GT ────────────────────────────────
            matched_det_idx, matched_gt_person, matched_gt_type = \
                match_detections_to_gt(det_pos, det_types, gt_kps_list)

            if len(matched_det_idx) < 2:
                continue

            # Count total GT visible keypoints
            n_gt_kps = sum((kps[:, 3] > 0).sum() for kps in gt_kps_list)
            detection_stats["total_gt"] += n_gt_kps
            detection_stats["total_matched"] += len(matched_det_idx)
            detection_stats["total_detected"] += len(det_pos)

            n_gt_people = len(gt_kps_list)

            # ── 1. HigherHRNet AE grouping PGA ────────────────────────
            # Use HigherHRNet's own person assignments for matched keypoints
            hrnet_labels = torch.tensor(
                hrnet_person_ids[matched_det_idx], dtype=torch.long
            )
            gt_labels = torch.tensor(matched_gt_person, dtype=torch.long)
            hrnet_pga = compute_pga(hrnet_labels, gt_labels)
            results["hrnet_ae"].append(hrnet_pga)

            # ── 2 & 3. SA-GAT kNN and COP-Kmeans ─────────────────────
            # Build graph from matched detections only
            matched_pos = det_pos[matched_det_idx]
            matched_types = det_types[matched_det_idx]

            graph = build_graph_from_detections(
                matched_pos, matched_types, preprocessor, device,
            )
            if graph is None:
                continue

            graph = graph.to(device)

            # If using SA-GAT v2, sample MobileNet features at the matched
            # keypoint locations and attach to the graph.
            if mobilenet_extractor is not None:
                from cache_mobilenet_features import sample_features_at_keypoints
                imgH, imgW = image.shape[:2]
                feats_full = mobilenet_extractor.extract(image)
                feats_at_kps = sample_features_at_keypoints(
                    feats_full, matched_pos, (imgH, imgW),
                )
                graph.features = feats_at_kps.to(device)

            embeddings = gat(graph)

            # Skip if fewer matched keypoints than people
            if len(matched_det_idx) < n_gt_people:
                continue

            # kNN
            knn_pred = predict_knn(embeddings, n_gt_people)
            knn_pga = compute_pga(knn_pred, gt_labels.to(device))
            results["knn"].append(knn_pga)

            # COP-Kmeans
            cop_pred = predict_cop_kmeans(
                embeddings, n_gt_people, graph.joint_types,
            )
            cop_pga = compute_pga(cop_pred, gt_labels.to(device))
            results["cop_kmeans"].append(cop_pga)

            # THA
            tha_pred = predict_tha(
                embeddings, n_gt_people, graph.joint_types,
            )
            tha_pga = compute_pga(tha_pred, gt_labels.to(device))
            results["tha"].append(tha_pga)

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(valid_img_ids)} images processed...")

    # ── Print results ────────────────────────────────────────────────────
    n = len(results["hrnet_ae"])
    if n == 0:
        print("No valid images processed!")
        return

    print(f"\n{'='*65}")
    print(f"END-TO-END EVALUATION ({n} images)")
    print(f"{'='*65}")

    recall = detection_stats["total_matched"] / max(detection_stats["total_gt"], 1)
    precision = detection_stats["total_matched"] / max(detection_stats["total_detected"], 1)
    print(f"\nDetection stats:")
    print(f"  GT keypoints:       {detection_stats['total_gt']}")
    print(f"  Detected keypoints: {detection_stats['total_detected']}")
    print(f"  Matched keypoints:  {detection_stats['total_matched']}")
    print(f"  Recall:             {recall:.3f}")
    print(f"  Precision:          {precision:.3f}")

    print(f"\n{'Method':<25}{'PGA':>10}{'Std':>10}")
    print("-" * 45)
    for method, label in [
        ("hrnet_ae", "HigherHRNet AE"),
        ("knn", "SA-GAT + kNN"),
        ("cop_kmeans", "SA-GAT + COP-Kmeans"),
        ("tha", "SA-GAT + THA"),
    ]:
        vals = results[method]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {label:<23}{mean:>10.4f}{std:>10.4f}")
    print(f"{'='*65}")

    # Save results
    save_path = checkpoint_path.parent / "eval_end_to_end_coco.json"
    save_data = {
        "n_images": n,
        "detection": {k: int(v) for k, v in detection_stats.items()},
        "pga": {
            method: {
                "mean": sum(vals) / len(vals),
                "std": (sum((v - sum(vals)/len(vals)) ** 2 for v in vals) / len(vals)) ** 0.5,
            }
            for method, vals in results.items()
        },
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation: HigherHRNet → SA-GAT → COP-Kmeans"
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="SA-GAT or GAT checkpoint")
    parser.add_argument("--coco_img_dir", type=Path, required=True)
    parser.add_argument("--coco_ann_file", type=Path, required=True)
    parser.add_argument("--hrnet_weights", type=Path,
                        default=Path("vendors/simple-HigherHRNet/weights/pose_higher_hrnet_w32_512.pth"))
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hrnet_device", type=str, default="cpu",
                        help="Device for HigherHRNet (cpu recommended to save GPU memory)")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        coco_img_dir=args.coco_img_dir,
        coco_ann_file=args.coco_ann_file,
        hrnet_weights=args.hrnet_weights,
        device=args.device,
        max_images=args.max_images,
        hrnet_device=args.hrnet_device,
    )


if __name__ == "__main__":
    main()
