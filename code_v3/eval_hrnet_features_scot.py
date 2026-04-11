"""
Test SCOT (and THA, COP-Kmeans) with HigherHRNet backbone features as embeddings.

Diagnostic experiment: if SCOT fails even with HigherHRNet's own backbone
features (which were trained on 118k real COCO images), the SCOT architecture
is definitively the problem, not our embedding pipeline.

The 32-dim backbone features are extracted via a forward hook on the input
to HigherHRNet's final conv layer (final_layers[0]). These are the features
that both the heatmap head and the AE tag head read from.

Usage:
    python eval_hrnet_features_scot.py \
        --coco_img_dir data/coco2017/val2017 \
        --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json \
        --hrnet_weights vendors/simple-HigherHRNet/weights/pose_higher_hrnet_w32_512.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from evaluator import compute_pga, predict_knn
from eval_cop_kmeans import predict_cop_kmeans
from eval_hungarian_grouping import predict_tha
from eval_end_to_end import (
    match_detections_to_gt,
    hrnet_joints_to_detections,
)

sys.path.insert(0, str(Path(__file__).parent / "vendors" / "simple-HigherHRNet"))


class FeatureExtractor:
    """Hook into HigherHRNet to capture the 32-dim backbone features."""

    def __init__(self, hrnet_model):
        self.features = None
        # final_layers[0] is Conv2d(32, 34, 1x1) — the input is the 32-dim feature map
        self.hook = hrnet_model.final_layers[0].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # input is a tuple, first element is the 32-channel feature map
        self.features = input[0].detach()  # [B, 32, H, W]

    def close(self):
        self.hook.remove()


def sample_features_at_keypoints(
    features: torch.Tensor,          # [1, 32, H, W] feature map
    keypoints_xy: np.ndarray,        # [N, 2] (x, y) in original image coords
    image_size_hw: tuple,            # (H, W) of original image
) -> torch.Tensor:
    """
    Bilinearly sample feature vectors at keypoint locations.

    Returns:
        [N, 32] feature vectors
    """
    _, C, fH, fW = features.shape
    imgH, imgW = image_size_hw

    # Normalise keypoint coords to [-1, 1] for grid_sample
    xs = torch.tensor(keypoints_xy[:, 0], dtype=torch.float32) / imgW * 2 - 1
    ys = torch.tensor(keypoints_xy[:, 1], dtype=torch.float32) / imgH * 2 - 1
    grid = torch.stack([xs, ys], dim=1).view(1, -1, 1, 2).to(features.device)

    sampled = F.grid_sample(features, grid, mode="bilinear", align_corners=True)
    # sampled: [1, C, N, 1]
    return sampled.squeeze(-1).squeeze(0).T  # [N, C]


def evaluate(
    coco_img_dir: Path,
    coco_ann_file: Path,
    hrnet_weights: Path,
    device: str,
    max_images: Optional[int] = None,
    l2_normalise: bool = True,
):
    # ── Load HigherHRNet ─────────────────────────────────────────────────
    from SimpleHigherHRNet import SimpleHigherHRNet
    hrnet_wrapper = SimpleHigherHRNet(
        c=32, nof_joints=17,
        checkpoint_path=str(hrnet_weights),
        resolution=512,
        device=torch.device(device),
    )

    # Access the underlying model (may be DataParallel wrapped)
    hrnet_model = hrnet_wrapper.model
    if isinstance(hrnet_model, torch.nn.DataParallel):
        hrnet_model = hrnet_model.module

    extractor = FeatureExtractor(hrnet_model)
    print(f"Loaded HigherHRNet w32-512 with feature extractor hook")

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
        n = sum(1 for a in anns
                if sum(1 for v in a["keypoints"][2::3] if v > 0) >= 3)
        if n >= 1:
            valid_img_ids.append(img_id)

    print(f"COCO: {len(valid_img_ids)} valid images")

    if max_images:
        valid_img_ids = valid_img_ids[:max_images]

    # Build SCOT-KI head to test on these features
    from config import SCOTConfig
    from ot_head_kmeans_init import SCOTKmeansInitHead
    scot_cfg = SCOTConfig(hidden_dim=32, k_max=20, sinkhorn_iters=10, sinkhorn_tau=0.1)
    scot_ki = SCOTKmeansInitHead(scot_cfg, embedding_dim=32).to(device)
    scot_ki.node_encoder = torch.nn.Identity()  # use raw features
    scot_ki.eval()

    results = {"hrnet_ae": [], "knn": [], "cop_kmeans": [], "tha": [], "scot_ki": []}
    detection_stats = {"total_gt": 0, "total_matched": 0, "total_detected": 0}

    with torch.no_grad():
        for i, img_id in enumerate(valid_img_ids):
            img_info = coco.loadImgs(img_id)[0]
            img_path = coco_img_dir / img_info["file_name"]
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            imgH, imgW = image.shape[:2]

            # Run HigherHRNet — this triggers the hook and stores features
            joints = hrnet_wrapper.predict(image)
            det_pos, det_types, hrnet_labels = hrnet_joints_to_detections(joints)

            if len(det_pos) < 2:
                continue

            backbone_features = extractor.features  # [1, 32, fH, fW]
            if backbone_features is None:
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
                kps_4[:, 3] = kps[:, 2]
                if (kps[:, 2] > 0).sum() >= 3:
                    gt_kps_list.append(kps_4)

            if len(gt_kps_list) < 1:
                continue

            # Match detections to GT
            matched_det_idx, matched_gt_person, matched_gt_type = \
                match_detections_to_gt(det_pos, det_types, gt_kps_list)

            if len(matched_det_idx) < 2:
                continue

            n_gt_kps = sum((kps[:, 3] > 0).sum() for kps in gt_kps_list)
            detection_stats["total_gt"] += n_gt_kps
            detection_stats["total_matched"] += len(matched_det_idx)
            detection_stats["total_detected"] += len(det_pos)

            n_gt_people = len(gt_kps_list)
            if len(matched_det_idx) < n_gt_people:
                continue

            gt_labels = torch.tensor(matched_gt_person, dtype=torch.long, device=device)

            # ── HigherHRNet AE grouping ────────────────────────────────
            hrnet_labels_matched = torch.tensor(
                hrnet_labels[matched_det_idx], dtype=torch.long, device=device
            )
            results["hrnet_ae"].append(compute_pga(hrnet_labels_matched, gt_labels))

            # ── Sample backbone features at matched detection locations ─
            matched_pos = det_pos[matched_det_idx]
            matched_types = torch.tensor(
                det_types[matched_det_idx], dtype=torch.long, device=device
            )

            embeddings = sample_features_at_keypoints(
                backbone_features, matched_pos, (imgH, imgW),
            )

            if l2_normalise:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            # ── kNN, COP-Kmeans, THA, SCOT-KI on HRNet features ───────
            knn_pred = predict_knn(embeddings, n_gt_people)
            results["knn"].append(compute_pga(knn_pred, gt_labels))

            cop_pred = predict_cop_kmeans(embeddings, n_gt_people, matched_types)
            results["cop_kmeans"].append(compute_pga(cop_pred, gt_labels))

            tha_pred = predict_tha(embeddings, n_gt_people, matched_types)
            results["tha"].append(compute_pga(tha_pred, gt_labels))

            logits, _ = scot_ki(embeddings, n_gt_people, matched_types)
            scot_ki_pred = logits.argmax(dim=1)
            results["scot_ki"].append(compute_pga(scot_ki_pred, gt_labels))

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(valid_img_ids)} images processed...")

    extractor.close()

    # ── Print results ────────────────────────────────────────────────────
    n = len(results["hrnet_ae"])
    if n == 0:
        print("No valid images!")
        return

    print(f"\n{'='*70}")
    print(f"SCOT / THA / COP-KMEANS on HRNet BACKBONE FEATURES ({n} images)")
    print(f"{'='*70}")
    print(f"L2 normalised: {l2_normalise}")
    print(f"\nDetection: recall {detection_stats['total_matched']/max(detection_stats['total_gt'],1):.3f}")
    print(f"\n{'Method':<40}{'PGA':>10}{'Std':>10}")
    print("-" * 60)
    for method, label in [
        ("hrnet_ae", "HigherHRNet AE grouping"),
        ("knn", "kNN on HRNet features"),
        ("cop_kmeans", "COP-Kmeans on HRNet features"),
        ("tha", "THA on HRNet features"),
        ("scot_ki", "SCOT-KI on HRNet features"),
    ]:
        vals = results[method]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {label:<38}{mean:>10.4f}{std:>10.4f}")
    print(f"{'='*70}")

    save_path = Path("outputs") / f"eval_hrnet_features_{'l2' if l2_normalise else 'raw'}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "n_images": n,
        "l2_normalised": l2_normalise,
        "detection": {k: int(v) for k, v in detection_stats.items()},
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
    parser.add_argument("--coco_img_dir", type=Path, required=True)
    parser.add_argument("--coco_ann_file", type=Path, required=True)
    parser.add_argument("--hrnet_weights", type=Path,
                        default=Path("vendors/simple-HigherHRNet/weights/pose_higher_hrnet_w32_512.pth"))
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--no_l2", action="store_true",
                        help="Skip L2 normalisation of HRNet features")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate(
        coco_img_dir=args.coco_img_dir,
        coco_ann_file=args.coco_ann_file,
        hrnet_weights=args.hrnet_weights,
        device=args.device,
        max_images=args.max_images,
        l2_normalise=not args.no_l2,
    )


if __name__ == "__main__":
    main()
