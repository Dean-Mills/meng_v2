"""
Feasibility test: do general-purpose visual features (MobileNetV2) help SA-GAT group?

Pipeline:
  1. Detect keypoints with HigherHRNet (existing)
  2. Run MobileNetV2 on the same image, extract intermediate feature map
  3. Sample MobileNet features at each detected keypoint location
  4. Compare grouping methods on:
       (a) just the keypoint positions (current SA-GAT input)
       (b) keypoint positions + MobileNet features (concatenated)
  5. Run k-means/COP-Kmeans/THA on each and compare PGA

This is purely an evaluation experiment — no training. The question is:
  Does adding visual context to keypoint embeddings make grouping easier?

If yes: build the full Paper 3 pipeline.
If no: rethink.

Usage:
    python eval_mobilenet_features.py \
        --coco_img_dir data/coco2017/val2017 \
        --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json \
        --max_images 500
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
import torchvision.models as tvm
import torchvision.transforms as T

from evaluator import compute_pga, predict_knn
from eval_cop_kmeans import predict_cop_kmeans
from eval_hungarian_grouping import predict_tha
from eval_end_to_end import (
    match_detections_to_gt,
    hrnet_joints_to_detections,
)

sys.path.insert(0, str(Path(__file__).parent / "vendors" / "simple-HigherHRNet"))


# Layer indices in MobileNetV2.features to capture
# Layer 6 = 32-dim features at /8 resolution (high-res, low-semantic)
# Layer 13 = 96-dim features at /16 resolution (balanced)
# Layer 17 = 320-dim features at /32 resolution (low-res, high-semantic)
MOBILENET_LAYERS = {
    "shallow": 6,    # 32-dim, fine spatial
    "mid": 13,       # 96-dim, medium
    "deep": 17,      # 320-dim, coarse but semantic
}


class MobileNetExtractor:
    """Wraps MobileNetV2 to extract features from a chosen layer."""

    def __init__(self, layer_name: str = "deep", device: str = "cuda"):
        self.layer_idx = MOBILENET_LAYERS[layer_name]
        self.device = device
        self.model = tvm.mobilenet_v2(weights="DEFAULT").eval().to(device)

        # ImageNet normalisation
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, image_bgr: np.ndarray) -> torch.Tensor:
        """
        Args:
            image_bgr: [H, W, 3] BGR uint8 image

        Returns:
            features: [1, C, fH, fW] feature map at the chosen layer
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Forward through features up to the chosen layer
        out = x
        for i, layer in enumerate(self.model.features):
            out = layer(out)
            if i == self.layer_idx:
                break

        return out  # [1, C, fH, fW]


def sample_features_at_keypoints(
    features: torch.Tensor,
    keypoints_xy: np.ndarray,
    image_size_hw: tuple,
) -> torch.Tensor:
    """Bilinearly sample feature vectors at keypoint pixel locations."""
    _, C, fH, fW = features.shape
    imgH, imgW = image_size_hw

    xs = torch.tensor(keypoints_xy[:, 0], dtype=torch.float32) / imgW * 2 - 1
    ys = torch.tensor(keypoints_xy[:, 1], dtype=torch.float32) / imgH * 2 - 1
    grid = torch.stack([xs, ys], dim=1).view(1, -1, 1, 2).to(features.device)

    sampled = F.grid_sample(features, grid, mode="bilinear", align_corners=True)
    return sampled.squeeze(-1).squeeze(0).T  # [N, C]


def build_input_embeddings(
    positions: np.ndarray,
    types: np.ndarray,
    image_size_hw: tuple,
    mobilenet_features: Optional[torch.Tensor],
    device: str,
    type_embed_dim: int = 16,
) -> torch.Tensor:
    """
    Build a simple [N, D] embedding for each keypoint.

    If mobilenet_features is None: returns [x_norm, y_norm] only.
    If provided: concatenates [x_norm, y_norm, type_one_hot, mobilenet_features].

    For this feasibility test we don't run SA-GAT — we just use the raw input
    features and run grouping on them directly. This isolates the question
    "do visual features add useful information for grouping?" from the
    "does SA-GAT exploit them?" question.
    """
    imgH, imgW = image_size_hw
    n = len(positions)

    x_norm = torch.tensor(positions[:, 0], dtype=torch.float32) / imgW
    y_norm = torch.tensor(positions[:, 1], dtype=torch.float32) / imgH

    components = [x_norm.unsqueeze(1), y_norm.unsqueeze(1)]

    # One-hot type encoding
    type_one_hot = F.one_hot(
        torch.tensor(types, dtype=torch.long), num_classes=17
    ).float()
    components.append(type_one_hot)

    if mobilenet_features is not None:
        # L2-normalise the mobilenet features for stability
        mobilenet_norm = F.normalize(mobilenet_features.cpu(), p=2, dim=1)
        components.append(mobilenet_norm)

    embeddings = torch.cat(components, dim=1).to(device)
    # L2 normalise the whole vector for k-means consistency
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


def evaluate(
    coco_img_dir: Path,
    coco_ann_file: Path,
    hrnet_weights: Path,
    layer: str,
    device: str,
    max_images: Optional[int] = None,
):
    # ── Load HigherHRNet (for detection only) ───────────────────────────
    from SimpleHigherHRNet import SimpleHigherHRNet
    hrnet = SimpleHigherHRNet(
        c=32, nof_joints=17,
        checkpoint_path=str(hrnet_weights),
        resolution=512,
        device=torch.device(device),
    )
    print(f"Loaded HigherHRNet for keypoint detection")

    # ── Load MobileNetV2 ────────────────────────────────────────────────
    mobilenet = MobileNetExtractor(layer_name=layer, device=device)
    sample_feature_dim = MOBILENET_LAYERS[layer]
    print(f"Loaded MobileNetV2 (layer={layer}, idx={sample_feature_dim})")

    # ── Load COCO ───────────────────────────────────────────────────────
    from pycocotools.coco import COCO
    print("Loading COCO annotations...")
    coco = COCO(str(coco_ann_file))
    cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = sorted(coco.getImgIds(catIds=cat_ids))

    valid = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        n = sum(1 for a in anns
                if sum(1 for v in a["keypoints"][2::3] if v > 0) >= 3)
        if n >= 1:
            valid.append(img_id)

    if max_images:
        valid = valid[:max_images]
    print(f"Evaluating on {len(valid)} images")

    results = {
        "knn_pos": [], "knn_pos_feat": [],
        "cop_pos": [], "cop_pos_feat": [],
        "tha_pos": [], "tha_pos_feat": [],
    }

    with torch.no_grad():
        for i, img_id in enumerate(valid):
            img_info = coco.loadImgs(img_id)[0]
            img_path = coco_img_dir / img_info["file_name"]
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            imgH, imgW = image.shape[:2]

            # Detect with HRNet
            joints = hrnet.predict(image)
            det_pos, det_types, hrnet_labels = hrnet_joints_to_detections(joints)

            if len(det_pos) < 2:
                continue

            # GT
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

            matched_idx, matched_person, matched_type = match_detections_to_gt(
                det_pos, det_types, gt_kps_list,
            )
            if len(matched_idx) < 2:
                continue

            n_people = len(gt_kps_list)
            if len(matched_idx) < n_people:
                continue

            gt_labels = torch.tensor(matched_person, dtype=torch.long, device=device)
            matched_pos_arr = det_pos[matched_idx]
            matched_types_arr = det_types[matched_idx]
            matched_types_t = torch.tensor(
                matched_types_arr, dtype=torch.long, device=device,
            )

            # Extract MobileNet features
            mobilenet_feats_full = mobilenet.extract(image)  # [1, C, fH, fW]
            mobilenet_at_kps = sample_features_at_keypoints(
                mobilenet_feats_full, matched_pos_arr, (imgH, imgW),
            )  # [N, C]

            # Build two embedding versions
            emb_pos = build_input_embeddings(
                matched_pos_arr, matched_types_arr, (imgH, imgW),
                mobilenet_features=None, device=device,
            )
            emb_pos_feat = build_input_embeddings(
                matched_pos_arr, matched_types_arr, (imgH, imgW),
                mobilenet_features=mobilenet_at_kps, device=device,
            )

            # Run grouping methods on both
            for emb, suffix in [(emb_pos, "pos"), (emb_pos_feat, "pos_feat")]:
                knn_pred = predict_knn(emb, n_people)
                cop_pred = predict_cop_kmeans(emb, n_people, matched_types_t)
                tha_pred = predict_tha(emb, n_people, matched_types_t)

                results[f"knn_{suffix}"].append(compute_pga(knn_pred, gt_labels))
                results[f"cop_{suffix}"].append(compute_pga(cop_pred, gt_labels))
                results[f"tha_{suffix}"].append(compute_pga(tha_pred, gt_labels))

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(valid)}")

    n = len(results["knn_pos"])
    print(f"\n{'='*70}")
    print(f"MOBILENET FEATURE FEASIBILITY TEST ({n} images)")
    print(f"MobileNet layer: {layer} ({sample_feature_dim} idx)")
    print(f"{'='*70}")

    print(f"\n{'Method':<35}{'Position only':>15}{'+ MobileNet':>15}{'Diff':>10}")
    print("-" * 75)
    for base in ["knn", "cop", "tha"]:
        pos_vals = results[f"{base}_pos"]
        feat_vals = results[f"{base}_pos_feat"]
        pos_mean = sum(pos_vals) / len(pos_vals)
        feat_mean = sum(feat_vals) / len(feat_vals)
        diff = feat_mean - pos_mean
        sign = "+" if diff >= 0 else ""
        label = {"knn": "kNN", "cop": "COP-Kmeans", "tha": "THA"}[base]
        print(f"  {label:<33}{pos_mean:>15.4f}{feat_mean:>15.4f}{sign}{diff:>9.4f}")
    print(f"{'='*70}")

    save_path = Path("outputs") / f"eval_mobilenet_features_{layer}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "n_images": n,
        "layer": layer,
        "results": {
            k: {"mean": sum(v)/len(v),
                "std": (sum((x - sum(v)/len(v))**2 for x in v)/len(v))**0.5}
            for k, v in results.items()
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
    parser.add_argument("--layer", type=str, default="deep",
                        choices=list(MOBILENET_LAYERS.keys()))
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluate(
        coco_img_dir=args.coco_img_dir,
        coco_ann_file=args.coco_ann_file,
        hrnet_weights=args.hrnet_weights,
        layer=args.layer,
        device=args.device,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
