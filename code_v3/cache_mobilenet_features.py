"""
Cache MobileNetV2 features at GT keypoint locations for COCO train2017.

For each COCO image with annotated people:
  - Run MobileNetV2 (deep layer 17, 320-dim) on the image
  - For each annotated person, sample features at their visible keypoint locations
  - Save the keypoints + person labels + sampled features to a single torch file

Output: one .pt file per image containing a dict:
  {
    "img_id":        int,
    "img_size":      (H, W),
    "keypoints":     [N, 4] tensor (x, y, z=0, v) — flat list across people
    "joint_types":   [N] long tensor (0-16)
    "person_labels": [N] long tensor (0..K-1)
    "features":      [N, 320] float tensor (MobileNetV2 deep features at each kp)
    "num_people":    int
  }

Usage:
    python cache_mobilenet_features.py \
        --coco_img_dir data/coco2017/train2017 \
        --coco_ann_file data/coco2017/annotations/person_keypoints_train2017.json \
        --output_dir data/cached_mobilenet_train
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T


# MobileNetV2 layer 17: 320-dim features at /32 resolution (semantic, deep)
MOBILENET_LAYER = 17
MOBILENET_DIM = 320


class MobileNetExtractor:
    def __init__(self, device: str):
        self.device = device
        self.model = tvm.mobilenet_v2(weights="DEFAULT").eval().to(device)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, image_bgr: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(image_rgb).unsqueeze(0).to(self.device)

        out = x
        for i, layer in enumerate(self.model.features):
            out = layer(out)
            if i == MOBILENET_LAYER:
                break

        return out  # [1, 320, fH, fW]


def sample_features_at_keypoints(
    features: torch.Tensor,
    keypoints_xy: np.ndarray,
    image_size_hw: tuple,
) -> torch.Tensor:
    """Bilinearly sample feature vectors at keypoint pixel locations."""
    imgH, imgW = image_size_hw
    xs = torch.tensor(keypoints_xy[:, 0], dtype=torch.float32) / imgW * 2 - 1
    ys = torch.tensor(keypoints_xy[:, 1], dtype=torch.float32) / imgH * 2 - 1
    grid = torch.stack([xs, ys], dim=1).view(1, -1, 1, 2).to(features.device)

    sampled = F.grid_sample(features, grid, mode="bilinear", align_corners=True)
    return sampled.squeeze(-1).squeeze(0).T  # [N, C]


def process_image(
    coco,
    img_id: int,
    img_dir: Path,
    cat_ids,
    extractor: MobileNetExtractor,
):
    """
    Returns a dict ready to save, or None if the image should be skipped.
    """
    img_info = coco.loadImgs(img_id)[0]
    img_path = img_dir / img_info["file_name"]
    image = cv2.imread(str(img_path))
    if image is None:
        return None

    imgH, imgW = image.shape[:2]

    # Collect annotated people with at least 3 visible keypoints
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    keypoints_flat = []  # list of (x, y, z=0, v)
    joint_types_flat = []
    person_labels_flat = []

    person_idx = 0
    for ann in anns:
        kps = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)
        n_visible = (kps[:, 2] > 0).sum()
        if n_visible < 3:
            continue

        for jt in range(17):
            v = kps[jt, 2]
            if v == 0:
                continue
            x = kps[jt, 0]
            y = kps[jt, 1]
            keypoints_flat.append([x, y, 0.0, v])
            joint_types_flat.append(jt)
            person_labels_flat.append(person_idx)

        person_idx += 1

    if person_idx < 1 or len(keypoints_flat) < 2:
        return None

    keypoints_arr = np.array(keypoints_flat, dtype=np.float32)

    # Run MobileNet
    features_full = extractor.extract(image)  # [1, 320, fH, fW]

    # Sample at keypoint pixel positions (use actual x, y not normalised)
    sampled = sample_features_at_keypoints(
        features_full, keypoints_arr[:, :2], (imgH, imgW),
    )  # [N, 320]

    return {
        "img_id":        int(img_id),
        "img_size":      (int(imgH), int(imgW)),
        "keypoints":     torch.tensor(keypoints_arr, dtype=torch.float32),
        "joint_types":   torch.tensor(joint_types_flat, dtype=torch.long),
        "person_labels": torch.tensor(person_labels_flat, dtype=torch.long),
        "features":      sampled.cpu(),
        "num_people":    int(person_idx),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_img_dir", type=Path, required=True)
    parser.add_argument("--coco_ann_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_images", type=int, default=None,
                        help="Limit number of images for testing")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    extractor = MobileNetExtractor(args.device)
    print(f"Loaded MobileNetV2 (layer {MOBILENET_LAYER}, {MOBILENET_DIM}-dim) on {args.device}")

    from pycocotools.coco import COCO
    print("Loading COCO annotations...")
    coco = COCO(str(args.coco_ann_file))
    cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = sorted(coco.getImgIds(catIds=cat_ids))
    print(f"Found {len(img_ids)} images with people")

    if args.max_images:
        img_ids = img_ids[:args.max_images]

    n_saved = 0
    n_skipped = 0
    t0 = time.time()

    for i, img_id in enumerate(img_ids):
        out_path = args.output_dir / f"{img_id:012d}.pt"
        if out_path.exists():
            n_saved += 1
            continue

        sample = process_image(coco, img_id, args.coco_img_dir, cat_ids, extractor)
        if sample is None:
            n_skipped += 1
            continue

        torch.save(sample, out_path)
        n_saved += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(img_ids) - (i + 1)) / rate
            print(f"  {i+1}/{len(img_ids)} | "
                  f"saved {n_saved} | skipped {n_skipped} | "
                  f"{rate:.1f} img/s | ETA {eta/60:.1f} min")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Saved {n_saved} files to {args.output_dir}")
    print(f"Skipped {n_skipped} images (no usable people)")


if __name__ == "__main__":
    main()
