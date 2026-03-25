"""COCO dataset adapter
Reads COCO 2017 person keypoint annotations via pycocotools, estimates
per-joint depth using MiDaS DPT_Hybrid, and returns the unified data contract:

    {
        "image":      tensor [3, H, W]        uint8
        "keypoints":  list of tensor [17, 4]  float32  [x, y, z, v]
        "img_id":     str                      COCO image ID as string
        "num_people": int
    }

Depth (z) is sampled from a MiDaS depth map at each keypoint's pixel
location. MiDaS outputs relative inverse depth — the preprocessor
normalises both sources to [0, 1] so the model sees a consistent range.

MiDaS is initialised once at adapter construction time, not per sample.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from torchvision.io import read_image
from pycocotools.coco import COCO


class CocoAdapter:
    """
    Loads COCO 2017 person keypoint samples with MiDaS depth estimation.

    Args:
        img_dir:      Directory containing the COCO images.
        ann_file:     Path to the COCO annotation JSON
                      (e.g. person_keypoints_val2017.json).
        min_people:   Minimum number of people per image (inclusive).
        max_people:   Maximum number of people per image (inclusive).
                      None means no upper limit.
        device:       Device to run MiDaS on ('cuda' or 'cpu').
    """

    def __init__(
        self,
        img_dir: Path,
        ann_file: Path,
        min_people: int = 1,
        max_people: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.img_dir    = Path(img_dir)
        self.ann_file   = Path(ann_file)
        self.min_people = min_people
        self.max_people = max_people
        self.device     = device

        if not self.img_dir.exists():
            raise ValueError(f"COCO image directory not found: {self.img_dir}")
        if not self.ann_file.exists():
            raise ValueError(f"COCO annotation file not found: {self.ann_file}")

        # ── COCO index ────────────────────────────────────────────────────
        print("Loading COCO annotations...")
        self.coco    = COCO(str(self.ann_file))
        self.img_ids = self._index_samples()

        if len(self.img_ids) == 0:
            raise ValueError(
                f"No COCO samples found with person count "
                f"[{min_people}, {max_people or '∞'}]"
            )

        print(
            f"CocoAdapter: {len(self.img_ids)} images "
            f"(people: {min_people}–{max_people or '∞'})"
        )

        # ── MiDaS ─────────────────────────────────────────────────────────
        print(f"Loading MiDaS DPT_Hybrid on {device}...")
        import torch.nn as nn
        self.depth_model: nn.Module = torch.hub.load(  # type: ignore[assignment]
            "intel-isl/MiDaS", "DPT_Hybrid", verbose=False
        )
        self.depth_model.to(self.device)
        self.depth_model.eval()

        midas_transforms     = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
        self.depth_transform = midas_transforms.dpt_transform  # type: ignore[union-attr]
        print("MiDaS ready.")

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _index_samples(self) -> List[int]:
        """Return COCO image IDs whose person count is within [min, max]."""
        cat_ids     = self.coco.getCatIds(catNms=["person"])
        all_img_ids = self.coco.getImgIds(catIds=cat_ids)

        valid = []
        for img_id in all_img_ids:
            anns = self._get_person_anns(img_id)
            n    = len(anns)
            if n < self.min_people:
                continue
            if self.max_people is not None and n > self.max_people:
                continue
            valid.append(img_id)

        return valid

    def _get_person_anns(self, img_id: int) -> list:
        """Return valid person annotations for an image."""
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns    = self.coco.loadAnns(ann_ids)
        return [
            ann for ann in anns
            if ann["category_id"] == 1
            and "keypoints" in ann
            and sum(ann["keypoints"]) > 0
        ]

    # ── Depth estimation ──────────────────────────────────────────────────────

    def _estimate_depth(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run MiDaS DPT_Hybrid on an image tensor.

        Args:
            image: [3, H, W] uint8 tensor

        Returns:
            depth_map: [H, W] float32 tensor of relative inverse depth
        """
        # MiDaS expects a numpy HWC float image
        img_np = image.permute(1, 2, 0).cpu().numpy().astype(np.float32) / 255.0

        input_tensor = self.depth_transform(img_np).to(self.device)

        with torch.no_grad():
            prediction: torch.Tensor = self.depth_model(input_tensor)  # type: ignore[operator]
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[1:],   # restore to original H, W
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu()         # [H, W]

    def _sample_depth(
        self,
        depth_map: torch.Tensor,
        x: float,
        y: float,
    ) -> float:
        """Sample depth map at a keypoint pixel location with boundary clamping."""
        H, W = depth_map.shape
        xi   = int(min(max(round(x), 0), W - 1))
        yi   = int(min(max(round(y), 0), H - 1))
        return depth_map[yi, xi].item()

    # ── Public interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id   = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # ── Load image ────────────────────────────────────────────────────
        img_path = self.img_dir / img_info["file_name"]
        image    = read_image(str(img_path))    # [C, H, W] uint8

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] > 3:
            image = image[:3]

        # ── Estimate depth ────────────────────────────────────────────────
        depth_map = self._estimate_depth(image)     # [H, W]

        # ── Extract keypoints with depth ──────────────────────────────────
        anns      = self._get_person_anns(img_id)
        keypoints: List[torch.Tensor] = []

        for ann in anns:
            # COCO format: flat [x, y, v, x, y, v, ...] — 51 values
            raw = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)

            kps = np.zeros((17, 4), dtype=np.float32)

            for j in range(17):
                v = raw[j, 2]
                x = raw[j, 0]
                y = raw[j, 1]

                if v == 0:
                    # Not labeled — no position information
                    kps[j] = [-1.0, -1.0, 0.0, 0]
                elif v == 1:
                    # Labeled but occluded — real position, sample depth
                    kps[j, 0] = x
                    kps[j, 1] = y
                    kps[j, 2] = self._sample_depth(depth_map, x, y)
                    kps[j, 3] = 1
                else:
                    # Fully visible — real position, sample depth
                    kps[j, 0] = x
                    kps[j, 1] = y
                    kps[j, 2] = self._sample_depth(depth_map, x, y)
                    kps[j, 3] = 2

            keypoints.append(torch.tensor(kps, dtype=torch.float32))

        return {
            "image":      image,                    # [3, H, W] uint8
            "keypoints":  keypoints,                # list of [17, 4]
            "img_id":     str(img_id),              # string for consistency with virtual
            "num_people": len(keypoints),
        }