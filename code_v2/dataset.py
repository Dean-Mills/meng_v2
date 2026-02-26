"""Unified Dataset
Wraps either a VirtualAdapter or CocoAdapter and applies a consistent
resize + pad transform so all downstream components see fixed-size tensors.

Transform:
  - Image resized to fit within target_size x target_size preserving aspect ratio
  - Padded symmetrically to exactly target_size x target_size
  - Keypoint x, y scaled and shifted to match — z and v left untouched
  - Joints with v=0 (sentinel position -1,-1) are not transformed
"""
from __future__ import annotations

from typing import Union

import torch
import torchvision.transforms.functional as TF

from virtual_adapter import VirtualAdapter
from coco_adapter import CocoAdapter

Adapter = Union[VirtualAdapter, CocoAdapter]


class PoseDataset(torch.utils.data.Dataset):
    """
    Unified dataset for pose grouping.

    Args:
        adapter:     A VirtualAdapter or CocoAdapter instance.
        target_size: Image will be resized and padded to target_size x target_size.
    """

    def __init__(self, adapter: Adapter, target_size: int = 512):
        self.adapter     = adapter
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.adapter)

    def __getitem__(self, idx: int) -> dict:
        sample = self.adapter[idx]

        image_t, keypoints_t = self._transform(
            sample["image"],
            sample["keypoints"],
        )

        return {
            "image":      image_t,
            "keypoints":  keypoints_t,
            "img_id":     sample["img_id"],
            "num_people": sample["num_people"],
        }

    # ── Transform ─────────────────────────────────────────────────────────────

    def _transform(self, image, keypoints):
        """
        Resize and pad image to target_size × target_size.
        Scale x, y keypoint coordinates to match.
        z and v columns are untouched.
        Joints with v=0 (position -1,-1) are left as-is.
        """
        _, H, W = image.shape
        target  = self.target_size

        # ── Resize ────────────────────────────────────────────────────────
        scale     = target / max(H, W)
        new_H     = int(H * scale)
        new_W     = int(W * scale)
        image_r   = TF.resize(image, [new_H, new_W], antialias=True)

        # ── Pad ───────────────────────────────────────────────────────────
        pad_left   = (target - new_W) // 2
        pad_top    = (target - new_H) // 2
        pad_right  = target - new_W - pad_left
        pad_bottom = target - new_H - pad_top

        image_p = TF.pad(image_r, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

        # ── Keypoints ─────────────────────────────────────────────────────
        transformed = []
        for kps in keypoints:
            kps = kps.clone()

            # Only transform joints that have a valid position (v > 0)
            valid = kps[:, 3] > 0

            kps[valid, 0] = kps[valid, 0] * scale + pad_left   # x
            kps[valid, 1] = kps[valid, 1] * scale + pad_top    # y
            # kps[:, 2] z depth — untouched
            # kps[:, 3] v visibility — untouched

            transformed.append(kps)

        return image_p, transformed