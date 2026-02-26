"""Virtual dataset adapter
Reads <guid>_<n>.png + <guid>_<n>.json files produced by the v2 generator
and returns the unified data contract:

    {
        "image":      tensor [3, H, W]        uint8
        "keypoints":  list of tensor [17, 4]  float32  [x, y, z, v]
        "img_id":     str                      GUID
        "num_people": int
    }

Depth (z) is ground-truth world-space distance in metres, taken directly
from the generator annotations — no estimation needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from torchvision.io import read_image


class VirtualAdapter:
    """
    Loads virtual pose samples from a directory of <guid>_<n>.json files.

    Args:
        data_dir:   Directory containing the .png / .json pairs.
                    Defaults to settings.virtual_data_dir.
        min_people: Minimum number of people in a sample (inclusive).
        max_people: Maximum number of people in a sample (inclusive).
                    None means no upper limit.
    """

    def __init__(
        self,
        data_dir: Path,
        min_people: int = 1,
        max_people: Optional[int] = None,
    ):
        self.data_dir   = Path(data_dir)
        self.min_people = min_people
        self.max_people = max_people

        if not self.data_dir.exists():
            raise ValueError(f"Virtual data directory not found: {self.data_dir}")

        self.samples = self._index_samples()

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found in {self.data_dir} "
                f"with person count [{min_people}, {max_people or '∞'}]"
            )

        print(
            f"VirtualAdapter: {len(self.samples)} samples from '{self.data_dir.name}' "
            f"(people: {min_people}–{max_people or '∞'})"
        )

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _index_samples(self) -> List[Path]:
        """
        Collect all JSON files whose filename encodes a person count within
        the requested range.

        Filename format: <guid>_<n>.json  e.g. 97c00739-..._2.json
        The person count <n> is the part after the last underscore.
        """
        samples = []

        for json_path in sorted(self.data_dir.glob("*.json")):
            stem = json_path.stem                   # e.g. "97c00739-..._2"
            parts = stem.rsplit("_", 1)

            if len(parts) != 2:
                continue                             # unexpected filename format

            try:
                n = int(parts[1])
            except ValueError:
                continue

            if n < self.min_people:
                continue
            if self.max_people is not None and n > self.max_people:
                continue

            # Confirm the matching PNG exists
            png_path = json_path.with_suffix(".png")
            if not png_path.exists():
                print(f"  ⚠ Missing PNG for {json_path.name}, skipping")
                continue

            samples.append(json_path)

        return samples

    # ── Public interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        json_path = self.samples[idx]
        png_path  = json_path.with_suffix(".png")

        # ── Load annotation ───────────────────────────────────────────────
        with open(json_path, "r") as f:
            data = json.load(f)

        image_meta  = data["image"]
        annotations = data["annotations"]

        # ── Load image ────────────────────────────────────────────────────
        image = read_image(str(png_path))   # [C, H, W] uint8

        # Normalise channels: grayscale → RGB, RGBA → RGB
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] > 3:
            image = image[:3]

        # ── Extract keypoints ─────────────────────────────────────────────
        # Generator format: flat list of 68 floats [x,y,z,v, x,y,z,v, ...]
        keypoints: List[torch.Tensor] = []

        for ann in annotations:
            kps = torch.tensor(
                ann["keypoints"], dtype=torch.float32
            ).reshape(17, 4)           # [17, 4]  columns: x, y, z, v
            keypoints.append(kps)

        return {
            "image":      image,                        # [3, H, W] uint8
            "keypoints":  keypoints,                    # list of [17, 4]
            "img_id":     image_meta["id"],             # GUID string
            "num_people": image_meta["num_people"],
        }