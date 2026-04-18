"""
Dataset for pre-cached MobileNet (or other) features.

Loads .pt files produced by `cache_mobilenet_features.py` and yields
ready-to-use PyG Data objects with a `features` attribute on each node.

This bypasses the standard adapter/preprocessor pipeline because the
keypoints, person labels, and features are all already computed and we
don't need image transforms.

Output Data object:
    x:             [N, D_pos] node positional features (x, y, [z], v)
    features:      [N, D_feat] cached visual features per node
    edge_index:    [2, N*k] kNN edges
    joint_types:   [N] long
    person_labels: [N] long
    num_people:    int
    img_id:        int
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from torch_geometric.data import Data


class CachedFeaturesDataset(torch.utils.data.Dataset):
    """
    Args:
        cache_dir:  Directory of .pt files from cache_mobilenet_features.py
        k_neighbors: Number of kNN neighbours per node
        use_depth:   If False, drops the z coordinate from node features
        max_people:  Optional cap (skip images with more people than this)
        min_people:  Skip images with fewer people than this
    """

    def __init__(
        self,
        cache_dir: Path,
        k_neighbors: int = 8,
        use_depth: bool = False,
        max_people: Optional[int] = None,
        min_people: int = 1,
    ):
        self.cache_dir   = Path(cache_dir)
        self.k_neighbors = k_neighbors
        self.use_depth   = use_depth
        self.max_people  = max_people
        self.min_people  = min_people

        if not self.cache_dir.exists():
            raise ValueError(f"Cache directory not found: {self.cache_dir}")

        # Index all valid samples
        self.files: List[Path] = []
        for pt_path in sorted(self.cache_dir.glob("*.pt")):
            self.files.append(pt_path)

        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {self.cache_dir}")

        print(
            f"CachedFeaturesDataset: {len(self.files)} samples from "
            f"'{self.cache_dir.name}'"
        )

    def __len__(self) -> int:
        return len(self.files)

    def _knn_edges(self, coords: torch.Tensor, k: int) -> torch.Tensor:
        n = coords.size(0)
        k = min(k, n - 1)

        dist = torch.cdist(coords, coords, p=2)
        dist.fill_diagonal_(float("inf"))

        _, indices = dist.topk(k, dim=1, largest=False)

        source = torch.arange(n, device=coords.device).repeat_interleave(k)
        target = indices.flatten()

        return torch.stack([source, target], dim=0)

    def __getitem__(self, idx: int) -> Data:
        sample = torch.load(self.files[idx], weights_only=False)

        keypoints     = sample["keypoints"]      # [N, 4] (x, y, z, v)
        joint_types   = sample["joint_types"]    # [N]
        person_labels = sample["person_labels"]  # [N]
        features      = sample["features"]       # [N, D_feat]
        img_size      = sample["img_size"]       # (H, W)
        num_people    = sample["num_people"]
        img_id        = sample["img_id"]

        H, W = img_size
        scale = max(H, W)  # use longer edge for normalisation
        x_norm = keypoints[:, 0] / scale
        y_norm = keypoints[:, 1] / scale
        z_raw  = keypoints[:, 2]
        v_raw  = keypoints[:, 3]

        # Normalise visibility 0/1/2 -> 0/0.5/1
        v_norm = v_raw / 2.0

        # Per-graph z normalisation (only over valid keypoints, but here all are valid)
        if z_raw.numel() > 1 and (z_raw.max() - z_raw.min()) > 1e-6:
            z_norm = (z_raw - z_raw.min()) / (z_raw.max() - z_raw.min())
        else:
            z_norm = torch.zeros_like(z_raw)

        if self.use_depth:
            node_x = torch.stack([x_norm, y_norm, z_norm, v_norm], dim=1)
            spatial = node_x[:, :3]
        else:
            node_x = torch.stack([x_norm, y_norm, v_norm], dim=1)
            spatial = node_x[:, :2]

        edge_index = self._knn_edges(spatial, k=self.k_neighbors)

        return Data(
            x=node_x,
            features=features,
            edge_index=edge_index,
            joint_types=joint_types,
            person_labels=person_labels,
            num_people=num_people,
            img_id=img_id,
        )
