"""Unified Preprocessor
Converts a batch from the DataLoader into a list of PyTorch Geometric graphs.
One graph per image — all people's joints are nodes in the same graph.

Node features (4 values per node):
    x_norm      — x pixel coordinate normalised to [0, 1]
    y_norm      — y pixel coordinate normalised to [0, 1]
    z_norm      — depth normalised per-graph to [0, 1]
    v_norm      — visibility normalised to [0, 1] (0/0.5/1.0)

Separate tensors per graph:
    joint_types   — [N] long, joint index 0-16 for embedding layer
    person_labels — [N] long, ground truth person assignment for training

Edges:
    kNN graph with k=8 built on (x_norm, y_norm, z_norm)
    Joints with v=0 (sentinel -1,-1) are excluded from the graph entirely
    since they carry no positional information.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch_geometric.data import Data


class PosePreprocessor:
    """
    Converts DataLoader batches into PyTorch Geometric graphs.

    Args:
        image_size:  Target image size used in dataset transform (for x,y normalisation).
        k_neighbors: Number of kNN neighbours per node.
        device:      Device to place graph tensors on.
    """

    def __init__(
        self,
        image_size: int = 512,
        k_neighbors: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.image_size  = image_size
        self.k_neighbors = k_neighbors
        self.device      = device

    # ── kNN graph ─────────────────────────────────────────────────────────────

    def _knn_edges(self, coords: torch.Tensor, k: int) -> torch.Tensor:
        """
        Build a kNN edge index from spatial coordinates.

        Args:
            coords: [N, D] float tensor
            k:      Number of neighbours

        Returns:
            edge_index: [2, N*k] long tensor
        """
        n = coords.size(0)
        k = min(k, n - 1)

        dist = torch.cdist(coords, coords, p=2)
        dist.fill_diagonal_(float("inf"))

        _, indices = dist.topk(k, dim=1, largest=False)

        source = torch.arange(n, device=coords.device).repeat_interleave(k)
        target = indices.flatten()

        return torch.stack([source, target], dim=0)

    # ── Normalisation ─────────────────────────────────────────────────────────

    def _normalise_depth(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Normalise depth values per-graph using min-max over valid joints only.
        Invalid joints (v=0) keep z=0 after normalisation.
        """
        z_norm = torch.zeros_like(z)

        if mask.sum() > 1:
            z_valid   = z[mask]
            z_min     = z_valid.min()
            z_max     = z_valid.max()
            z_range   = z_max - z_min

            if z_range > 1e-6:
                z_norm[mask] = (z_valid - z_min) / z_range
            else:
                z_norm[mask] = 0.0

        return z_norm

    # ── Graph construction ────────────────────────────────────────────────────

    def create_graph(
        self,
        keypoints_list: List[torch.Tensor],
    ) -> Optional[Data]:
        """
        Build a PyG graph from a list of [17, 4] keypoint tensors.

        Joints with v=0 are excluded — they have no position information.
        All other joints (v=1 occluded, v=2 visible) are included as nodes.

        Args:
            keypoints_list: list of [17, 4] tensors, one per person

        Returns:
            PyG Data object or None if fewer than 2 valid joints
        """
        node_features  = []
        joint_types    = []
        person_labels  = []

        for person_idx, kps in enumerate(keypoints_list):
            for joint_idx in range(17):
                x, y, z, v = kps[joint_idx]

                # Skip joints with no position information
                if v == 0:
                    continue

                node_features.append([float(x), float(y), float(z), float(v)])
                joint_types.append(joint_idx)
                person_labels.append(person_idx)

        if len(node_features) < 2:
            return None

        # ── Tensors ───────────────────────────────────────────────────────
        x_raw     = torch.tensor(node_features, dtype=torch.float32)
        jt        = torch.tensor(joint_types,   dtype=torch.long)
        pl        = torch.tensor(person_labels, dtype=torch.long)

        # ── Normalise x, y ────────────────────────────────────────────────
        x_norm    = x_raw[:, 0] / self.image_size
        y_norm    = x_raw[:, 1] / self.image_size

        # ── Normalise z per-graph ─────────────────────────────────────────
        valid_mask = torch.ones(len(node_features), dtype=torch.bool)  # all included
        z_norm     = self._normalise_depth(x_raw[:, 2], valid_mask)

        # ── Normalise v ───────────────────────────────────────────────────
        v_norm    = x_raw[:, 3] / 2.0   # 0→0.0, 1→0.5, 2→1.0

        # ── Final node feature matrix [N, 4] ──────────────────────────────
        node_x = torch.stack([x_norm, y_norm, z_norm, v_norm], dim=1).to(self.device)
        jt     = jt.to(self.device)
        pl     = pl.to(self.device)

        # ── kNN edges on spatial coords ───────────────────────────────────
        spatial    = node_x[:, :3]   # x_norm, y_norm, z_norm
        edge_index = self._knn_edges(spatial, k=self.k_neighbors)

        return Data(
            x=node_x,
            edge_index=edge_index,
            joint_types=jt,
            person_labels=pl,
            num_people=len(keypoints_list),
        )

    # ── Batch processing ──────────────────────────────────────────────────────

    def process_batch(self, batch: Dict[str, Any]) -> List[Data]:
        """
        Process a full DataLoader batch into a list of PyG graphs.
        Images with fewer than 2 valid joints are skipped.

        Args:
            batch: output of pose_collate_fn

        Returns:
            list of PyG Data objects, one per valid image in the batch
        """
        graphs = []

        for keypoints_list, img_id in zip(batch["keypoints"], batch["img_id"]):
            graph = self.create_graph(keypoints_list)

            if graph is not None:
                graph.img_id = img_id
                graphs.append(graph)

        return graphs