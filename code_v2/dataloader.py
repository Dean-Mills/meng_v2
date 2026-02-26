"""Unified DataLoader
Wraps a PoseDataset and handles variable-length keypoints via a custom
collate function. Images are stacked into a single tensor, keypoints are
kept as a list of lists since person count varies per image.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from dataset import PoseDataset


def pose_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate a batch of samples from PoseDataset.

    Images are stacked — they are all the same size after the dataset transform.
    Keypoints are kept as a list of lists — person count varies per image.

    Returns:
        {
            "image":      tensor [B, 3, H, W]
            "keypoints":  list[list[tensor [17, 4]]]   outer = batch, inner = people
            "img_id":     list[str]
            "num_people": list[int]
        }
    """
    return {
        "image":      torch.stack([s["image"] for s in batch], dim=0),
        "keypoints":  [s["keypoints"]  for s in batch],
        "img_id":     [s["img_id"]     for s in batch],
        "num_people": [s["num_people"] for s in batch],
    }


def create_dataloader(
    dataset: PoseDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a DataLoader for a PoseDataset.

    Args:
        dataset:     A PoseDataset wrapping either adapter.
        batch_size:  Number of samples per batch.
        shuffle:     Whether to shuffle the dataset each epoch.
        num_workers: Number of subprocesses for data loading.

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pose_collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )