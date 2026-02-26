#virtual_dataloader.py
import torch
from torch.utils.data import DataLoader
from virtual_dataset import VirtualKeypointsDataset
from typing import Literal, List, Dict, Any
from settings import settings


def virtual_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for virtual dataset with distance information.
    """
    # Stack images
    images = torch.stack([item["image"] for item in batch], dim=0)

    # Keep keypoints as list of lists
    keypoints = [item["keypoints"] for item in batch]
    
    # Keep distances as list of lists
    distances = [item["distances"] for item in batch]

    # Metadata
    img_ids = [item["img_id"] for item in batch]
    ann_ids = [item["ann_ids"] for item in batch]

    return {
        "image": images,
        "keypoints": keypoints,
        "distances": distances,  # NEW!
        "img_id": img_ids,
        "ann_ids": ann_ids,
    }


def create_virtual_dataloader(
    split: Literal["two_persons", "three_persons", "four_persons", "mixed"],
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for the virtual keypoints dataset.
    
    Args:
        split: Dataset split to use ('two_persons', 'three_persons', 'four_persons')
        shuffle: Whether to shuffle the dataset
        num_workers: Number of subprocesses for data loading
    
    Returns:
        DataLoader for virtual dataset
    """
    dataset = VirtualKeypointsDataset(split=split)

    dataloader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=virtual_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
    )

    return dataloader