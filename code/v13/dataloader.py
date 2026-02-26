import torch
from torch.utils.data import DataLoader
from dataset import CocoKeypointsDataset
from typing import Literal, List, Dict, Any
from settings import settings


def coco_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle batches with a variable number of persons.
    """

    # Stack images, which are all the same size
    images = torch.stack([item["image"] for item in batch], dim=0)

    # For keypoints, can't stack them directly.
    # Keep them as a list of lists of tensors.
    # The outer list represents the batch, the inner list is the people in an image.
    keypoints = [item["keypoints"] for item in batch]

    # Other metadata can be collected into lists
    img_ids = [item["img_id"] for item in batch]
    ann_ids = [item["ann_ids"] for item in batch]

    return {
        "image": images,
        "keypoints": keypoints,
        "img_id": img_ids,
        "ann_ids": ann_ids,
    }


def create_coco_dataloader(
    split: Literal["train", "val"],
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for the COCO keypoints dataset.

    Args:
        split: Dataset split to use ('train' or 'val')
        batch_size: Number of samples in each batch
        shuffle: Whether to shuffle the dataset
        num_workers: Number of subprocesses for data loading

    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = CocoKeypointsDataset(split=split)

    dataloader = DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=coco_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
    )

    return dataloader
