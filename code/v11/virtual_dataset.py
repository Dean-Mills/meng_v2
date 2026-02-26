# virtual_dataset.py
import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.io import read_image
import numpy as np
import matplotlib.pyplot as plt
from settings import settings
from typing import Literal
from pathlib import Path


class VirtualKeypointsDataset(Dataset):
    def __init__(
        self, 
        split: Literal["two_persons", "three_persons", "four_persons"] = "two_persons",
        virtual_data_dir: Path = None
    ):
        """
        Dataset for virtual Mixamo-generated keypoint data.
        
        Args:
            split: Which folder to use ('two_persons', 'three_persons', 'four_persons')
            virtual_data_dir: Root directory containing virtual data folders
        """
        self.split = split
        
        # Default to /data/virtual if not specified
        if virtual_data_dir is None:
            virtual_data_dir = settings.data_dir / "virtual"
        
        self.data_dir = Path(virtual_data_dir) / split
        
        if not self.data_dir.exists():
            raise ValueError(f"Virtual data directory not found: {self.data_dir}")
        
        # Find all JSON files
        self.annotation_files = sorted(list(self.data_dir.glob("*.json")))
        
        if len(self.annotation_files) == 0:
            raise ValueError(f"No JSON files found in {self.data_dir}")
        
        print(f"Loaded {len(self.annotation_files)} annotations from '{split}' split")
    
    def __len__(self):
        return len(self.annotation_files)
    
    def _transform_data(self, image, keypoints_list):
        """
        Same transformation as COCO dataset: resize and pad to target size.
        """
        target_size = settings.target_image_size
        _, original_height, original_width = image.shape

        scale = float(target_size) / max(original_width, original_height)
        new_width, new_height = int(original_width * scale), int(original_height * scale)

        image_resized = F.resize(image, (new_height, new_width), antialias=True)

        pad_left = (target_size - new_width) // 2
        pad_top = (target_size - new_height) // 2
        pad_right = target_size - new_width - pad_left
        pad_bottom = target_size - new_height - pad_top
        padding = (pad_left, pad_top, pad_right, pad_bottom)

        image_padded = F.pad(image_resized, padding, fill=0)

        transformed_keypoints = []
        for kps in keypoints_list:
            kps[:, 0] = kps[:, 0] * scale + pad_left
            kps[:, 1] = kps[:, 1] * scale + pad_top
            transformed_keypoints.append(kps)

        return image_padded, transformed_keypoints
    
    def __getitem__(self, idx):
        json_path = self.annotation_files[idx]
        
        # Load annotation
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get image path (same name, .png extension)
        image_path = json_path.with_suffix('.png')
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = read_image(str(image_path))
        
        # Handle grayscale or RGBA
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] > 3:
            image = image[:3, :, :]
        
        # Extract keypoints for all people in the image
        keypoints_list = []
        valid_ann_ids = []
        distances = []  # Store distances for later use
        
        for ann in data['annotations']:
            # Get keypoints [x, y, v, x, y, v, ...] -> reshape to [17, 3]
            kps = np.array(ann['keypoints']).reshape(-1, 3)
            keypoints_list.append(torch.tensor(kps, dtype=torch.float32))
            valid_ann_ids.append(ann['id'])
            distances.append(ann['distance_to_camera'])
        
        # Transform image and keypoints
        image_transformed, keypoints_transformed = self._transform_data(
            image, keypoints_list
        )
        
        return {
            "image": image_transformed,
            "keypoints": keypoints_transformed,
            "img_id": data['image']['id'],
            "ann_ids": valid_ann_ids,
            "distances": distances  # NEW: ground truth distances
        }
    
    def visualize_item(self, idx, save_to_file=True):
        """
        Visualize a transformed image and its keypoints (same as COCO).
        """
        data = self[idx]
        image = data["image"]
        keypoints = data["keypoints"]
        img_id = data["img_id"]

        img_np = image.permute(1, 2, 0).numpy()

        plt.figure(figsize=(10, 10))
        plt.imshow(img_np)
        plt.title(
            f"Virtual Image ID: {img_id} ({settings.target_image_size}x{settings.target_image_size})"
        )
        plt.axis("off")

        colors = plt.cm.jet(np.linspace(0, 1, len(keypoints)))

        # COCO skeleton connections
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
            (13, 15), (12, 14), (14, 16),
        ]

        for i, kps in enumerate(keypoints):
            color = colors[i]
            visible_kps = kps[kps[:, 2] > 0]
            plt.scatter(
                visible_kps[:, 0], visible_kps[:, 1],
                c=[color], s=40, edgecolors="black", linewidth=1,
            )

            for conn in connections:
                if kps[conn[0], 2] > 0 and kps[conn[1], 2] > 0:
                    plt.plot(
                        [kps[conn[0], 0], kps[conn[1], 0]],
                        [kps[conn[0], 1], kps[conn[1], 1]],
                        color=color, linewidth=2,
                    )

        if save_to_file:
            output_path = settings.output_dir / f"vis_virtual_{img_id}.jpg"
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
            print(f"Visualization saved to: {output_path}")

        plt.show()
        plt.close()