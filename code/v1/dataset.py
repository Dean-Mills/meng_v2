import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.io import read_image
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
from settings import settings
from typing import Literal

class CocoKeypointsDataset(Dataset):
    def __init__(self, split=Literal["val","train"]):
        """
        Args:
            settings: Settings object with paths
            split: 'train' or 'val'
        """
        self.split = split
        
        if split == 'val':
            self.img_dir = settings.coco_val_dir
            self.ann_file = os.path.join(settings.coco_annotations_dir, f'person_keypoints_val2017.json')
        elif split == 'train':
            self.img_dir = settings.coco_train_dir
            self.ann_file = os.path.join(settings.coco_annotations_dir, f'person_keypoints_train2017.json')
        else:
            raise ValueError(f"Split {split} not recognized. Use 'train' or 'val'.")
        
        self.coco = COCO(self.ann_file)
        
        cat_ids = self.coco.getCatIds(catNms=['person'])
        self.img_ids = self.coco.getImgIds(catIds=cat_ids)
        
        print(f"Loaded {len(self.img_ids)} images with person keypoints")
    
    def __len__(self):
        return len(self.img_ids)
    
    def _transform_data(self, image, keypoints_list):

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
        """
        Returns a dictionary containing the transformed image and keypoints.
        """
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = read_image(img_path)

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] > 3:
            image = image[:3, :, :]
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        annotations = self.coco.loadAnns(ann_ids)
        
        keypoints_list = []
        valid_ann_ids = []
        for ann in annotations:
            if ann['category_id'] == 1 and 'keypoints' in ann and sum(ann['keypoints']) > 0:
                kps = np.array(ann['keypoints']).reshape(-1, 3)
                keypoints_list.append(torch.tensor(kps, dtype=torch.float32))
                valid_ann_ids.append(ann['id'])
        
        image_transformed, keypoints_transformed = self._transform_data(image, keypoints_list)
        
        return {
            'image': image_transformed,
            'keypoints': keypoints_transformed,
            'img_id': img_id,
            'ann_ids': valid_ann_ids
        }
    
    def visualize_item(self, idx, save_to_file=True):
            """
            Visualize a transformed image and its keypoints.
            """
            data = self[idx]
            image = data['image']
            keypoints = data['keypoints']
            img_id = data['img_id']
            
            img_np = image.permute(1, 2, 0).numpy()
            
            plt.figure(figsize=(10, 10))
            plt.imshow(img_np)
            plt.title(f"Image ID: {img_id} (Transformed to {settings.target_image_size}x{settings.target_image_size})")
            plt.axis('off')
            
            colors = plt.cm.jet(np.linspace(0, 1, len(keypoints)))
            
            connections = [
                (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
                (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
            ]
            
            for i, kps in enumerate(keypoints):
                color = colors[i]
                visible_kps = kps[kps[:, 2] > 0]
                plt.scatter(visible_kps[:, 0], visible_kps[:, 1], c=[color], s=40, edgecolors='black', linewidth=1)
                
                for conn in connections:
                    if kps[conn[0], 2] > 0 and kps[conn[1], 2] > 0:
                        plt.plot(
                            [kps[conn[0], 0], kps[conn[1], 0]],
                            [kps[conn[0], 1], kps[conn[1], 1]],
                            color=color, linewidth=2
                        )
            
            if save_to_file:
                output_path = settings.output_dir / f"vis_transformed_{img_id}.jpg"
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
                # The print statement is now correctly inside the 'if' block.
                print(f"Visualization saved to: {output_path}")

            plt.show()

            plt.close()