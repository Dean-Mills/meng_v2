import torch
import torch.nn as nn
import numpy as np
from depth_estimation import estimate_depth
from torch_geometric.data import Data

COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (5, 7), (7, 9),  # left shoulder to left elbow to left wrist
    (6, 8), (8, 10),  # right shoulder to right elbow to right wrist
    (5, 6), (5, 11), (6, 12),  # shoulders to hips
    (11, 13), (13, 15),  # left hip to left knee to left ankle
    (12, 14), (14, 16)  # right hip to right knee to right ankle
]

class KeypointPreprocessor:
    def __init__(self, device='cuda'):
        """
        Simplified preprocessor for person ID prediction.
        
        Args:
            device: Device to use for computations
        """
        self.device = device
        
        # Depth model setup
        print("\nInitializing Depth Estimation Model (MiDaS)...")
        self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", verbose=False)
        self.depth_model.to(self.device)
        self.depth_model.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
        self.depth_transform = midas_transforms.small_transform
        print("Depth model ready.")
    
    def create_mixed_graph(self, image, keypoints_list, min_confidence=0.0):
        image = image.to(self.device)
        depth_map = estimate_depth(self.depth_model, image, self.depth_transform, self.device)
        h, w = depth_map.shape

        node_features = []
        person_labels = []

        for person_idx, keypoints in enumerate(keypoints_list):
            for joint_idx in range(keypoints.shape[0]):
                x, y, visibility = keypoints[joint_idx]

                if visibility <= min_confidence:
                    continue

                xi = min(max(int(x), 0), w - 1)
                yi = min(max(int(y), 0), h - 1)
                depth_value = depth_map[yi, xi].item()

                node_features.append([
                    float(x),
                    float(y),
                    float(depth_value)
                ])
                person_labels.append(person_idx)

        if not node_features:
            return None

        x = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        person_labels = torch.tensor(person_labels, dtype=torch.long, device=self.device)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        return Data(
            x=x,
            edge_index=edge_index,
            person_labels=person_labels,
            num_people=len(keypoints_list)
        )
    
    def process_batch(self, batch):
        """
        Process a batch of data into a list of PyTorch Geometric Data objects.
        Each image becomes one mixed graph with all people's keypoints.
        """
        images = batch['image']
        keypoints_batch = batch['keypoints']
        
        pyg_graphs = []
        
        for img_idx, (image, keypoints_list) in enumerate(zip(images, keypoints_batch)):
            graph_data = self.create_mixed_graph(image, keypoints_list)
            
            if graph_data is not None:  # Skip images with no valid keypoints
                # Add image metadata if needed
                graph_data.img_id = batch['img_id'][img_idx]
                pyg_graphs.append(graph_data)
                
        return pyg_graphs