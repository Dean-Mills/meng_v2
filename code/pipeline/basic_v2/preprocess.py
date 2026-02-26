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
    def __init__(self, num_keypoint_types=17, embedding_dim=32, device='cuda'):
        """
        Initialize the preprocessor with learnable joint-type embeddings.
        
        Args:
            num_keypoint_types: Number of different keypoint types (17 for COCO)
            embedding_dim: Dimension of the joint-type embeddings
            device: Device to use for computations
        """
        self.device = device
        self.num_keypoint_types = num_keypoint_types
        self.embedding_dim = embedding_dim
        
        self.joint_embeddings = nn.Embedding(num_keypoint_types, embedding_dim)
        self.joint_embeddings.to(device)
        
        # Depth model setup (keeping your existing code)
        print("\nInitializing Depth Estimation Model (MiDaS)...")
        self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", verbose=False)
        self.depth_model.to(self.device)
        self.depth_model.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
        self.depth_transform = midas_transforms.small_transform
        print("Depth model ready.")
    
    def create_mixed_graph(self, image, keypoints_list, min_confidence=0.0):
        """
        Create a single graph containing all people's keypoints
        
        Args:
            image: Image tensor [C, H, W]
            keypoints_list: List of keypoint tensors, each [17, 3] for each person
            min_confidence: Minimum visibility score to include a keypoint
            
        Returns:
            Data object with:
            - x: node features [N, 4 + embedding_dim] 
            - edge_index: graph connectivity [2, num_edges]
            - person_labels: which person each node belongs to [N]
            - num_people: total number of people
        """
        image = image.to(self.device)
        
        depth_map = estimate_depth(self.depth_model, image, self.depth_transform, self.device)
        
        nodes = []
        node_info = []  # (person_idx, keypoint_idx) 
        person_labels = []  # Which person each node belongs to
        
        # Loop through all people and all their keypoints
        for person_idx, keypoints in enumerate(keypoints_list):
            for kp_idx in range(keypoints.shape[0]):
                x, y, visibility = keypoints[kp_idx]
                
                # Skip keypoints with low visibility
                if visibility <= min_confidence:
                    continue
                    
                # Get depth value (your existing code)
                h, w = depth_map.shape
                x_int = int(torch.clamp(x, 0, w-1))
                y_int = int(torch.clamp(y, 0, h-1))
                depth_value = depth_map[y_int, x_int].item()
                
                # Create node features [x, y, depth, confidence]
                confidence = visibility / 2.0
                node_feat = torch.tensor([x.item(), y.item(), depth_value, confidence], 
                                        dtype=torch.float32, device=self.device)
                
                nodes.append(node_feat)
                node_info.append((person_idx, kp_idx))
                person_labels.append(person_idx)  # Track which person this node belongs to
        
        # Handle empty case
        if len(nodes) == 0:
            return None  # or return empty Data object?
        
        # Stack node features
        node_features = torch.stack(nodes)  # [N, 4]
        
        # Add joint-type embeddings
        joint_types = torch.tensor([info[1] for info in node_info], 
                                dtype=torch.long, device=self.device)
        embeddings = self.joint_embeddings(joint_types)  # [N, embedding_dim]
        
        # Concatenate basic features with embeddings
        node_features = torch.cat([node_features, embeddings], dim=1)  # [N, 4 + embedding_dim]
        
        # Create edges (we need to decide on this)
        num_nodes = node_features.shape[0]
        edge_index = self.create_edge_index(node_info, num_nodes)
        
        # Create person labels tensor
        person_labels = torch.tensor(person_labels, dtype=torch.long, device=self.device)
        
        # Return PyG Data object
        return Data(
            x=node_features.detach(),
            edge_index=edge_index,
            person_labels=person_labels,
            num_people=len(keypoints_list)
        )
        
    # I'm using this function here so I can modify it later
    # I'm basically returning no edges initially and will let the GAT attention mechanism learn the connections
    # NOTE: In the future I might want to do proximity based edges or something similar which will be here
    def create_edge_index(self, node_info, num_nodes):
        """
        Create edge index for mixed graph. 
        Using no initial edges - let GAT attention learn all connections.
        
        Args:
            node_info: List of tuples (person_idx, keypoint_idx) - not used but kept for consistency
            num_nodes: Total number of nodes - not used but kept for consistency
            
        Returns:
            edge_index: Empty edge tensor [2, 0] - no initial connections
        """
        # No initial edges - let GAT attention mechanism figure out all connections
        return torch.empty((2, 0), dtype=torch.long, device=self.device)
    
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