import torch
import torch.nn as nn
import numpy as np
from depth_estimation import estimate_depth
from torch_geometric.data import Data

# For reference
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Define skeleton connections
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
        
        # Create learnable joint-type embeddings
        self.joint_embeddings = nn.Embedding(num_keypoint_types, embedding_dim)
        self.joint_embeddings.to(device)
        
        print("\nInitializing Depth Estimation Model (MiDaS)...")
        self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", verbose=False)
        self.depth_model.to(self.device)
        self.depth_model.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", verbose=False)
        self.depth_transform = midas_transforms.small_transform
        print("Depth model ready.")

    def create_edge_index(self, node_info, num_nodes):
        """
        Create graph connectivity in COO format for PyG. Ensures edges are bidirectional.
        """
        edge_list = []
        info_to_node_idx = {info: idx for idx, info in enumerate(node_info)}
        
        # Iterate through skeleton connections for each person
        for person_id in set(p_id for p_id, kp_id in node_info):
            for kp_start, kp_end in SKELETON_CONNECTIONS:
                
                # Check if both keypoints for this connection exist for the person
                start_node_key = (person_id, kp_start)
                end_node_key = (person_id, kp_end)
                
                if start_node_key in info_to_node_idx and end_node_key in info_to_node_idx:
                    start_node_idx = info_to_node_idx[start_node_key]
                    end_node_idx = info_to_node_idx[end_node_key]
                    
                    # Add edges in both directions
                    edge_list.append((start_node_idx, end_node_idx))
                    edge_list.append((end_node_idx, start_node_idx))

        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
            
        # Convert to tensor. The .t() transposes it to the [2, num_edges] format.
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t().contiguous()
        
        return edge_index
    
    def extract_node_features(self, image, keypoints_list, min_confidence=0.0):
        """
        Extract node features from an image and its keypoints.
        
        Args:
            image: Image tensor [C, H, W]
            keypoints_list: List of keypoint tensors, each [17, 3] for each person
            min_confidence: Minimum visibility score to include a keypoint
            
        Returns:
            node_features: Tensor of shape [N, 4 + embedding_dim] where N is total valid keypoints
            node_info: List of tuples (person_idx, keypoint_idx) for tracking
        """
        # Move image to device
        image = image.to(self.device)
        
        # Estimate depth map
        depth_map = estimate_depth(self.depth_model, image, self.depth_transform, self.device)
        
        # Collect all valid keypoints
        nodes = []
        node_info = []
        
        for person_idx, keypoints in enumerate(keypoints_list):
            # keypoints shape: [17, 3] where 3 = [x, y, visibility]
            for kp_idx in range(keypoints.shape[0]):
                x, y, visibility = keypoints[kp_idx]
                
                # Skip keypoints with low visibility
                if visibility <= min_confidence:
                    continue
                
                # Get depth value at keypoint location
                # Ensure coordinates are within bounds
                h, w = depth_map.shape
                x_int = int(torch.clamp(x, 0, w-1))
                y_int = int(torch.clamp(y, 0, h-1))
                
                depth_value = depth_map[y_int, x_int].item()
                
                # Use visibility as confidence (normalize to 0-1 range)
                # visibility: 0=not labeled, 1=occluded, 2=visible
                confidence = visibility / 2.0
                
                # Create node features [x, y, d, c]
                node_feat = torch.tensor([x.item(), y.item(), depth_value, confidence], 
                                        dtype=torch.float32, device=self.device)
                
                nodes.append(node_feat)
                node_info.append((person_idx, kp_idx))
        
        if len(nodes) == 0:
            # Return empty tensors if no valid keypoints
            return torch.zeros((0, 4 + self.embedding_dim), device=self.device), []
        
        # Stack node features
        node_features = torch.stack(nodes)  # [N, 4]
        
        # Add joint-type embeddings
        joint_types = torch.tensor([info[1] for info in node_info], 
                                  dtype=torch.long, device=self.device)
        embeddings = self.joint_embeddings(joint_types)  # [N, embedding_dim]
        
        # Concatenate basic features with embeddings
        node_features = torch.cat([node_features, embeddings], dim=1)  # [N, 4 + embedding_dim]
        
        return node_features.detach(), node_info
    
    # No need for this function now that I've implemented create_edge_index
    # I needed to do this for torch_geometric.data Data class compatibility.
    
    # def create_adjacency_matrix(self, node_info, num_nodes):
    #     """
    #     Create adjacency matrix based on skeleton connections.
        
    #     Args:
    #         node_info: List of tuples (person_idx, keypoint_idx)
    #         num_nodes: Total number of nodes
            
    #     Returns:
    #         adj_matrix: Adjacency matrix [num_nodes, num_nodes]
    #     """
    #     adj_matrix = torch.zeros((num_nodes, num_nodes), device=self.device)
        
    #     # Create a mapping from (person_idx, keypoint_idx) to node index
    #     info_to_node_idx = {info: idx for idx, info in enumerate(node_info)}
        
    #     # Add edges based on skeleton connections
    #     for (p1_idx, kp1_idx), node1_idx in info_to_node_idx.items():
    #         for (p2_idx, kp2_idx), node2_idx in info_to_node_idx.items():
    #             # Only connect keypoints from the same person
    #             if p1_idx != p2_idx:
    #                 continue
                
    #             # Check if these keypoints are connected in the skeleton
    #             if (kp1_idx, kp2_idx) in SKELETON_CONNECTIONS or \
    #                (kp2_idx, kp1_idx) in SKELETON_CONNECTIONS:
    #                 adj_matrix[node1_idx, node2_idx] = 1.0
    #                 adj_matrix[node2_idx, node1_idx] = 1.0
        
    #     return adj_matrix
    
    def process_batch(self, batch):
        """
        Process a batch of data into a list of PyTorch Geometric Data objects.
        """
        images = batch['image']
        keypoints_batch = batch['keypoints']
        
        pyg_graphs = []
        
        for img_idx, (image, keypoints_list) in enumerate(zip(images, keypoints_batch)):
            node_features, node_info = self.extract_node_features(image, keypoints_list)
            
            if len(node_info) == 0:
                continue
            
            # Create edge index instead of adjacency matrix
            num_nodes = node_features.shape[0]
            edge_index = self.create_edge_index(node_info, num_nodes)
            
            # Create a PyG Data object
            graph_data = Data(
                x=node_features,
                edge_index=edge_index,
                img_id=batch['img_id'][img_idx],
                num_people=len(keypoints_list)
            )
            pyg_graphs.append(graph_data)
            
        return pyg_graphs