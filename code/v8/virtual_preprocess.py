# virtual_preprocess.py
import torch
from torch_geometric.data import Data

COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


def knn_graph_pure_torch(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Build k-NN graph using pure PyTorch (no torch-cluster needed).
    
    Args:
        x: [N, D] node features (uses all dims for distance)
        k: Number of neighbors
    
    Returns:
        edge_index: [2, N*k] edges
    """
    n = x.size(0)
    
    # Compute pairwise distances [N, N]
    dist = torch.cdist(x, x, p=2)
    
    # Set self-distance to inf so we don't pick ourselves
    dist.fill_diagonal_(float('inf'))
    
    # Get k nearest neighbors for each node [N, k]
    _, indices = dist.topk(k, dim=1, largest=False)
    
    # Build edge index
    # Source: each node index repeated k times [0,0,0,1,1,1,2,2,2,...]
    source = torch.arange(n, device=x.device).repeat_interleave(k)
    # Target: the k neighbors for each node, flattened
    target = indices.flatten()
    
    edge_index = torch.stack([source, target], dim=0)
    
    return edge_index


class VirtualKeypointPreprocessor:
    """
    Preprocessor for virtual dataset.
    
    Converts keypoints + distances into PyTorch Geometric graphs with:
        - Node features: [x_norm, y_norm, depth_norm, confidence]
        - Joint type indices for embedding layer
        - k-NN edges based on 3D spatial proximity
        - Ground truth person labels
    """
    
    def __init__(self, device='cuda', k_neighbors=8, image_size=512):
        self.device = device
        self.k_neighbors = k_neighbors
        self.image_size = image_size
        print(f"VirtualKeypointPreprocessor initialized (k={k_neighbors}, image_size={image_size})")
    
    def create_graph(self, keypoints_list, distances, min_visibility=1):
        """
        Create a PyG graph from keypoints and distances.
        """
        node_features = []
        joint_types = []
        person_labels = []
        
        # Normalize depth across scene
        all_distances = list(distances)
        min_dist, max_dist = min(all_distances), max(all_distances)
        dist_range = max_dist - min_dist if max_dist > min_dist else 1.0
        
        for person_idx, (keypoints, distance) in enumerate(zip(keypoints_list, distances)):
            depth_norm = (distance - min_dist) / dist_range
            
            for joint_idx in range(keypoints.shape[0]):
                x, y, visibility = keypoints[joint_idx]
                
                if visibility < min_visibility:
                    continue
                
                node_features.append([
                    float(x) / self.image_size,
                    float(y) / self.image_size,
                    float(depth_norm),
                    float(visibility) / 2.0,
                ])
                joint_types.append(joint_idx)
                person_labels.append(person_idx)
        
        if len(node_features) < 2:
            return None
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float32, device=self.device)
        joint_types = torch.tensor(joint_types, dtype=torch.long, device=self.device)
        person_labels = torch.tensor(person_labels, dtype=torch.long, device=self.device)
        
        # Build k-NN graph using pure PyTorch
        spatial_coords = x[:, :3]  # x, y, depth
        k = min(self.k_neighbors, len(node_features) - 1)
        edge_index = knn_graph_pure_torch(spatial_coords, k=k)
        
        return Data(
            x=x,
            joint_types=joint_types,
            edge_index=edge_index,
            person_labels=person_labels,
            num_people=len(keypoints_list)
        )
    
    def process_batch(self, batch):
        """
        Process a batch from the dataloader into PyG graphs.
        """
        keypoints_batch = batch['keypoints']
        distances_batch = batch['distances']
        img_ids = batch['img_id']
        
        graphs = []
        
        for img_idx, (keypoints_list, distances) in enumerate(zip(keypoints_batch, distances_batch)):
            graph = self.create_graph(keypoints_list, distances)
            
            if graph is not None:
                graph.img_id = img_ids[img_idx]
                graphs.append(graph)
        
        return graphs