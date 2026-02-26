# virtual_preprocess.py
import torch
from torch_geometric.data import Data

COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

class VirtualKeypointPreprocessor:
    def __init__(self, device='cuda'):
        """
        Preprocessor for virtual dataset - uses ground truth distance instead of depth estimation.
        
        Args:
            device: Device to use for computations
        """
        self.device = device
        print("\nVirtual Preprocessor initialized (no depth model needed)")
    
    def create_mixed_graph(self, image, keypoints_list, distances, min_confidence=0.0):
        """
        Create graph with ground truth distances.
        
        Args:
            image: Image tensor [C, H, W] (not used, kept for compatibility)
            keypoints_list: List of keypoint tensors [17, 3] for each person
            distances: List of ground truth distances from camera
            min_confidence: Minimum visibility threshold
        
        Returns:
            PyTorch Geometric Data object
        """
        node_features = []
        person_labels = []

        for person_idx, (keypoints, distance) in enumerate(zip(keypoints_list, distances)):
            for joint_idx in range(keypoints.shape[0]):
                x, y, visibility = keypoints[joint_idx]

                if visibility <= min_confidence:
                    continue

                # Use ground truth distance for all keypoints of this person
                node_features.append([
                    float(x),
                    float(y),
                    float(distance)  # Ground truth distance!
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
        Process a batch of virtual data into PyTorch Geometric Data objects.
        
        Args:
            batch: Dict with keys 'image', 'keypoints', 'distances', 'img_id'
        
        Returns:
            List of PyTorch Geometric Data objects
        """
        images = batch['image']
        keypoints_batch = batch['keypoints']
        distances_batch = batch['distances']
        
        pyg_graphs = []
        
        for img_idx, (image, keypoints_list, distances) in enumerate(
            zip(images, keypoints_batch, distances_batch)
        ):
            graph_data = self.create_mixed_graph(image, keypoints_list, distances)
            
            if graph_data is not None:
                # Add image metadata
                graph_data.img_id = batch['img_id'][img_idx]
                pyg_graphs.append(graph_data)
                
        return pyg_graphs