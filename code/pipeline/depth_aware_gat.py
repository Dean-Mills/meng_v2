import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple, Optional


class DepthAwareGATLayer(MessagePassing):
    """
    Custom GAT layer that uses edge affinity features for attention computation
    """
    def __init__(self, in_channels, out_channels, edge_features=4, heads=4, dropout=0.6):
        super(DepthAwareGATLayer, self).__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # Linear transformations for node features
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention mechanism parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        # Edge feature transformation
        self.edge_lin = nn.Linear(edge_features, heads, bias=False)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.edge_lin.weight)
        
    def forward(self, x, edge_index, edge_attr):
        # Add self-loops
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value=0., num_nodes=x.size(0)
        )
        
        # Linear transformation
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        
        # Compute attention scores
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        # Compute attention scores
        alpha = (x_i * self.att_src).sum(dim=-1) + (x_j * self.att_dst).sum(dim=-1)
        
        # Include edge features in attention
        if edge_attr is not None and edge_attr.size(0) > 0:
            edge_scores = self.edge_lin(edge_attr)
            alpha = alpha + edge_scores
            
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        alpha = torch.softmax(alpha, dim=0)
        
        # Return weighted message
        return x_j * alpha.unsqueeze(-1)
        
    def update(self, aggr_out):
        # Average over heads
        aggr_out = aggr_out.mean(dim=1)
        return aggr_out


class JointTypeEmbedding(nn.Module):
    """Learnable embeddings for each joint type"""
    def __init__(self, num_joint_types=17, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(num_joint_types, embedding_dim)
        
    def forward(self, joint_indices):
        return self.embedding(joint_indices)


class DepthAwarePoseGAT(nn.Module):
    """
    Depth-aware Graph Attention Network for multi-person pose estimation
    """
    def __init__(self, 
                 node_features=4,  # x, y, depth, confidence
                 joint_embedding_dim=16,
                 hidden_dim=64,
                 output_dim=128,
                 num_heads=4,
                 num_layers=3,
                 dropout=0.3):
        super().__init__()
        
        self.joint_embedding = JointTypeEmbedding(17, joint_embedding_dim)
        
        # Input dimension = node_features + joint_embedding_dim
        input_dim = node_features + joint_embedding_dim
        
        # Build GAT layers
        self.gat_layers = nn.ModuleList()
        
        # First layer
        self.gat_layers.append(
            DepthAwareGATLayer(input_dim, hidden_dim, edge_features=4, 
                              heads=num_heads, dropout=dropout)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                DepthAwareGATLayer(hidden_dim, hidden_dim, edge_features=4,
                                  heads=num_heads, dropout=dropout)
            )
        
        # Output layer
        self.gat_layers.append(
            DepthAwareGATLayer(hidden_dim, output_dim, edge_features=4,
                              heads=1, dropout=dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr, joint_types):
        # Get joint embeddings
        joint_embeds = self.joint_embedding(joint_types)
        
        # Concatenate with node features
        x = torch.cat([x, joint_embeds], dim=-1)
        
        # Pass through GAT layers
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index, edge_attr)
            if i < len(self.gat_layers) - 1:
                x = F.elu(x)
                x = self.dropout(x)
                
        return x


def create_multi_person_graph(keypoints_list: List[np.ndarray], 
                            depth_map: torch.Tensor,
                            distance_threshold: float = 150.0,
                            confidence_threshold: float = 0.3) -> Data:
    """
    Create a graph from multiple people's keypoints with depth information
    
    Args:
        keypoints_list: List of arrays [N_people, 17, 3] with (x, y, confidence)
        depth_map: Depth map tensor [H, W]
        distance_threshold: Maximum distance for edge connections (tau_r)
        confidence_threshold: Minimum confidence to include a keypoint
        
    Returns:
        PyTorch Geometric Data object with node features and edge attributes
    """
    all_nodes = []
    all_joint_types = []
    person_ids = []
    
    # Collect all nodes from all people
    for person_id, person_kps in enumerate(keypoints_list):
        for joint_idx, (x, y, conf) in enumerate(person_kps):
            if conf > confidence_threshold:
                # Get depth value
                x_int = int(np.clip(np.round(x), 0, depth_map.shape[1] - 1))
                y_int = int(np.clip(np.round(y), 0, depth_map.shape[0] - 1))
                depth = depth_map[y_int, x_int].item()
                
                # Node features: [x, y, depth, confidence]
                node_features = [x, y, depth, conf]
                all_nodes.append(node_features)
                all_joint_types.append(joint_idx)
                person_ids.append(person_id)
    
    if len(all_nodes) == 0:
        # Return empty graph
        return Data(x=torch.zeros((0, 4)), 
                   edge_index=torch.zeros((2, 0), dtype=torch.long),
                   edge_attr=torch.zeros((0, 4)),
                   joint_types=torch.zeros(0, dtype=torch.long))
    
    # Convert to tensors
    node_features = torch.tensor(all_nodes, dtype=torch.float32)
    joint_types = torch.tensor(all_joint_types, dtype=torch.long)
    
    # Build edges based on distance threshold
    edges = []
    edge_features = []
    
    num_nodes = len(all_nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Calculate Euclidean distance
            xi, yi, di, ci = node_features[i]
            xj, yj, dj, cj = node_features[j]
            
            distance = torch.sqrt((xi - xj)**2 + (yi - yj)**2)
            
            if distance <= distance_threshold:
                # Add bidirectional edges
                edges.append([i, j])
                edges.append([j, i])
                
                # Calculate edge affinity features
                depth_diff = torch.abs(di - dj)
                
                # Calculate cosine of angle (using normalized direction vector)
                if distance > 0:
                    dx = xj - xi
                    dy = yj - yi
                    # Normalized direction gives us cos and sin of angle
                    cos_theta = dx / distance
                else:
                    cos_theta = torch.tensor(1.0)
                
                conf_product = ci * cj
                
                # Edge features: [distance, depth_diff, cos_theta, conf_product]
                edge_feat = [distance.item(), depth_diff.item(), 
                           cos_theta.item(), conf_product.item()]
                
                # Add features for both directions
                edge_features.append(edge_feat)
                edge_features.append(edge_feat)
    
    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 4), dtype=torch.float32)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    
    # Create graph data
    data = Data(x=node_features, 
                edge_index=edge_index,
                edge_attr=edge_attr,
                joint_types=joint_types,
                person_ids=torch.tensor(person_ids, dtype=torch.long))
    
    return data


def prepare_depth_aware_dataset(dataloader, num_batches=5, distance_threshold=150.0):
    """
    Prepare dataset of graphs for training depth-aware GAT
    
    Args:
        dataloader: COCO dataloader
        num_batches: Number of batches to process
        distance_threshold: Maximum distance for edge connections
        
    Returns:
        dataset: List of PyTorch Geometric Data objects
    """
    from depth_estimation import estimate_depth
    from pose_rcnn import detect_keypoints
    
    dataset = []
    batch_count = 0
    
    for batch in dataloader:
        images = batch['image']
        img_ids = batch['img_id']
        
        print(f"Processing batch {batch_count + 1}/{num_batches}")
        
        for img, img_id in zip(images, img_ids):
            # Get depth map
            depth_map = estimate_depth(img)
            
            # Detect keypoints
            boxes, keypoints, scores = detect_keypoints(img)
            
            if len(keypoints) > 0:
                # Create graph
                graph_data = create_multi_person_graph(
                    keypoints, depth_map, distance_threshold
                )
                
                # Only add graphs with nodes
                if graph_data.x.size(0) > 0:
                    dataset.append(graph_data)
                    print(f"  Image {img_id}: {len(keypoints)} people, "
                          f"{graph_data.x.size(0)} nodes, "
                          f"{graph_data.edge_index.size(1)} edges")
        
        batch_count += 1
        if batch_count >= num_batches:
            break
    
    return dataset


def train_depth_aware_gat(dataset, epochs=100, batch_size=4, lr=0.001):
    """
    Train the depth-aware GAT model
    
    Args:
        dataset: List of graph Data objects
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        
    Returns:
        model: Trained model
    """
    from torch_geometric.loader import DataLoader
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = DepthAwarePoseGAT(
        node_features=4,
        joint_embedding_dim=16,
        hidden_dim=64,
        output_dim=128,
        num_heads=4,
        num_layers=3
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # For now, we'll use a simple reconstruction loss
    # In practice, you'd want a task-specific loss (e.g., for tracking)
    def loss_fn(output, target, mask=None):
        # Reconstruction loss on node features
        mse = F.mse_loss(output[:, :4], target, reduction='mean')
        return mse
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch.x, batch.edge_index, 
                         batch.edge_attr, batch.joint_types)
            
            # Calculate loss (reconstruction for now)
            loss = loss_fn(output, batch.x)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch + 1:03d}, Loss: {avg_loss:.4f}')
    
    return model