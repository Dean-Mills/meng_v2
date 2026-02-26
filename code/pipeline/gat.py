import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader

class PoseGAT(nn.Module):
    """
    Graph Attention Network for human pose analysis
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=4):
        super(PoseGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.conv3 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        x = self.conv3(x, edge_index)
        
        return x

def create_pose_graph(node_features):
    """
    Create a graph from pose keypoints
    
    Args:
        node_features: List of arrays with shape [17, 4] for each person
                      Each keypoint has features (x, y, depth, confidence)
        
    Returns:
        graph_data: PyTorch Geometric Data object
    """
    skeleton_edges = [
        (0, 1), (0, 2),  # nose to eyes
        (1, 3), (2, 4),  # eyes to ears
        (5, 7), (7, 9),  # left shoulder to left elbow to left wrist
        (6, 8), (8, 10),  # right shoulder to right elbow to right wrist
        (5, 6), (5, 11), (6, 12),  # shoulders to hips
        (11, 13), (13, 15),  # left hip to left knee to left ankle
        (12, 14), (14, 16)  # right hip to right knee to right ankle
    ]
    
    edges = []
    for edge in skeleton_edges:
        edges.extend([edge, (edge[1], edge[0])])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    
    return data

def prepare_dataset(all_node_features):
    """
    Prepare a dataset of pose graphs for GAT training
    
    Args:
        all_node_features: List of node features for each person
        
    Returns:
        dataset: List of PyTorch Geometric Data objects
    """
    dataset = []
    
    for node_features in all_node_features:
        # Create graph data
        graph_data = create_pose_graph(node_features)
        dataset.append(graph_data)
    
    return dataset

def train_gat(dataset, epochs=100):
    """
    Train a GAT model on pose graphs
    
    Args:
        dataset: List of PyTorch Geometric Data objects
        epochs: Number of training epochs
        
    Returns:
        model: Trained GAT model
    """
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model
    in_channels = 4  # x, y, depth, confidence
    hidden_channels = 32
    out_channels = 4  # Match the input features dimension
    model = PoseGAT(in_channels, hidden_channels, out_channels)
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data.x, data.edge_index)
            
            # Compare output with the original features
            loss = F.mse_loss(output, data.x)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {total_loss/len(dataloader):.4f}')
    
    return model