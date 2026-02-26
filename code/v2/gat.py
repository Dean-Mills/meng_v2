import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from settings import settings

class DepthAwareGAT(nn.Module):
    """
    Simple GAT that outputs embeddings for keypoint clustering
    """
    
    def __init__(self):
        super().__init__()
        
        self.input_dim = settings.input_dim # 5: [x, y, depth, confidence, joint_type]
        self.hidden_dim = settings.hidden_dim
        self.output_dim = settings.output_dim
        self.num_heads = settings.num_heads
        self.dropout = settings.dropout
        
        self.conv1 = GATConv(
            in_channels=self.input_dim, 
            out_channels=self.hidden_dim,
            heads=self.num_heads,
            dropout=self.dropout,
            concat=True  # Concatenate attention heads
        )
        
        self.conv2 = GATConv(
            in_channels=self.hidden_dim * self.num_heads,  # Because concat=True above
            out_channels=self.output_dim,
            heads=1,  # Single head for final layer
            dropout=self.dropout,
            concat=False
        )
        
    def forward(self, x, edge_index):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim] 
            edge_index: Graph edges [2, num_edges]
            
        Returns:
            embeddings: Node embeddings [num_nodes, output_dim]
        """
        
        # First GAT layer
        x = self.conv1(x, edge_index)  # [num_nodes, hidden_dim * num_heads]
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GAT layer  
        x = self.conv2(x, edge_index)  # [num_nodes, output_dim]
        
        # No activation on final embeddings
        return x
    
    def get_embeddings(self, x, edge_index):
        """
        Convenience method - same as forward but more explicit name
        """
        return self.forward(x, edge_index)