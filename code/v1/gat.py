import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class DepthAwareGAT(nn.Module):
    """
    Simple GAT that outputs embeddings for keypoint clustering
    """
    
    def __init__(self, input_dim=5, hidden_dim=32, output_dim=16, num_heads=2, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim      # 5: [x, y, depth, confidence, joint_type]
        self.hidden_dim = hidden_dim    # 32: intermediate size
        self.output_dim = output_dim    # 16: final embedding size
        self.num_heads = num_heads      # 2: attention heads
        self.dropout = dropout          # 0.1: dropout rate
        
        self.conv1 = GATConv(
            in_channels=input_dim, 
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True  # Concatenate attention heads
        )
        
        self.conv2 = GATConv(
            in_channels=hidden_dim * num_heads,  # Because concat=True above
            out_channels=output_dim,
            heads=1,  # Single head for final layer
            dropout=dropout,
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