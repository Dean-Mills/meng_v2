import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class DepthAwareGAT(nn.Module):
    """
    Depth-Aware Graph Attention Network for pose tracking via pairwise prediction
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=32, num_heads=4, dropout=0.6):
        super().__init__()
        
        # GAT layers for learning joint embeddings
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=dropout)
        
        # Pairwise classifier head
        self.pairwise_classifier = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),  # Concatenate two embeddings
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Output: same person probability
        )
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass - returns embeddings
        
        Args:
            x: Node features [num_nodes_in_batch, input_dim]
            edge_index: Graph connectivity [2, num_edges_in_batch] 
            batch: Assignment of nodes to graphs [num_nodes_in_batch]
            
        Returns:
            embeddings: Joint embeddings [num_nodes_in_batch, output_dim]
        """
        # GAT encoder
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        x = F.dropout(x, p=0.6, training=self.training)
        embeddings = self.conv2(x, edge_index) 
        
        return embeddings
    
    def predict_pairwise(self, embeddings):
        """
        Predict if pairs of joints belong to same person
        
        Args:
            embeddings: Joint embeddings [N, output_dim]
            
        Returns:
            predictions: Pairwise predictions [N*(N-1)/2] - same person probabilities
            pair_indices: Which pairs these predictions correspond to
        """
        n_nodes = embeddings.size(0)
        pairs = []
        pair_indices = []
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                pair_embedding = torch.cat([embeddings[i], embeddings[j]], dim=0)
                pairs.append(pair_embedding)
                pair_indices.append((i, j))
        
        if not pairs:
            return torch.empty(0, device=embeddings.device), []
        
        pair_features = torch.stack(pairs)
        predictions = self.pairwise_classifier(pair_features).squeeze()
        
        return predictions, pair_indices