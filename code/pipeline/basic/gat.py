
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class DepthAwareGAT(nn.Module):
    """
    Depth-Aware Graph Attention Network for pose estimation
    This version is compatible with PyTorch Geometric.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=32, num_heads=4, dropout=0.6):
        super().__init__()
        
        # We will use GATConv which handles multi-head internally
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        
        # The output of the first layer is hidden_dim * num_heads
        # The second layer will be an "encoder" to get the final embeddings
        self.conv2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=dropout)

        # A decoder to reconstruct the original features from the embeddings
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim) # Reconstruct back to original input_dim
        )
        
    def forward(self, x, edge_index, batch):
        """
        Forward pass compatible with PyG's DataLoader batch object.
        
        Args:
            x: Node features [num_nodes_in_batch, input_dim]
            edge_index: Graph connectivity [2, num_edges_in_batch]
            batch: Assignment of nodes to graphs [num_nodes_in_batch]
            
        Returns:
            embeddings: Latent node embeddings [num_nodes_in_batch, output_dim]
            reconstructed: Reconstructed node features [num_nodes_in_batch, input_dim]
        """
        # --- Encoder ---
        # Layer 1
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        # Layer 2
        x = F.dropout(x, p=0.6, training=self.training)
        # The output of conv2 is our final node-level embeddings
        embeddings = self.conv2(x, edge_index) 

        # --- Decoder ---
        # Use the learned embeddings to reconstruct original features
        reconstructed = self.decoder(embeddings)
        
        return embeddings, reconstructed

    def compute_graph_embedding(self, x, edge_index, batch):
        """
        Compute a single embedding for the entire graph batch.
        """
        # Get node embeddings from the forward pass
        node_embeddings, _ = self.forward(x, edge_index, batch)
        
        # Use PyG's global_mean_pool to average nodes for each graph in the batch
        graph_embedding = global_mean_pool(node_embeddings, batch)
        
        return graph_embedding

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class GraphAttentionLayer(nn.Module):
#     """
#     Simple Graph Attention Layer
#     """
#     def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
#         super(GraphAttentionLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.dropout = dropout
#         self.alpha = alpha
        
#         # Linear transformation for node features
#         self.W = nn.Linear(in_features, out_features, bias=False)
        
#         # Attention mechanism parameters
#         self.a = nn.Linear(2 * out_features, 1, bias=False)
        
#         # LeakyReLU for attention
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
        
#         self.reset_parameters()
    
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.W.weight)
#         nn.init.xavier_uniform_(self.a.weight)
    
#     def forward(self, features, adj):
#         """
#         Args:
#             features: Node features [N, in_features]
#             adj: Adjacency matrix [N, N]
#         Returns:
#             Updated node features [N, out_features]
#         """
#         N = features.size(0)
        
#         # Apply linear transformation
#         h = self.W(features)  # [N, out_features]
        
#         # Self-attention on nodes
#         a_input = self._prepare_attention_input(h)  # [N, N, 2*out_features]
#         e = self.leakyrelu(self.a(a_input).squeeze(2))  # [N, N]
        
#         # Mask attention scores using adjacency matrix
#         zero_vec = -9e15 * torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        
#         # Apply softmax
#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
        
#         # Apply attention to features
#         h_prime = torch.matmul(attention, h)  # [N, out_features]
        
#         return h_prime
    
#     def _prepare_attention_input(self, h):
#         """
#         Prepare input for attention mechanism
#         """
#         N = h.size(0)
        
#         # Repeat h1 and h2 to create all pairs
#         h_repeat = h.repeat(N, 1).view(N, N, -1)  # [N, N, out_features]
#         h_repeat_interleave = h.repeat_interleave(N, dim=0).view(N, N, -1)  # [N, N, out_features]
        
#         # Concatenate them
#         a_input = torch.cat([h_repeat_interleave, h_repeat], dim=-1)  # [N, N, 2*out_features]
        
#         return a_input


# class DepthAwareGAT(nn.Module):
#     """
#     Depth-Aware Graph Attention Network for pose estimation
#     """
#     def __init__(self, input_dim, hidden_dim=64, output_dim=32, num_heads=4, dropout=0.6):
#         super(DepthAwareGAT, self).__init__()
        
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.num_heads = num_heads
#         self.dropout = dropout
        
#         # Multi-head attention layers
#         self.attention_heads = nn.ModuleList([
#             GraphAttentionLayer(input_dim, hidden_dim, dropout=dropout)
#             for _ in range(num_heads)
#         ])
        
#         # Output layer
#         self.out_attention = GraphAttentionLayer(
#             hidden_dim * num_heads, 
#             output_dim, 
#             dropout=dropout
#         )
        
#         # Additional layers for task-specific outputs
#         self.mlp = nn.Sequential(
#             nn.Linear(output_dim, output_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(output_dim, output_dim)
#         )
        
#     def forward(self, node_features, adj_matrix):
#         """
#         Forward pass through the GAT
        
#         Args:
#             node_features: Node features [N, input_dim]
#             adj_matrix: Adjacency matrix [N, N]
            
#         Returns:
#             output: Processed node features [N, output_dim]
#         """
#         # Apply dropout to input
#         x = F.dropout(node_features, self.dropout, training=self.training)
        
#         # Multi-head attention
#         head_outputs = []
#         for attention_head in self.attention_heads:
#             head_output = attention_head(x, adj_matrix)
#             head_outputs.append(head_output)
        
#         # Concatenate all heads
#         x = torch.cat(head_outputs, dim=1)  # [N, hidden_dim * num_heads]
#         x = F.elu(x)
#         x = F.dropout(x, self.dropout, training=self.training)
        
#         # Final attention layer
#         x = self.out_attention(x, adj_matrix)  # [N, output_dim]
#         x = F.elu(x)
        
#         # Task-specific processing
#         output = self.mlp(x)
        
#         return output
    
#     def compute_graph_embedding(self, node_features, adj_matrix):
#         """
#         Compute a single embedding for the entire graph
        
#         Args:
#             node_features: Node features [N, input_dim]
#             adj_matrix: Adjacency matrix [N, N]
            
#         Returns:
#             graph_embedding: Graph-level embedding [output_dim]
#         """
#         # Get node embeddings
#         node_embeddings = self.forward(node_features, adj_matrix)
        
#         # Simple mean pooling
#         graph_embedding = torch.mean(node_embeddings, dim=0)
        
#         return graph_embedding