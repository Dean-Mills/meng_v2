import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GATConfig:
    # Input features
    num_joint_types: int = 17
    joint_embedding_dim: int = 32
    raw_feature_dim: int = 4  # x, y, depth, confidence
    
    # Architecture
    hidden_dim: int = 64
    output_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    
    # Options
    use_layer_norm: bool = True
    l2_normalize: bool = True
    
    @property
    def input_dim(self):
        return self.raw_feature_dim + self.joint_embedding_dim


class GATEmbedding(nn.Module):
    """
    Dynamic GAT-based joint embedding network.
    
    Architecture pattern:
        - Layers 1 to N-1: concat mode (output = hidden_dim * num_heads)
        - Layer N: average mode (output = hidden_dim)
        - Final projection: hidden_dim -> output_dim
    """
    
    def __init__(self, config: GATConfig = None):
        super().__init__()
        self.config = config or GATConfig()
        c = self.config
        
        # Joint type embedding
        self.joint_embedding = nn.Embedding(c.num_joint_types, c.joint_embedding_dim)
        
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(c.num_layers):
            if i == 0:
                in_dim = c.input_dim  # 36
            else:
                in_dim = c.hidden_dim * c.num_heads  # Previous concat output
            
            # Last layer uses average mode, others use concat
            is_last = (i == c.num_layers - 1)
            concat = not is_last
            
            # Output dimension after this layer
            # concat=True: hidden_dim * num_heads
            # concat=False: hidden_dim
            
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=c.hidden_dim,
                    heads=c.num_heads,
                    dropout=c.dropout,
                    concat=concat
                )
            )
            
            # Optional: intermediate layer norms
            out_dim = c.hidden_dim * c.num_heads if concat else c.hidden_dim
            self.norms.append(nn.LayerNorm(out_dim) if c.use_layer_norm else nn.Identity())
        
        # Final projection
        self.projection = nn.Linear(c.hidden_dim, c.output_dim)
        self.final_norm = nn.LayerNorm(c.output_dim) if c.use_layer_norm else nn.Identity()
        
        self.act = nn.ELU()
        self.dropout = nn.Dropout(c.dropout)
    
    def forward(self, data):
        x = data.x  # [N, 4]
        joint_types = data.joint_types  # [N]
        edge_index = data.edge_index  # [2, E]
        
        # Embed joint types and concatenate
        joint_emb = self.joint_embedding(joint_types)
        x = torch.cat([x, joint_emb], dim=-1)  # [N, 36]
        
        # Pass through GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            x = gat(x, edge_index)
            x = norm(x)
            x = self.act(x)
            
            # Dropout on all but last layer
            if i < len(self.gat_layers) - 1:
                x = self.dropout(x)
        
        # Project to output dimension
        emb = self.projection(x)
        emb = self.final_norm(emb)
        
        if self.config.l2_normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        
        return emb