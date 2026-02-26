import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from settings import settings

class TinyGAT(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = settings.input_dim
        hidden_dim = settings.hidden_dim
        out_dim = settings.output_dim
        heads = settings.num_heads
        dropout = settings.dropout

        self.gat1 = GATv2Conv(in_channels=in_dim, out_channels=hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATv2Conv(in_channels=hidden_dim * heads, out_channels=hidden_dim, heads=1, dropout=dropout, concat=True)

        self.lin = nn.Linear(hidden_dim, out_dim)

        self.act = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, getattr(data, "edge_index", None)
        if edge_index is None:
            edge_index = x.new_zeros((2, 0), dtype=torch.long)

        x = self.gat1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index)
        x = self.act(x)

        emb = self.lin(x)
        return emb