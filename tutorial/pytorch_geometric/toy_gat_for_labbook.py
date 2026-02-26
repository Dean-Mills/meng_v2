import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class SimpleGAT(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleGAT, self).__init__()
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
    def forward(self, features, adj):
        h = self.W(features)
        N = h.size(0)
        
        h_i = h.unsqueeze(1).repeat(1, N, 1)  # [N, N, out_features]
        h_j = h.unsqueeze(0).repeat(N, 1, 1)  # [N, N, out_features]
        
        pairs = torch.cat([h_i, h_j], dim=2)  # [N, N, 2*out_features]
        
        e = F.leaky_relu(self.a(pairs).squeeze(2))  # [N, N]
        
        mask = -1e9 * (1.0 - adj)
        masked_e = e + mask
        
        attention = F.softmax(masked_e, dim=1)
        
        output = torch.matmul(attention, h)
        
        return output, attention

def create_toy_example():
    """
    Create a simple 3-node graph representing:
    Node 0 -- Node 1 -- Node 2
    Like: Shoulder -- Elbow -- Wrist
    """
    # Node features: [x, y, depth] (simplified)
    features = torch.tensor([
        [10.0, 20.0, 0.5],  # Node 0 (shoulder)
        [15.0, 25.0, 0.6],  # Node 1 (elbow)
        [20.0, 30.0, 0.7],  # Node 2 (wrist)
    ], dtype=torch.float32)
    
    # Adjacency matrix (who connects to whom)
    adj = torch.tensor([
        [0, 1, 0],  # Node 0 connects to Node 1
        [1, 0, 1],  # Node 1 connects to Node 0 and 2
        [0, 1, 0],  # Node 2 connects to Node 1
    ], dtype=torch.float32)
    
    return features, adj

def visualize_attention(features, adj, attention):
    """Visualize the graph and attention weights"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Original graph structure
    ax1.set_title("Graph Structure")
    ax1.imshow(adj.numpy(), cmap='Blues', vmin=0, vmax=1)
    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([0, 1, 2])
    ax1.set_xticklabels(['Node 0', 'Node 1', 'Node 2'])
    ax1.set_yticklabels(['Node 0', 'Node 1', 'Node 2'])
    
    # Add values
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f'{adj[i,j]:.0f}', ha='center', va='center')
    
    # Plot 2: Attention weights
    ax2.set_title("Learned Attention Weights")
    im = ax2.imshow(attention.detach().numpy(), cmap='Reds', vmin=0, vmax=1)
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(['Node 0', 'Node 1', 'Node 2'])
    ax2.set_yticklabels(['Node 0', 'Node 1', 'Node 2'])
    
    # Add values
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f'{attention[i,j]:.2f}', ha='center', va='center')
    
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.show()

def main():
    print("=== Simple GAT Toy Example ===\n")
    
    features, adj = create_toy_example()
    
    print("Input Features (3 nodes):")
    print("Node 0 (shoulder):", features[0].numpy())
    print("Node 1 (elbow):", features[1].numpy())
    print("Node 2 (wrist):", features[2].numpy())
    
    print("\nAdjacency Matrix:")
    print(adj.numpy())
    print("(1 = connected, 0 = not connected)")
    
    model = SimpleGAT(in_features=3, out_features=2)
    
    output, attention = model(features, adj)
    
    print("\n=== After GAT Processing ===")
    print("\nOutput Features (transformed):")
    for i in range(3):
        print(f"Node {i}:", output[i].detach().numpy())
    
    print("\nAttention Weights (who pays attention to whom):")
    for i in range(3):
        print(f"Node {i} attention:", attention[i].detach().numpy())
    
    visualize_attention(features, adj, attention)
    
    print("\n=== Detailed look at Node 1 (elbow) ===")
    print("Node 1 can see: Node 0 (shoulder) and Node 2 (wrist)")
    print(f"Attention to Node 0: {attention[1, 0]:.3f}")
    print(f"Attention to Node 2: {attention[1, 2]:.3f}")
    print("(These sum to 1.0)")
    
    weighted_sum = (attention[1, 0] * model.W(features[0]) + 
                   attention[1, 2] * model.W(features[2]))
    print(f"\nNode 1's output is weighted sum of its neighbors' transformed features")
    print(weighted_sum)

if __name__ == "__main__":
    main()