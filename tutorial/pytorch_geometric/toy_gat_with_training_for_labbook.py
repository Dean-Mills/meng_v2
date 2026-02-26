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
        
        h_i = h.unsqueeze(1).repeat(1, N, 1)
        h_j = h.unsqueeze(0).repeat(N, 1, 1)
        
        pairs = torch.cat([h_i, h_j], dim=2)
        e = F.leaky_relu(self.a(pairs).squeeze(2))
        
        mask = -1e9 * (1.0 - adj)
        masked_e = e + mask
        
        attention = F.softmax(masked_e, dim=1)
        output = torch.matmul(attention, h)
        
        return output, attention

def create_toy_data_with_targets():
    """
    Create toy data with made-up target outputs
    The network to learn to output 
    specific features based on the node position
    """
    # Input features: [x, y, depth]
    features = torch.tensor([
        [10.0, 20.0, 0.5],  # Shoulder
        [15.0, 25.0, 0.6],  # Elbow
        [20.0, 30.0, 0.7],  # Wrist
    ], dtype=torch.float32)
    
    # Adjacency matrix
    adj = torch.tensor([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=torch.float32)
    
    # Made-up target outputs: 
    # [forward_reach, upward_reach] for each joint
    targets = torch.tensor([
        [0.3, 0.8],  # Shoulder: low forward, high upward
        [0.6, 0.5],  # Elbow: medium both
        [0.9, 0.2],  # Wrist: high forward, low upward
    ], dtype=torch.float32)
    
    return features, adj, targets

def train_toy_gat(num_epochs=100):
    """
    Train the toy GAT model
    """
    # Create data
    features, adj, targets = create_toy_data_with_targets()
    
    # Initialize model
    model = SimpleGAT(in_features=3, out_features=2)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training history
    losses = []
    
    print("=== Training Toy GAT ===\n")
    print("Target outputs want to learn:")
    print("Shoulder:", targets[0].numpy(), "(low forward, high upward)")
    print("Elbow:", targets[1].numpy(), "(medium both)")
    print("Wrist:", targets[2].numpy(), "(high forward, low upward)")
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        # Forward pass
        output, attention = model(features, adj)
        
        # Compute loss
        loss = criterion(output, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record loss
        losses.append(loss.item())
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")
    
    print("\n=== Training Complete ===")
    
    # Final predictions
    model.eval()
    with torch.no_grad():
        final_output, final_attention = model(features, adj)
    
    print("\nFinal predictions vs targets:")
    for i, joint in enumerate(['Shoulder', 'Elbow', 'Wrist']):
        pred = final_output[i].numpy()
        targ = targets[i].numpy()
        print(f"{joint}: Predicted {pred}, Target {targ}")
    
    print("\nFinal attention weights:")
    print(final_attention.numpy())
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    
    # Plot final attention
    plt.subplot(1, 2, 2)
    plt.imshow(final_attention.detach().numpy(), cmap='Reds', vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Final Attention Weights')
    plt.xticks([0, 1, 2], ['Shoulder', 'Elbow', 'Wrist'])
    plt.yticks([0, 1, 2], ['Shoulder', 'Elbow', 'Wrist'])
    
    # Add values
    for i in range(3):
        for j in range(3):
            plt.text(j, i, f'{final_attention[i,j]:.2f}', 
                    ha='center', va='center')
    
    plt.tight_layout()
    plt.show()
    
    return model, losses

def analyze_what_gat_learned(model, features, adj):
    """
    Analyze what the GAT learned during training
    """
    print("\n=== Analyzing What GAT Learned ===\n")
    
    # Get the learned weights
    W_weights = model.W.weight.data.numpy()
    a_weights = model.a.weight.data.numpy()
    
    print("Learned transformation matrix W:")
    print(W_weights)
    print("\nThis transforms [x, y, depth] -> [feature1, feature2]")
    
    print("\nLearned attention weights a:")
    print(a_weights)
    print("\nThis computes importance of node pairs")
    
    # Show how each node's features are transformed
    with torch.no_grad():
        transformed = model.W(features)
        
    print("\nHow each node's features are transformed:")
    for i, joint in enumerate(['Shoulder', 'Elbow', 'Wrist']):
        orig = features[i].numpy()
        trans = transformed[i].numpy()
        print(f"{joint}: {orig} -> {trans}")

if __name__ == "__main__":
    # Train the model
    model, losses = train_toy_gat(num_epochs=100)
    
    # Analyze what was learned
    features, adj, _ = create_toy_data_with_targets()
    analyze_what_gat_learned(model, features, adj)