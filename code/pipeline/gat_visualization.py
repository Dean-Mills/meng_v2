import torch
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data
from typing import List, Dict, Tuple


def visualize_attention_weights(model, graph_data, layer_idx=0):
    """
    Visualize attention weights from the GAT model
    
    Args:
        model: Trained DepthAwarePoseGAT model
        graph_data: PyTorch Geometric Data object
        layer_idx: Which GAT layer to visualize
    """
    model.eval()
    
    # Forward pass with hooks to capture attention
    attention_weights = []
    
    def hook_fn(module, input, output):
        # Capture attention weights
        attention_weights.append(output)
    
    # Register hook on the specified layer
    hook = model.gat_layers[layer_idx].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        _ = model(graph_data.x, graph_data.edge_index, 
                 graph_data.edge_attr, graph_data.joint_types)
    
    hook.remove()
    
    # TODO: Implement attention weight extraction and visualization
    # This would require modifying the GAT layer to expose attention weights
    print(f"Attention visualization for layer {layer_idx} - Implementation needed")


def analyze_edge_features(dataset: List[Data]):
    """
    Analyze the distribution of edge features across the dataset
    
    Args:
        dataset: List of graph Data objects
    """
    all_edge_features = []
    
    for graph in dataset:
        if graph.edge_attr is not None and graph.edge_attr.size(0) > 0:
            all_edge_features.append(graph.edge_attr.numpy())
    
    if len(all_edge_features) == 0:
        print("No edge features found in dataset")
        return
    
    edge_features = np.concatenate(all_edge_features, axis=0)
    feature_names = ['Distance', 'Depth Diff', 'Cos Theta', 'Conf Product']
    
    # Create subplots for each feature
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        ax.hist(edge_features[:, i], bins=50, alpha=0.7, color=f'C{i}')
        ax.set_xlabel(name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {name}')
        
        # Add statistics
        mean_val = np.mean(edge_features[:, i])
        std_val = np.std(edge_features[:, i])
        ax.axvline(mean_val, color='red', linestyle='--', 
                  label=f'Mean: {mean_val:.2f}')
        ax.text(0.7, 0.9, f'Std: {std_val:.2f}', 
               transform=ax.transAxes)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{settings.output_dir}/edge_feature_distributions.png")
    plt.show()
    plt.close()


def visualize_pose_connections(graph_data: Data, image=None):
    """
    Visualize the pose connections overlaid on the original image
    
    Args:
        graph_data: Graph data with node positions and edges
        image: Optional original image tensor
    """
    plt.figure(figsize=(12, 10))
    
    if image is not None:
        img_np = image.permute(1, 2, 0).cpu().numpy()
        plt.imshow(img_np)
    
    # Extract node positions
    positions = graph_data.x[:, :2].numpy()  # x, y coordinates
    depths = graph_data.x[:, 2].numpy()
    confidences = graph_data.x[:, 3].numpy()
    
    # Color by person ID if available
    if hasattr(graph_data, 'person_ids'):
        colors = plt.cm.tab10(graph_data.person_ids.numpy())
    else:
        # Color by depth
        colors = plt.cm.viridis(depths / depths.max())
    
    # Plot nodes
    scatter = plt.scatter(positions[:, 0], positions[:, 1], 
                         c=colors, s=100*confidences, 
                         alpha=0.8, edgecolors='black', linewidth=2)
    
    # Plot edges
    edge_index = graph_data.edge_index.t().numpy()
    for i, j in edge_index:
        if i < j:  # Avoid duplicate edges
            x_coords = [positions[i, 0], positions[j, 0]]
            y_coords = [positions[i, 1], positions[j, 1]]
            
            # Edge properties
            if graph_data.edge_attr is not None:
                distance = graph_data.edge_attr[edge_index.tolist().index([i, j]), 0]
                alpha = max(0.1, 1.0 - distance / 200.0)  # Fade with distance
            else:
                alpha = 0.5
            
            plt.plot(x_coords, y_coords, 'gray', alpha=alpha, linewidth=1)
    
    plt.title("Pose Graph Connections")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_training_metrics(losses: List[float], save_path: str = None):
    """
    Plot training loss over epochs
    
    Args:
        losses: List of loss values per epoch
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    
    # Add smoothed line
    if len(losses) > 10:
        window = min(10, len(losses) // 10)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window//2, len(losses)-window//2+1), smoothed, 
                'r-', linewidth=2, alpha=0.7, label='Smoothed')
        plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()


def compare_node_features(original_features: torch.Tensor, 
                         predicted_features: torch.Tensor,
                         feature_names: List[str] = None):
    """
    Compare original and predicted node features
    
    Args:
        original_features: Original node features [N, D]
        predicted_features: Predicted node features [N, D']
        feature_names: Names of features for labeling
    """
    if feature_names is None:
        feature_names = ['X', 'Y', 'Depth', 'Confidence']
    
    # Take first 4 dimensions if predicted has more
    if predicted_features.shape[1] > 4:
        predicted_features = predicted_features[:, :4]
    
    num_features = min(original_features.shape[1], predicted_features.shape[1])
    
    fig, axes = plt.subplots(1, num_features, figsize=(15, 4))
    if num_features == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, feature_names[:num_features])):
        ax.scatter(original_features[:, i].cpu(), 
                  predicted_features[:, i].cpu(), 
                  alpha=0.5, s=20)
        
        # Add diagonal line
        min_val = min(original_features[:, i].min(), 
                     predicted_features[:, i].min())
        max_val = max(original_features[:, i].max(), 
                     predicted_features[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', alpha=0.7)
        
        ax.set_xlabel(f'Original {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name} Comparison')
        
        # Calculate R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(original_features[:, i].cpu(), 
                      predicted_features[:, i].cpu())
        ax.text(0.05, 0.95, f'R²: {r2:.3f}', 
               transform=ax.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    plt.close()


def analyze_graph_statistics(dataset: List[Data]):
    """
    Analyze and visualize dataset statistics
    
    Args:
        dataset: List of graph Data objects
    """
    stats = {
        'num_nodes': [],
        'num_edges': [],
        'avg_degree': [],
        'edge_density': [],
        'num_people': []
    }
    
    for graph in dataset:
        num_nodes = graph.x.size(0)
        num_edges = graph.edge_index.size(1) // 2  # Undirected graph
        
        stats['num_nodes'].append(num_nodes)
        stats['num_edges'].append(num_edges)
        
        if num_nodes > 0:
            stats['avg_degree'].append(2 * num_edges / num_nodes)
            max_edges = num_nodes * (num_nodes - 1) / 2
            stats['edge_density'].append(num_edges / max_edges if max_edges > 0 else 0)
            
            if hasattr(graph, 'person_ids'):
                num_people = len(torch.unique(graph.person_ids))
                stats['num_people'].append(num_people)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (key, values) in enumerate(stats.items()):
        if i < len(axes) and len(values) > 0:
            ax = axes[i]
            ax.hist(values, bins=20, alpha=0.7, color=f'C{i}')
            ax.set_xlabel(key.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {key.replace("_", " ").title()}')
            
            # Add statistics
            mean_val = np.mean(values)
            ax.axvline(mean_val, color='red', linestyle='--', 
                      label=f'Mean: {mean_val:.2f}')
            ax.legend()
    
    # Hide unused subplots
    for i in range(len(stats), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{settings.output_dir}/dataset_statistics.png")
    plt.show()
    plt.close()
    
    # Print summary statistics
    print("\nDataset Statistics Summary:")
    print("-" * 40)
    for key, values in stats.items():
        if len(values) > 0:
            print(f"{key.replace('_', ' ').title()}:")
            print(f"  Mean: {np.mean(values):.2f}")
            print(f"  Std:  {np.std(values):.2f}")
            print(f"  Min:  {np.min(values):.2f}")
            print(f"  Max:  {np.max(values):.2f}")
            print()


# Import settings for output directory
from settings import settings