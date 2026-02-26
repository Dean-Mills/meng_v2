"""
Basic training loop with visualization for understanding GAT
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os
from .gat import DepthAwareGAT 
from .preprocess import KeypointPreprocessor

class BasicGATTrainer:
    """Simple trainer with visualization for learning"""
    
    def __init__(self, model: DepthAwareGAT,preprocessor: KeypointPreprocessor,output_dir: str = "outputs/basic"):
        self.preprocessor = preprocessor
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # History tracking
        self.history = {
            'loss': [],
            'embeddings': [],
            'reconstructions': []
        }
        
    def train(self, dataset: List, epochs: int = 100, batch_size: int = 8, lr: float = 0.001):
        """
        Simple training loop
        
        Args:
            dataset: List of Data objects
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
        """
        # Create dataloader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        all_trainable_params = list(self.model.parameters()) + list(self.preprocessor.joint_embeddings.parameters())
        optimizer = torch.optim.Adam(all_trainable_params, lr=lr)
        
        print(f"Training on {len(dataset)} graphs for {epochs} epochs")
        print(f"Device: {self.device}")
        print("-" * 50)
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            self.model.train()
            for batch in loader:
                batch = batch.to(self.device)
                
                # Forward pass
                embeddings, reconstructed = self.model(batch.x, batch.edge_index, batch.batch)
                
                # Loss: reconstruction error
                loss = F.mse_loss(reconstructed, batch.x)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Record history
            avg_loss = epoch_loss / num_batches
            self.history['loss'].append(avg_loss)
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1:03d}/{epochs} - Loss: {avg_loss:.6f}")
            
            # Visualize periodically
            if (epoch + 1) % 100 == 0:
                self._visualize_progress(epoch + 1, loader)
        
        # Final visualization
        self._create_final_plots()
        
        return self.model
    
    def _visualize_progress(self, epoch: int, loader: DataLoader):
        """Create visualizations during training"""
        print(f"\nCreating visualizations at epoch {epoch}...")
        
        self.model.eval()
        with torch.no_grad():
            # Get one batch for visualization
            for batch in loader:
                batch = batch.to(self.device)
                embeddings, reconstructed = self.model(batch.x, batch.edge_index, batch.batch)
                
                # Plot reconstruction quality
                self._plot_reconstruction(
                    batch.x.cpu(), 
                    reconstructed.cpu(),
                    epoch,
                    batch.num_nodes
                )
                
                # Visualize embeddings (using PCA for 2D projection)
                self._plot_embeddings(
                    embeddings.cpu(),
                    batch.batch.cpu() if batch.batch is not None else None,
                    epoch
                )
                
                break  # Only visualize first batch
    
    def _plot_reconstruction(self, original: torch.Tensor, reconstructed: torch.Tensor, 
                           epoch: int, num_nodes: int):
        """Plot original vs reconstructed features"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        feature_names = ['X Position', 'Y Position', 'Depth', 'Confidence']
        
        # Limit to first 100 nodes for clarity
        max_nodes = min(100, original.shape[0])
        
        for i, (ax, name) in enumerate(zip(axes.flat, feature_names)):
            # Scatter plot
            ax.scatter(original[:max_nodes, i], reconstructed[:max_nodes, i], 
                      alpha=0.6, s=30)
            
            # Perfect reconstruction line
            min_val = min(original[:, i].min(), reconstructed[:, i].min())
            max_val = max(original[:, i].max(), reconstructed[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            # Labels
            ax.set_xlabel(f'Original {name}')
            ax.set_ylabel(f'Reconstructed {name}')
            ax.set_title(f'{name} Reconstruction (Epoch {epoch})')
            
            # Calculate R²
            mse = F.mse_loss(original[:, i], reconstructed[:, i])
            ax.text(0.05, 0.95, f'MSE: {mse:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/reconstruction_epoch_{epoch}.png", dpi=150)
        plt.close()
    
    def _plot_embeddings(self, embeddings: torch.Tensor, batch_assignment: torch.Tensor, 
                        epoch: int):
        """Visualize learned embeddings using PCA"""
        from sklearn.decomposition import PCA
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings.numpy())
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        if batch_assignment is not None:
            # Color by graph/person
            scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=batch_assignment, cmap='tab10', 
                                s=50, alpha=0.7)
            plt.colorbar(scatter, label='Person/Graph ID')
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                       s=50, alpha=0.7)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'Node Embeddings Visualization (Epoch {epoch})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/embeddings_epoch_{epoch}.png", dpi=150)
        plt.close()
    
    def _create_final_plots(self):
        """Create final summary plots"""
        # Loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['loss'], linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True, alpha=0.3)
        
        # Add smoothed line
        if len(self.history['loss']) > 10:
            window = 10
            smoothed = np.convolve(self.history['loss'], 
                                 np.ones(window)/window, mode='valid')
            plt.plot(range(window//2, len(self.history['loss'])-window//2+1), 
                    smoothed, 'r-', linewidth=2, alpha=0.7, label='Smoothed')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_loss.png", dpi=150)
        plt.close()
        
        print(f"\nTraining complete! Visualizations saved to {self.output_dir}/")


def analyze_dataset(dataset: List):
    """Analyze and visualize dataset statistics"""
    stats = {
        'num_nodes': [],
        'num_edges': [],
        'avg_x': [],
        'avg_y': [],
        'avg_depth': [],
        'avg_conf': []
    }
    
    for graph in dataset:
        if graph.x.shape[0] > 0:
            stats['num_nodes'].append(graph.x.shape[0])
            stats['num_edges'].append(graph.edge_index.shape[1])
            stats['avg_x'].append(graph.x[:, 0].mean().item())
            stats['avg_y'].append(graph.x[:, 1].mean().item())
            stats['avg_depth'].append(graph.x[:, 2].mean().item())
            stats['avg_conf'].append(graph.x[:, 3].mean().item())
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (key, values) in enumerate(stats.items()):
        ax = axes[i]
        ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel(key.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {key.replace("_", " ").title()}')
        
        # Add mean line
        mean_val = np.mean(values)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig("outputs/basic/dataset_statistics.png", dpi=150)
    plt.close()
    
    print("\nDataset Statistics:")
    print("-" * 40)
    for key, values in stats.items():
        print(f"{key.replace('_', ' ').title()}:")
        print(f"  Mean: {np.mean(values):.2f}")
        print(f"  Std:  {np.std(values):.2f}")
        print(f"  Min:  {np.min(values):.2f}")
        print(f"  Max:  {np.max(values):.2f}\n")