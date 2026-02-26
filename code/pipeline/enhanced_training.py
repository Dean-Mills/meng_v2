import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from depth_aware_gat import DepthAwarePoseGAT
from gat_visualization import (
    plot_training_metrics, 
    compare_node_features,
    analyze_edge_features,
    analyze_graph_statistics
)
from settings import settings


class DepthAwareGATTrainer:
    """Enhanced trainer for Depth-Aware GAT with visualization and monitoring"""
    
    def __init__(self, model: DepthAwarePoseGAT, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_stats': []
        }
        
    def train(self, 
              train_dataset: List,
              val_dataset: Optional[List] = None,
              epochs: int = 100,
              batch_size: int = 4,
              lr: float = 0.001,
              weight_decay: float = 0.0001,
              early_stopping_patience: int = 20,
              visualize_every: int = 25):
        """
        Train the model with enhanced monitoring
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: L2 regularization
            early_stopping_patience: Patience for early stopping
            visualize_every: Visualize results every N epochs
        """
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                   lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer)
            self.history['train_loss'].append(train_loss)
            
            # Validation phase
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint(epoch, val_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            else:
                scheduler.step(train_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                self._print_progress(epoch, train_loss, 
                                   val_loss if val_loader else None)
            
            # Visualize progress
            if (epoch + 1) % visualize_every == 0:
                self._visualize_progress(epoch, train_loader, val_loader)
        
        # Final visualization
        self._final_visualization(train_loader, val_loader)
        
        return self.model
    
    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(batch.x, batch.edge_index, 
                              batch.edge_attr, batch.joint_types)
            
            # Compute loss (reconstruction + regularization)
            loss = self._compute_loss(output, batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
        
        return total_loss / len(loader.dataset)
    
    def _validate_epoch(self, loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                
                output = self.model(batch.x, batch.edge_index, 
                                  batch.edge_attr, batch.joint_types)
                
                loss = self._compute_loss(output, batch)
                total_loss += loss.item() * batch.num_graphs
        
        return total_loss / len(loader.dataset)
    
    def _compute_loss(self, output: torch.Tensor, batch) -> torch.Tensor:
        """
        Compute the training loss
        
        For now, using reconstruction loss on node features.
        You can extend this for task-specific losses.
        """
        # Reconstruction loss on position and confidence
        recon_loss = F.mse_loss(output[:, :4], batch.x)
        
        # Optional: Add constraints or task-specific losses
        # e.g., temporal consistency for tracking
        
        return recon_loss
    
    def _print_progress(self, epoch: int, train_loss: float, 
                       val_loss: Optional[float] = None):
        """Print training progress"""
        if val_loss is not None:
            print(f'Epoch {epoch+1:03d}: '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1:03d}: Train Loss: {train_loss:.4f}')
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint_path = f"{settings.output_dir}/checkpoint_best.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }, checkpoint_path)
        print(f"Saved checkpoint at epoch {epoch} with val_loss {val_loss:.4f}")
    
    def _visualize_progress(self, epoch: int, train_loader: DataLoader, 
                           val_loader: Optional[DataLoader] = None):
        """Visualize training progress"""
        print(f"\nVisualizing progress at epoch {epoch+1}...")
        
        # Plot loss curves
        plot_training_metrics(
            self.history['train_loss'],
            save_path=f"{settings.output_dir}/training_loss_epoch_{epoch+1}.png"
        )
        
        # Sample predictions
        self._visualize_predictions(train_loader, prefix=f"train_epoch_{epoch+1}")
        
        if val_loader:
            self._visualize_predictions(val_loader, prefix=f"val_epoch_{epoch+1}")
    
    def _visualize_predictions(self, loader: DataLoader, prefix: str = ""):
        """Visualize model predictions on a sample batch"""
        self.model.eval()
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                
                # Get predictions
                output = self.model(batch.x, batch.edge_index,
                                  batch.edge_attr, batch.joint_types)
                
                # Compare features
                compare_node_features(
                    batch.x[:, :4], 
                    output[:, :4],
                    feature_names=['X', 'Y', 'Depth', 'Confidence']
                )
                
                plt.savefig(f"{settings.output_dir}/{prefix}_predictions.png")
                plt.close()
                
                break  # Only visualize first batch
    
    def _final_visualization(self, train_loader: DataLoader, 
                           val_loader: Optional[DataLoader] = None):
        """Create final visualization and analysis"""
        print("\nCreating final visualizations...")
        
        # Plot final loss curves
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.history['train_loss'], label='Train Loss')
        if val_loader:
            ax.plot(self.history['val_loss'], label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(f"{settings.output_dir}/final_training_history.png")
        plt.close()
        
        # Analyze dataset statistics
        all_data = list(train_loader.dataset)
        if val_loader:
            all_data.extend(list(val_loader.dataset))
        
        analyze_graph_statistics(all_data)
        analyze_edge_features(all_data)


def enhanced_train_depth_aware_gat(dataset: List,
                                  val_split: float = 0.2,
                                  epochs: int = 150,
                                  batch_size: int = 4,
                                  lr: float = 0.001,
                                  **kwargs) -> DepthAwarePoseGAT:
    """
    Train depth-aware GAT with enhanced monitoring and visualization
    
    Args:
        dataset: Full dataset
        val_split: Validation split ratio
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        **kwargs: Additional arguments for trainer
        
    Returns:
        Trained model
    """
    # Split dataset
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    
    # Random split
    indices = torch.randperm(len(dataset))
    train_dataset = [dataset[i] for i in indices[:n_train]]
    val_dataset = [dataset[i] for i in indices[n_train:]]
    
    print(f"Training set: {len(train_dataset)} graphs")
    print(f"Validation set: {len(val_dataset)} graphs")
    
    # Initialize model
    model = DepthAwarePoseGAT(
        node_features=4,
        joint_embedding_dim=16,
        hidden_dim=64,
        output_dim=128,
        num_heads=4,
        num_layers=3
    )
    
    # Create trainer
    trainer = DepthAwareGATTrainer(model)
    
    # Train
    trained_model = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        **kwargs
    )
    
    return trained_model