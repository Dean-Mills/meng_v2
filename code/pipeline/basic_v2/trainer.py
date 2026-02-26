import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from .gat import DepthAwareGAT 
from .preprocess import KeypointPreprocessor

class BasicGATTrainer:
    """Trainer for pairwise joint grouping"""
    
    def __init__(self, model: DepthAwareGAT, preprocessor: KeypointPreprocessor, output_dir: str = "code/pipeline/outputs/basic_v2"):
        self.model = model
        self.preprocessor = preprocessor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # History tracking
        self.history = {
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': []
        }

    def train(self, dataset, epochs: int = 100, batch_size: int = 8, lr: float = 0.001):
        """
        Train the GAT for pairwise joint grouping
        
        Args:
            dataset: List of Data objects (from preprocessor)
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
        """
        # Create dataloader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer - include both model and preprocessor parameters
        all_params = list(self.model.parameters()) + list(self.preprocessor.joint_embeddings.parameters())
        optimizer = torch.optim.Adam(all_params, lr=lr)
        
        print(f"Training on {len(dataset)} graphs for {epochs} epochs")
        print(f"Device: {self.device}")
        print("-" * 50)
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            num_batches = 0
            
            self.model.train()
            for batch in loader:
                batch = batch.to(self.device)
                
                # Get embeddings from GAT
                embeddings = self.model(batch.x, batch.edge_index, batch.batch)
                
                # Create pairwise training data
                pair_labels, pair_predictions = self.create_pairwise_batch_data(
                    embeddings, batch.person_labels, batch.batch
                )
                
                if len(pair_labels) == 0:
                    continue
                    
                # Loss and metrics
                loss = F.binary_cross_entropy_with_logits(pair_predictions, pair_labels.float())
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                epoch_loss += loss.item()
                predicted = (pair_predictions > 0).long()
                epoch_correct += (predicted == pair_labels).sum().item()
                epoch_total += len(pair_labels)
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
            
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(accuracy)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
            
            if (epoch + 1) % 50 == 0:
                self.visualize_progress(epoch + 1, loader)
    
        self._create_final_plots()
    
    def create_pairwise_batch_data(self, embeddings, person_labels, batch_assignment):
        """
        Create pairwise training data from batched graphs
        
        Args:
            embeddings: Joint embeddings [total_nodes_in_batch, output_dim]
            person_labels: Person assignment for each node [total_nodes_in_batch]
            batch_assignment: Which graph each node belongs to [total_nodes_in_batch]
            
        Returns:
            pair_labels: Binary labels [num_pairs] - 1 if same person, 0 if different
            pair_predictions: Model predictions [num_pairs] 
        """
        pair_labels = []
        pair_embeddings = []
        
        # Get unique batch indices
        unique_batches = torch.unique(batch_assignment)
        
        # Process each graph in the batch separately
        for batch_idx in unique_batches:
            # Get nodes for this specific graph
            mask = (batch_assignment == batch_idx)
            graph_embeddings = embeddings[mask]  # [nodes_in_this_graph, output_dim]
            graph_person_labels = person_labels[mask]  # [nodes_in_this_graph]
            
            n_nodes = graph_embeddings.size(0)
            
            # Create all pairs within this graph
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    # Get the pair of embeddings
                    pair_embedding = torch.cat([graph_embeddings[i], graph_embeddings[j]], dim=0)
                    pair_embeddings.append(pair_embedding)
                    
                    # Create label: 1 if same person, 0 if different person
                    same_person = (graph_person_labels[i] == graph_person_labels[j]).item()
                    pair_labels.append(1 if same_person else 0)
        
        # Handle empty case
        if not pair_embeddings:
            empty_preds = torch.empty(0, device=embeddings.device)
            empty_labels = torch.empty(0, dtype=torch.long, device=embeddings.device)
            return empty_labels, empty_preds
        
        # Stack all pairs and get predictions
        pair_features = torch.stack(pair_embeddings)  # [total_pairs, output_dim * 2]
        pair_predictions = self.model.pairwise_classifier(pair_features).squeeze()  # [total_pairs]
        
        # Convert labels to tensor
        pair_labels = torch.tensor(pair_labels, dtype=torch.long, device=embeddings.device)
        
        return pair_labels, pair_predictions
    
    def visualize_progress(self, epoch, loader):
        """Create visualizations during training"""
        self.model.eval()
        
        with torch.no_grad():
            # Get one batch for visualization
            for batch in loader:
                batch = batch.to(self.device)
                embeddings = self.model(batch.x, batch.edge_index, batch.batch)
                
                # Visualize embeddings colored by true person labels
                self._plot_embeddings(embeddings, batch.person_labels, batch.batch, epoch)
                
                # Visualize pairwise prediction accuracy
                pair_labels, pair_predictions = self.create_pairwise_batch_data(
                    embeddings, batch.person_labels, batch.batch
                )
                if len(pair_labels) > 0:
                    self._plot_pairwise_accuracy(pair_labels, pair_predictions, epoch)
                
                break  # Only visualize first batch
    
    def _plot_embeddings(self, embeddings, person_labels, batch_assignment, epoch):
        """Plot embeddings colored by person using PCA"""
        from sklearn.decomposition import PCA
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings.cpu().numpy())
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=person_labels.cpu(), cmap='tab10', s=50, alpha=0.7)
        plt.colorbar(scatter, label='Person ID')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title(f'Joint Embeddings by Person (Epoch {epoch})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/embeddings_epoch_{epoch}.png", dpi=150)
        plt.close()

    def _plot_pairwise_accuracy(self, pair_labels, pair_predictions, epoch):
        """Plot pairwise prediction results"""
        # Convert to numpy
        labels = pair_labels.cpu().numpy()
        preds = (pair_predictions > 0).cpu().numpy()
        
        # Confusion matrix style plot
        same_person_correct = ((labels == 1) & (preds == 1)).sum()
        same_person_total = (labels == 1).sum()
        diff_person_correct = ((labels == 0) & (preds == 0)).sum()
        diff_person_total = (labels == 0).sum()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy by category
        categories = ['Same Person', 'Different Person']
        accuracies = [
            same_person_correct / max(same_person_total, 1),
            diff_person_correct / max(diff_person_total, 1)
        ]
        
        ax1.bar(categories, accuracies, color=['green', 'blue'], alpha=0.7)
        ax1.set_ylabel('Accuracy')
        ax1.set_title(f'Pairwise Prediction Accuracy (Epoch {epoch})')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Distribution of prediction scores
        same_scores = pair_predictions[pair_labels == 1].cpu().numpy()
        diff_scores = pair_predictions[pair_labels == 0].cpu().numpy()
        
        ax2.hist(same_scores, bins=20, alpha=0.7, label='Same Person', color='green')
        ax2.hist(diff_scores, bins=20, alpha=0.7, label='Different Person', color='blue')
        ax2.set_xlabel('Prediction Score')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Prediction Scores')
        ax2.legend()
        ax2.axvline(0, color='red', linestyle='--', label='Decision Boundary')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/pairwise_epoch_{epoch}.png", dpi=150)
        plt.close()
    
    def _create_final_plots(self):
        """Create final summary plots"""
        # Training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        ax1.plot(self.history['loss'], linewidth=2, label='Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Accuracy curve
        ax2.plot(self.history['accuracy'], linewidth=2, label='Accuracy', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_summary.png", dpi=150)
        plt.close()
        
        print(f"\nTraining complete! Visualizations saved to {self.output_dir}/")