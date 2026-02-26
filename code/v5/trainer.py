import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from settings import settings
import os
import uuid

class Trainer:
    """Simple trainer using contrastive loss for keypoint embeddings"""
    
    def __init__(self, model, preprocessor, device='cuda'):
        settings.output_dir
        self.model = model.to(device)
        self.preprocessor = preprocessor
        self.device = device
        self.reference = str(uuid.uuid4())
        self.output_dir = settings.output_dir / self.reference
        os.makedirs(self.output_dir, exist_ok=True)
        settings.save_hyperparameters(self.output_dir)
        
        self.history = {
            'loss': [],
            'positive_distance': [],  # Average distance between same-person pairs
            'negative_distance': []   # Average distance between different-person pairs
        }
    
    def contrastive_loss(self, embeddings, person_labels):
        """
        Contrastive loss: pull same-person embeddings together, push different-person apart
        
        Args:
            embeddings: Node embeddings [N, embedding_dim]
            person_labels: Which person each node belongs to [N]
            margin: Minimum distance for negative pairs
            
        Returns:
            loss: Contrastive loss value
            pos_dist: Average positive pair distance
            neg_dist: Average negative pair distance
        """
        margin = settings.margin
        n_nodes = embeddings.size(0)
        
        if n_nodes < 2:
            return torch.tensor(0.0, device=self.device), 0.0, 0.0
        
        # Create all pairs
        positive_losses = []
        negative_losses = []
        pos_distances = []
        neg_distances = []
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                # Calculate distance between embeddings
                emb_i = embeddings[i]
                emb_j = embeddings[j]
                # distance = F.pairwise_distance(emb_i.unsqueeze(0), emb_j.unsqueeze(0))
                cos_sim = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0))
                
                # Check if same person or different person
                same_person = (person_labels[i] == person_labels[j]).item()
                
                if same_person:
                    positive_losses.append(1 - cos_sim)
                    pos_distances.append(cos_sim.item())
                else:
                    negative_losses.append(F.relu(cos_sim - margin))
                    neg_distances.append(cos_sim.item())
        
        # Calculate total loss
        total_loss = 0.0
        avg_pos_dist = 0.0
        avg_neg_dist = 0.0
        
        if positive_losses:
            pos_loss = torch.stack(positive_losses).mean()
            total_loss += pos_loss
            avg_pos_dist = np.mean(pos_distances)
        
        if negative_losses:
            neg_loss = torch.stack(negative_losses).mean()
            total_loss += neg_loss
            avg_neg_dist = np.mean(neg_distances)
        
        return total_loss, avg_pos_dist, avg_neg_dist
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_pos_dist = 0
        epoch_neg_dist = 0
        num_batches = 0
        
        for batch in dataloader:
            # Process batch into graphs
            graphs = self.preprocessor.process_batch(batch)
            
            if not graphs:
                continue
            
            batch_loss = 0
            batch_pos_dist = 0
            batch_neg_dist = 0
            
            # Process each graph separately (no batching for simplicity)
            for graph in graphs:
                graph = graph.to(self.device)
                
                # Get embeddings
                embeddings = self.model(graph)
                
                # Calculate contrastive loss
                loss, pos_dist, neg_dist = self.contrastive_loss(
                    embeddings, graph.person_labels
                )
                
                batch_loss += loss
                batch_pos_dist += pos_dist
                batch_neg_dist += neg_dist
            
            if len(graphs) > 0:
                # Average over graphs in batch
                batch_loss = batch_loss / len(graphs)
                batch_pos_dist = batch_pos_dist / len(graphs)
                batch_neg_dist = batch_neg_dist / len(graphs)
                
                # Backward pass
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                epoch_loss += batch_loss.item()
                epoch_pos_dist += batch_pos_dist
                epoch_neg_dist += batch_neg_dist
                num_batches += 1
        
        # Return average metrics
        if num_batches > 0:
            return epoch_loss / num_batches, epoch_pos_dist / num_batches, epoch_neg_dist / num_batches
        else:
            return 0.0, 0.0, 0.0
    
    def train(self, dataloader):
        """Main training loop"""
        epochs = settings.epochs
        lr = settings.learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print("-" * 50)
        
        for epoch in range(epochs):
            avg_loss, avg_pos_dist, avg_neg_dist = self.train_epoch(dataloader)
            
            self.history['loss'].append(avg_loss)
            self.history['positive_distance'].append(avg_pos_dist)
            self.history['negative_distance'].append(avg_neg_dist)
            
            if (epoch + 1) % settings.print_trainer_results_every_n_epochs == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Loss: {avg_loss:.4f}")
                print(f"  Pos Distance: {avg_pos_dist:.4f}")
                print(f"  Neg Distance: {avg_neg_dist:.4f}")
                print()
            
            # Visualize progress
            if (epoch + 1) % settings.visualize_trainer_results_every_n_epochs == 0:
                self.visualize_progress(dataloader, epoch + 1)
        
        self.plot_training_curves()
        print("Training complete!")
    
    def visualize_progress(self, dataloader, epoch):
        """Visualize embeddings during training"""
        self.model.eval()
        
        with torch.no_grad():
            # Get first batch for visualization
            batch = next(iter(dataloader))
            graphs = self.preprocessor.process_batch(batch)
            
            if not graphs:
                return
            
            # Use first graph
            graph = graphs[0].to(self.device)
            embeddings = self.model(graph)
            
            # Reduce to 2D with PCA
            if embeddings.size(0) > 1:
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(embeddings.cpu().numpy())
                person_labels = graph.person_labels.cpu().numpy()
                
                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                    c=person_labels, cmap='tab10', s=50, alpha=0.7)
                plt.colorbar(scatter, label='Person ID')
                plt.title(f'Keypoint Embeddings (Epoch {epoch})')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/embeddings_epoch_{epoch}.png", dpi=150)
                plt.close()
    
    def plot_training_curves(self):
        """Plot training progress (cosine similarity version)."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Loss curve
        ax1.plot(self.history['loss'], linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Contrastive Loss')
        ax1.grid(True, alpha=0.3)

        # Similarity curves (cosine)
        ax2.plot(self.history['positive_distance'], label='Same Person (Pos Sim)', color='green', linewidth=2)
        ax2.plot(self.history['negative_distance'], label='Different Person (Neg Sim)', color='red', linewidth=2)
        ax2.set_title('Embedding Similarities (Cosine)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Average Cosine Similarity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Similarity separation (what we want to maximize): pos - neg
        pos = np.array(self.history['positive_distance'])
        neg = np.array(self.history['negative_distance'])
        separation = pos - neg
        ax3.plot(separation, color='blue', linewidth=2)
        ax3.set_title('Similarity Separation')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Pos Sim - Neg Sim')
        ax3.grid(True, alpha=0.3)

        # Summary stats
        final_loss = self.history['loss'][-1] if self.history['loss'] else 0.0
        final_pos  = self.history['positive_distance'][-1] if self.history['positive_distance'] else 0.0
        final_neg  = self.history['negative_distance'][-1] if self.history['negative_distance'] else 0.0
        final_sep  = final_pos - final_neg

        ax4.text(0.1, 0.72, f"Final Loss: {final_loss:.4f}", fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.52, f"Final Pos Sim: {final_pos:.4f}", fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.32, f"Final Neg Sim: {final_neg:.4f}", fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.12, f"Separation (Pos-Neg): {final_sep:.4f}", fontsize=12, transform=ax4.transAxes)
        ax4.set_title('Final Metrics')
        ax4.axis('off')

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_summary.png", dpi=150)
        plt.show()
        plt.close()