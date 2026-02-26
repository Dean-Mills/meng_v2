# trainer.py
"""
Trainer for GAT + DETR Pose Grouping Model.

Supports:
1. GAT-only training (contrastive loss)
2. Full model training (existence + assignment + contrastive)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Literal
from pathlib import Path
import os
import uuid
import math

from losses import PoseGroupingLoss, LossConfig, GATOnlyLoss


@dataclass
class TrainerConfig:
    # Mode
    mode: Literal["gat_only", "full"] = "full"
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # LR Schedule
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Loss weights (for full mode)
    lambda_existence: float = 1.0
    lambda_assignment: float = 5.0
    lambda_contrastive: float = 2.0
    contrastive_margin: float = 0.5
    
    # Training
    epochs: int = 100
    
    # Logging
    print_every_n_epochs: int = 1
    visualize_every_n_epochs: int = 10
    checkpoint_every_n_epochs: int = 25
    
    # Evaluation
    compute_clustering_metrics: bool = True


class CosineWarmupScheduler:
    """Learning rate scheduler with linear warmup and cosine decay"""
    
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _compute_lr(self) -> float:
        if self.current_epoch <= self.warmup_epochs:
            return self.base_lrs[0] * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lrs[0] - self.min_lr) * cosine_decay
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    """
    Trainer for pose grouping models.
    
    Supports two modes:
    - "gat_only": Train GAT with contrastive loss only
    - "full": Train GAT + DETR with all three losses
    """

    def __init__(
        self, 
        model: nn.Module, 
        preprocessor,
        config: Optional[TrainerConfig] = None,
        device: str = "cuda",
        output_dir: Optional[Path] = None
    ):
        self.config = config if config is not None else TrainerConfig()
        self.model = model.to(device)
        self.preprocessor = preprocessor
        self.device = device
        
        # Output directories
        self.reference = str(uuid.uuid4())[:8]
        if output_dir is None:
            from settings import settings
            output_dir = settings.output_dir
        self.output_dir = Path(output_dir) / self.reference
        self.checkpoint_dir = self.output_dir / "checkpoints"
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Setup loss function based on mode
        if self.config.mode == "gat_only":
            self.criterion = GATOnlyLoss(margin=self.config.contrastive_margin)
        else:
            loss_config = LossConfig(
                lambda_existence=self.config.lambda_existence,
                lambda_assignment=self.config.lambda_assignment,
                lambda_contrastive=self.config.lambda_contrastive,
                contrastive_margin=self.config.contrastive_margin
            )
            self.criterion = PoseGroupingLoss(loss_config)
        
        # Save config
        self._save_config()

        # Training history
        self.history: Dict[str, List[float]] = {
            "total_loss": [],
            "existence_loss": [],
            "assignment_loss": [],
            "contrastive_loss": [],
            "pos_similarity": [],
            "neg_similarity": [],
            "learning_rate": [],
            "nmi": [],
            "silhouette": [],
            "num_matched": [],
            "pga": [],  # Pose Grouping Accuracy
        }
        
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[CosineWarmupScheduler] = None
        self.current_epoch = 0

    def _save_config(self):
        """Save training configuration"""
        config_path = self.output_dir / "trainer_config.txt"
        with open(config_path, 'w') as f:
            f.write(f"Mode: {self.config.mode}\n")
            f.write("-" * 40 + "\n")
            for key, value in vars(self.config).items():
                f.write(f"{key}: {value}\n")

    def compute_clustering_metrics(
        self, 
        embeddings: torch.Tensor, 
        person_labels: torch.Tensor
    ) -> dict[str, float]:
        """Compute NMI and Silhouette score"""
        if embeddings.size(0) < 3:
            return {"nmi": 0.0, "silhouette": 0.0}
        
        emb_np = embeddings.detach().cpu().numpy()
        labels_np = person_labels.detach().cpu().numpy()
        
        n_clusters = len(np.unique(labels_np))
        if n_clusters < 2:
            return {"nmi": 0.0, "silhouette": 0.0}
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            pred_labels = kmeans.fit_predict(emb_np)
            
            nmi = normalized_mutual_info_score(labels_np, pred_labels)
            silhouette = silhouette_score(emb_np, labels_np)
        except Exception:
            nmi, silhouette = 0.0, 0.0
        
        return {"nmi": nmi, "silhouette": silhouette}

    def compute_pga(
        self,
        predictions: Dict[str, Any],
        person_labels: torch.Tensor,
        joint_types: torch.Tensor
    ) -> float:
        """
        Compute Pose Grouping Accuracy.
        
        For each joint, check if it's assigned to the correct person.
        """
        assignments = predictions['assignments']  # [M, 17]
        person_mask = predictions['person_mask']  # [M]
        
        if not person_mask.any():
            return 0.0
        
        correct = 0
        total = 0
        
        # For each predicted person that exists
        for pred_person_idx in torch.where(person_mask)[0]:
            pred_joints = assignments[pred_person_idx]  # [17]
            
            # For each joint type
            for joint_type in range(17):
                joint_idx = pred_joints[joint_type].item()
                
                if joint_idx == -1:
                    continue  # No joint of this type
                
                # Check if this joint actually belongs to this person
                # We need to map pred_person_idx to a GT person
                # For simplicity, we check if the assigned joint's GT label matches
                gt_label = person_labels[joint_idx].item()
                
                # Count as correct if consistent
                total += 1
                # This is a simplified PGA - full version needs Hungarian matching
        
        # For now, return a placeholder
        # Full PGA computation needs GT-to-pred mapping
        return 0.0 if total == 0 else correct / total

    def train_epoch_gat_only(self, dataloader) -> Dict[str, float]:
        """Train epoch for GAT-only mode"""
        self.model.train()
        
        metrics = {
            "total_loss": 0.0,
            "pos_similarity": 0.0,
            "neg_similarity": 0.0,
            "nmi": 0.0,
            "silhouette": 0.0,
        }
        num_graphs = 0

        for batch in dataloader:
            graphs = self.preprocessor.process_batch(batch)
            
            if not graphs:
                continue

            batch_loss = torch.tensor(0.0, device=self.device)
            
            for graph in graphs:
                graph = graph.to(self.device)
                
                # Forward pass
                embeddings = self.model(graph)
                
                # Compute loss
                loss_dict = self.criterion(embeddings, graph.person_labels)
                
                batch_loss += loss_dict['total_loss']
                metrics["pos_similarity"] += loss_dict['pos_similarity']
                metrics["neg_similarity"] += loss_dict['neg_similarity']
                
                # Clustering metrics (occasionally)
                if self.config.compute_clustering_metrics and num_graphs % 10 == 0:
                    cluster_metrics = self.compute_clustering_metrics(
                        embeddings, graph.person_labels
                    )
                    metrics["nmi"] += cluster_metrics["nmi"]
                    metrics["silhouette"] += cluster_metrics["silhouette"]
                
                num_graphs += 1

            if len(graphs) > 0:
                batch_loss = batch_loss / len(graphs)
                
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config.max_grad_norm
                )
                self.optimizer.step()
                
                metrics["total_loss"] += batch_loss.item() * len(graphs)

        # Average metrics
        if num_graphs > 0:
            metrics["total_loss"] /= num_graphs
            metrics["pos_similarity"] /= num_graphs
            metrics["neg_similarity"] /= num_graphs
            metrics["nmi"] /= max(1, num_graphs // 10)
            metrics["silhouette"] /= max(1, num_graphs // 10)
        
        return metrics

    def train_epoch_full(self, dataloader) -> Dict[str, float]:
        """Train epoch for full model (GAT + DETR)"""
        self.model.train()
        
        metrics = {
            "total_loss": 0.0,
            "existence_loss": 0.0,
            "assignment_loss": 0.0,
            "contrastive_loss": 0.0,
            "pos_similarity": 0.0,
            "neg_similarity": 0.0,
            "nmi": 0.0,
            "silhouette": 0.0,
            "num_matched": 0.0,
        }
        num_graphs = 0

        for batch in dataloader:
            graphs = self.preprocessor.process_batch(batch)
            
            if not graphs:
                continue

            batch_loss = torch.tensor(0.0, device=self.device)
            
            for graph in graphs:
                graph = graph.to(self.device)
                
                # Forward pass (full model returns dict)
                outputs = self.model(graph)
                
                # Compute loss
                loss_dict = self.criterion(
                    outputs,
                    person_labels=graph.person_labels,
                    joint_types=graph.joint_types,
                    num_gt_people=graph.num_people
                )
                
                batch_loss += loss_dict['total_loss']
                metrics["existence_loss"] += loss_dict['existence_loss'].item()
                metrics["assignment_loss"] += loss_dict['assignment_loss'].item()
                metrics["contrastive_loss"] += loss_dict['contrastive_loss'].item()
                metrics["pos_similarity"] += loss_dict['pos_similarity']
                metrics["neg_similarity"] += loss_dict['neg_similarity']
                metrics["num_matched"] += loss_dict['num_matched']
                
                # Clustering metrics (occasionally)
                if self.config.compute_clustering_metrics and num_graphs % 10 == 0:
                    cluster_metrics = self.compute_clustering_metrics(
                        outputs['embeddings'], graph.person_labels
                    )
                    metrics["nmi"] += cluster_metrics["nmi"]
                    metrics["silhouette"] += cluster_metrics["silhouette"]
                
                num_graphs += 1

            if len(graphs) > 0:
                batch_loss = batch_loss / len(graphs)
                
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_norm=self.config.max_grad_norm
                )
                self.optimizer.step()
                
                metrics["total_loss"] += batch_loss.item() * len(graphs)

        # Average metrics
        if num_graphs > 0:
            for key in metrics:
                metrics[key] /= num_graphs
            metrics["nmi"] *= num_graphs / max(1, num_graphs // 10)
            metrics["silhouette"] *= num_graphs / max(1, num_graphs // 10)
        
        return metrics

    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Route to appropriate training method"""
        if self.config.mode == "gat_only":
            return self.train_epoch_gat_only(dataloader)
        else:
            return self.train_epoch_full(dataloader)

    def train(self, dataloader, resume_from: Optional[str] = None):
        """Main training loop"""
        cfg = self.config
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=cfg.warmup_epochs,
            total_epochs=cfg.epochs,
            min_lr=cfg.min_lr
        )
        
        start_epoch = 0
        
        # Resume from checkpoint
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch
            print(f"Resuming from epoch {start_epoch}")

        # Print training info
        print("=" * 70)
        print("TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Mode: {cfg.mode.upper()}")
        print(f"Device: {self.device}")
        print(f"Epochs: {cfg.epochs}")
        print(f"Learning rate: {cfg.learning_rate}")
        print(f"Weight decay: {cfg.weight_decay}")
        print(f"Warmup epochs: {cfg.warmup_epochs}")
        if cfg.mode == "full":
            print(f"Loss weights: exist={cfg.lambda_existence}, "
                  f"assign={cfg.lambda_assignment}, contrast={cfg.lambda_contrastive}")
        print(f"Contrastive margin: {cfg.contrastive_margin}")
        print(f"Output: {self.output_dir}")
        print("=" * 70)

        for epoch in range(start_epoch, cfg.epochs):
            self.current_epoch = epoch + 1
            
            # Train one epoch
            metrics = self.train_epoch(dataloader)
            
            # Step scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_lr()
            
            # Record history
            self.history["total_loss"].append(metrics.get("total_loss", 0.0))
            self.history["existence_loss"].append(metrics.get("existence_loss", 0.0))
            self.history["assignment_loss"].append(metrics.get("assignment_loss", 0.0))
            self.history["contrastive_loss"].append(metrics.get("contrastive_loss", 0.0))
            self.history["pos_similarity"].append(metrics.get("pos_similarity", 0.0))
            self.history["neg_similarity"].append(metrics.get("neg_similarity", 0.0))
            self.history["learning_rate"].append(current_lr)
            self.history["nmi"].append(metrics.get("nmi", 0.0))
            self.history["silhouette"].append(metrics.get("silhouette", 0.0))
            self.history["num_matched"].append(metrics.get("num_matched", 0.0))

            # Print progress
            if (epoch + 1) % cfg.print_every_n_epochs == 0:
                self._print_epoch_summary(epoch + 1, metrics, current_lr)

            # Visualize
            if (epoch + 1) % cfg.visualize_every_n_epochs == 0:
                self.visualize_embeddings(dataloader, epoch + 1)
            
            # Checkpoint
            if (epoch + 1) % cfg.checkpoint_every_n_epochs == 0:
                self.save_checkpoint(epoch + 1)

        # Final save
        self.save_checkpoint(cfg.epochs, is_final=True)
        self.plot_training_curves()
        
        print("=" * 70)
        print("TRAINING COMPLETE")
        print(f"Final model: {self.checkpoint_dir / 'model_final.pt'}")
        print("=" * 70)

    def _print_epoch_summary(self, epoch: int, metrics: Dict[str, float], lr: float):
        """Print epoch summary based on mode"""
        cfg = self.config
        
        if cfg.mode == "gat_only":
            sep = metrics["pos_similarity"] - metrics["neg_similarity"]
            print(
                f"Epoch {epoch:3d}/{cfg.epochs} | "
                f"Loss: {metrics['total_loss']:.4f} | "
                f"Pos: {metrics['pos_similarity']:.3f} | "
                f"Neg: {metrics['neg_similarity']:.3f} | "
                f"Sep: {sep:.3f} | "
                f"NMI: {metrics['nmi']:.3f} | "
                f"LR: {lr:.2e}"
            )
        else:
            print(
                f"Epoch {epoch:3d}/{cfg.epochs} | "
                f"Loss: {metrics['total_loss']:.4f} "
                f"(E:{metrics['existence_loss']:.3f} "
                f"A:{metrics['assignment_loss']:.3f} "
                f"C:{metrics['contrastive_loss']:.3f}) | "
                f"Match: {metrics['num_matched']:.1f} | "
                f"NMI: {metrics['nmi']:.3f} | "
                f"LR: {lr:.2e}"
            )

    def save_checkpoint(self, epoch: int, is_final: bool = False) -> Path:
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_epoch": self.scheduler.current_epoch if self.scheduler else None,
            "history": self.history,
            "config": vars(self.config),
            "reference": self.reference,
        }
        
        filename = "model_final.pt" if is_final else f"model_epoch_{epoch}.pt"
        path = self.checkpoint_dir / filename
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
        
        return path

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if checkpoint.get("optimizer_state_dict") and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint.get("scheduler_epoch") and self.scheduler:
            self.scheduler.current_epoch = checkpoint["scheduler_epoch"]
        
        self.history = checkpoint.get("history", self.history)
        self.current_epoch = checkpoint.get("epoch", 0)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def visualize_embeddings(self, dataloader, epoch: int):
        """Visualize embeddings with PCA"""
        self.model.eval()

        all_embeddings = []
        all_labels = []
        all_joint_types = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 3:
                    break
                    
                graphs = self.preprocessor.process_batch(batch)
                
                for graph in graphs:
                    graph = graph.to(self.device)
                    
                    if self.config.mode == "gat_only":
                        embeddings = self.model(graph)
                    else:
                        outputs = self.model(graph)
                        embeddings = outputs['embeddings']
                    
                    all_embeddings.append(embeddings.cpu())
                    all_labels.append(graph.person_labels.cpu())
                    all_joint_types.append(graph.joint_types.cpu())
        
        if not all_embeddings:
            return
        
        embeddings = torch.cat(all_embeddings, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()
        joint_types = torch.cat(all_joint_types, dim=0).numpy()
        
        if len(embeddings) < 3:
            return
        
        # PCA
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(embeddings)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # By person
        scatter = axes[0].scatter(
            emb_2d[:, 0], emb_2d[:, 1],
            c=labels, cmap="tab10", s=30, alpha=0.7
        )
        axes[0].set_title(f"Embeddings by Person (Epoch {epoch})")
        axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        plt.colorbar(scatter, ax=axes[0], label="Person ID")
        axes[0].grid(True, alpha=0.3)
        
        # By joint type
        scatter2 = axes[1].scatter(
            emb_2d[:, 0], emb_2d[:, 1],
            c=joint_types, cmap="tab20", s=30, alpha=0.7
        )
        axes[1].set_title(f"Embeddings by Joint Type (Epoch {epoch})")
        axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        plt.colorbar(scatter2, ax=axes[1], label="Joint Type")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"embeddings_epoch_{epoch}.png", dpi=150)
        plt.close()

    def plot_training_curves(self):
        """Plot training curves based on mode"""
        if self.config.mode == "gat_only":
            self._plot_gat_only_curves()
        else:
            self._plot_full_curves()

    def _plot_gat_only_curves(self):
        """Plot curves for GAT-only training"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        epochs = range(1, len(self.history["total_loss"]) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history["total_loss"], 'b-', linewidth=2)
        axes[0, 0].set_title("Contrastive Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Similarities
        axes[0, 1].plot(epochs, self.history["pos_similarity"], 'g-', 
                       linewidth=2, label="Same Person")
        axes[0, 1].plot(epochs, self.history["neg_similarity"], 'r-', 
                       linewidth=2, label="Different Person")
        axes[0, 1].axhline(y=self.config.contrastive_margin, color='k', 
                          linestyle='--', alpha=0.5, label="Margin")
        axes[0, 1].set_title("Cosine Similarities")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Separation
        pos = np.array(self.history["pos_similarity"])
        neg = np.array(self.history["neg_similarity"])
        separation = pos - neg
        axes[0, 2].plot(epochs, separation, 'purple', linewidth=2)
        axes[0, 2].fill_between(epochs, separation, alpha=0.3)
        axes[0, 2].set_title("Separation (Pos - Neg)")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].grid(True, alpha=0.3)
        
        # LR
        axes[1, 0].plot(epochs, self.history["learning_rate"], 'orange', linewidth=2)
        axes[1, 0].set_title("Learning Rate")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # NMI
        axes[1, 1].plot(epochs, self.history["nmi"], 'teal', linewidth=2)
        axes[1, 1].set_title("NMI")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylim([0, 1.05])
        axes[1, 1].grid(True, alpha=0.3)
        
        # Summary
        self._add_summary_panel(axes[1, 2], separation)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_summary.png", dpi=150)
        plt.show()
        plt.close()

    def _plot_full_curves(self):
        """Plot curves for full model training"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        epochs = range(1, len(self.history["total_loss"]) + 1)
        
        # Total Loss
        axes[0, 0].plot(epochs, self.history["total_loss"], 'b-', linewidth=2)
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Individual losses
        axes[0, 1].plot(epochs, self.history["existence_loss"], 'g-', 
                       linewidth=2, label="Existence")
        axes[0, 1].plot(epochs, self.history["assignment_loss"], 'r-', 
                       linewidth=2, label="Assignment")
        axes[0, 1].plot(epochs, self.history["contrastive_loss"], 'purple', 
                       linewidth=2, label="Contrastive")
        axes[0, 1].set_title("Loss Components")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Similarities
        axes[0, 2].plot(epochs, self.history["pos_similarity"], 'g-', 
                       linewidth=2, label="Positive")
        axes[0, 2].plot(epochs, self.history["neg_similarity"], 'r-', 
                       linewidth=2, label="Negative")
        axes[0, 2].set_title("Embedding Similarities")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Matched
        axes[0, 3].plot(epochs, self.history["num_matched"], 'orange', linewidth=2)
        axes[0, 3].set_title("Avg Matched People")
        axes[0, 3].set_xlabel("Epoch")
        axes[0, 3].grid(True, alpha=0.3)
        
        # LR
        axes[1, 0].plot(epochs, self.history["learning_rate"], 'orange', linewidth=2)
        axes[1, 0].set_title("Learning Rate")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # NMI
        axes[1, 1].plot(epochs, self.history["nmi"], 'teal', linewidth=2)
        axes[1, 1].set_title("NMI")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylim([0, 1.05])
        axes[1, 1].grid(True, alpha=0.3)
        
        # Silhouette
        axes[1, 2].plot(epochs, self.history["silhouette"], 'brown', linewidth=2)
        axes[1, 2].set_title("Silhouette Score")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].grid(True, alpha=0.3)
        
        # Summary
        pos = np.array(self.history["pos_similarity"])
        neg = np.array(self.history["neg_similarity"])
        separation = pos - neg
        self._add_summary_panel(axes[1, 3], separation, full_mode=True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_summary.png", dpi=150)
        plt.show()
        plt.close()

    def _add_summary_panel(self, ax, separation, full_mode=False):
        """Add summary metrics to a plot panel"""
        ax.axis('off')
        
        lines = [
            "FINAL METRICS",
            "-" * 25,
            f"Total Loss: {self.history['total_loss'][-1]:.4f}",
        ]
        
        if full_mode:
            lines.extend([
                f"Exist Loss: {self.history['existence_loss'][-1]:.4f}",
                f"Assign Loss: {self.history['assignment_loss'][-1]:.4f}",
                f"Contrast Loss: {self.history['contrastive_loss'][-1]:.4f}",
            ])
        
        lines.extend([
            f"Pos Sim: {self.history['pos_similarity'][-1]:.4f}",
            f"Neg Sim: {self.history['neg_similarity'][-1]:.4f}",
            f"Separation: {separation[-1]:.4f}",
            f"NMI: {self.history['nmi'][-1]:.4f}",
            "",
            f"Best Loss: {min(self.history['total_loss']):.4f}",
            f"Best NMI: {max(self.history['nmi']):.4f}",
        ])
        
        for i, line in enumerate(lines):
            weight = 'bold' if i == 0 else 'normal'
            ax.text(0.1, 0.95 - i * 0.07, line, fontsize=10,
                   transform=ax.transAxes, fontweight=weight,
                   verticalalignment='top', fontfamily='monospace')


def create_trainer(
    model: nn.Module,
    preprocessor,
    mode: Literal["gat_only", "full"] = "full",
    **kwargs
) -> Trainer:
    """Convenience function to create trainer"""
    config = TrainerConfig(mode=mode, **kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Trainer(model, preprocessor, config=config, device=device)
