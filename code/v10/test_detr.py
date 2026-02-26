"""
DETR Decoder Standalone Test
=============================
Tests the DETR decoder in isolation by feeding it synthetic "ideal" GAT embeddings.

What this proves:
- The DETR can learn to count people (existence head)
- The DETR can assign joints to the correct people (assignment heads)
- Hungarian matching loss is working
- The predict() method produces correct groupings

Synthetic data strategy:
- Each person gets a random "identity" vector in embedding space
- All joints belonging to that person get that vector + small noise
- This simulates what a well-trained GAT would produce
- If the DETR can't learn to group these, the architecture is broken
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import sys
import os

from detr_decoder import DETRDecoder, DETRConfig
from losses import PoseGroupingLoss, LossConfig


# ══════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════

def generate_synthetic_sample(
    num_people: int,
    embedding_dim: int = 128,
    num_joint_types: int = 17,
    noise_std: float = 0.05,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Generate one synthetic sample: fake GAT embeddings with known grouping.
    
    Each person gets a random unit vector as their "identity."
    All joints of that person = identity + small gaussian noise.
    This is what a perfect GAT would produce.
    """
    # Generate random identity vectors for each person
    person_identities = torch.randn(num_people, embedding_dim, device=device)
    person_identities = F.normalize(person_identities, p=2, dim=-1)
    
    all_embeddings = []
    all_joint_types = []
    all_person_labels = []
    
    for person_idx in range(num_people):
        for joint_type in range(num_joint_types):
            # Person identity + noise
            embedding = person_identities[person_idx] + torch.randn(embedding_dim, device=device) * noise_std
            embedding = F.normalize(embedding, p=2, dim=-1)
            
            all_embeddings.append(embedding)
            all_joint_types.append(joint_type)
            all_person_labels.append(person_idx)
    
    return {
        'embeddings': torch.stack(all_embeddings),          # [N, D]
        'joint_types': torch.tensor(all_joint_types, device=device),  # [N]
        'person_labels': torch.tensor(all_person_labels, device=device),  # [N]
        'num_people': num_people,
    }


def generate_batch(
    batch_size: int,
    min_people: int = 2,
    max_people: int = 5,
    **kwargs
) -> List[Dict[str, torch.Tensor]]:
    """Generate a batch of synthetic samples with varying person counts."""
    samples = []
    for _ in range(batch_size):
        num_people = torch.randint(min_people, max_people + 1, (1,)).item()
        samples.append(generate_synthetic_sample(num_people=num_people, **kwargs))
    return samples


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(
    model: DETRDecoder,
    samples: List[Dict],
    existence_threshold: float = 0.5
) -> Dict[str, float]:
    """Compute existence and assignment metrics on a set of samples."""
    model.eval()
    
    total_correct_assignments = 0
    total_assignments = 0
    total_existence_correct = 0
    total_existence_total = 0
    total_perfect_count = 0
    total_samples = 0
    
    pred_counts = []
    gt_counts = []
    
    for sample in samples:
        result = model.predict(
            sample['embeddings'],
            sample['joint_types'],
            existence_threshold=existence_threshold
        )
        
        num_gt = sample['num_people']
        num_pred = result['num_people']
        
        gt_counts.append(num_gt)
        pred_counts.append(num_pred)
        
        # Perfect count rate
        if num_pred == num_gt:
            total_perfect_count += 1
        total_samples += 1
        
        # Existence accuracy (per query)
        M = result['person_mask'].size(0)
        existence_target = torch.zeros(M, device=sample['embeddings'].device)
        # We can't directly map GT people to queries, but we can check count
        total_existence_total += 1
        
        # Assignment accuracy (for predicted people)
        if num_pred > 0 and num_gt > 0:
            assignments = result['assignments']  # [M, 17]
            person_mask = result['person_mask']
            
            # For each predicted person, check if their assigned joints
            # all belong to the same GT person
            for query_idx in torch.where(person_mask)[0]:
                assigned_joints = assignments[query_idx]  # [17]
                valid = assigned_joints >= 0
                
                if valid.any():
                    # Get GT labels for assigned joints
                    valid_indices = assigned_joints[valid]
                    gt_labels = sample['person_labels'][valid_indices]
                    
                    # What fraction of this person's joints belong to the majority GT person?
                    if len(gt_labels) > 0:
                        majority_label = gt_labels.mode().values.item()
                        correct = (gt_labels == majority_label).sum().item()
                        total_correct_assignments += correct
                        total_assignments += len(gt_labels)
    
    model.train()
    
    pga = total_correct_assignments / max(total_assignments, 1)
    perfect_count_rate = total_perfect_count / max(total_samples, 1)
    
    # Count error stats
    gt_arr = np.array(gt_counts)
    pred_arr = np.array(pred_counts)
    mean_count_error = np.mean(np.abs(gt_arr - pred_arr))
    
    return {
        'pga': pga,
        'perfect_count_rate': perfect_count_rate,
        'mean_count_error': mean_count_error,
        'avg_gt_count': gt_arr.mean(),
        'avg_pred_count': pred_arr.mean(),
    }


# ══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════

def train_detr_standalone(
    epochs: int = 200,
    batch_size: int = 16,
    lr: float = 1e-4,
    min_people: int = 2,
    max_people: int = 5,
    noise_std: float = 0.05,
    embedding_dim: int = 128,
    eval_every: int = 10,
    device: str = "cpu",
    output_dir: Path = Path("detr_test_outputs"),
):
    """Train the DETR decoder on synthetic embeddings and track metrics."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Model ─────────────────────────────────────────────────────────
    config = DETRConfig(
        embedding_dim=embedding_dim,
        max_people=max_people + 2,  # headroom above max GT
        num_decoder_layers=3,
        num_heads=8,
        ffn_dim=512,
        dropout=0.1,
        num_joint_types=17,
    )
    model = DETRDecoder(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DETR parameters: {num_params:,}")
    
    # ── Loss ──────────────────────────────────────────────────────────
    loss_config = LossConfig(
        lambda_existence=1.0,
        lambda_assignment=5.0,
        lambda_contrastive=0.0,  # No contrastive - we're testing DETR only
        contrastive_margin=0.5,
        label_smoothing=0.0,
    )
    criterion = PoseGroupingLoss(loss_config)
    
    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # ── Fixed eval set ────────────────────────────────────────────────
    eval_samples = generate_batch(
        batch_size=100,
        min_people=min_people,
        max_people=max_people,
        embedding_dim=embedding_dim,
        noise_std=noise_std,
        device=device,
    )
    
    # ── History ───────────────────────────────────────────────────────
    history = {
        'epoch': [],
        'total_loss': [],
        'existence_loss': [],
        'assignment_loss': [],
        'pga': [],
        'perfect_count_rate': [],
        'mean_count_error': [],
        'avg_pred_count': [],
    }
    
    # ── Training ──────────────────────────────────────────────────────
    print(f"\nTraining DETR decoder for {epochs} epochs")
    print(f"People range: {min_people}-{max_people}, noise_std: {noise_std}")
    print(f"Batch size: {batch_size}, LR: {lr}")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        # Generate fresh training batch each epoch
        batch = generate_batch(
            batch_size=batch_size,
            min_people=min_people,
            max_people=max_people,
            embedding_dim=embedding_dim,
            noise_std=noise_std,
            device=device,
        )
        
        epoch_losses = {
            'total': 0.0,
            'existence': 0.0,
            'assignment': 0.0,
        }
        
        for sample in batch:
            optimizer.zero_grad()
            
            # Forward through DETR
            outputs = model(sample['embeddings'], sample['joint_types'])
            
            # Add fake embeddings key for the loss (contrastive weight is 0)
            outputs['embeddings'] = sample['embeddings']
            
            # Compute loss
            loss_dict = criterion(
                outputs,
                sample['person_labels'],
                sample['joint_types'],
                sample['num_people'],
            )
            
            loss = loss_dict['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses['total'] += loss_dict['total_loss'].item()
            epoch_losses['existence'] += loss_dict['existence_loss'].item()
            epoch_losses['assignment'] += loss_dict['assignment_loss'].item()
        
        scheduler.step()
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= batch_size
        
        # ── Evaluate ──────────────────────────────────────────────────
        if epoch % eval_every == 0 or epoch == 1:
            metrics = compute_metrics(model, eval_samples)
            
            history['epoch'].append(epoch)
            history['total_loss'].append(epoch_losses['total'])
            history['existence_loss'].append(epoch_losses['existence'])
            history['assignment_loss'].append(epoch_losses['assignment'])
            history['pga'].append(metrics['pga'])
            history['perfect_count_rate'].append(metrics['perfect_count_rate'])
            history['mean_count_error'].append(metrics['mean_count_error'])
            history['avg_pred_count'].append(metrics['avg_pred_count'])
            
            print(
                f"Epoch {epoch:>4d} | "
                f"Loss: {epoch_losses['total']:.4f} "
                f"(exist: {epoch_losses['existence']:.4f}, assign: {epoch_losses['assignment']:.4f}) | "
                f"PGA: {metrics['pga']:.4f} | "
                f"Count: {metrics['perfect_count_rate']:.2%} perfect, "
                f"err: {metrics['mean_count_error']:.2f} | "
                f"Pred/GT: {metrics['avg_pred_count']:.1f}/{metrics['avg_gt_count']:.1f}"
            )
    
    # ── Final evaluation ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    final_metrics = compute_metrics(model, eval_samples)
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # ── Example predictions ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)
    
    model.eval()
    for i in range(min(5, len(eval_samples))):
        sample = eval_samples[i]
        result = model.predict(sample['embeddings'], sample['joint_types'])
        
        print(f"\n  Sample {i}: GT={sample['num_people']} people, "
              f"Predicted={result['num_people']} people")
        
        probs = result['existence_probs']
        top_probs = probs.sort(descending=True).values[:sample['num_people'] + 2]
        print(f"  Top existence probs: {[f'{p:.3f}' for p in top_probs.tolist()]}")
        
        # Check assignment correctness
        for query_idx in torch.where(result['person_mask'])[0]:
            assigned = result['assignments'][query_idx]
            valid = assigned >= 0
            if valid.any():
                gt_labels = sample['person_labels'][assigned[valid]]
                majority = gt_labels.mode().values.item()
                accuracy = (gt_labels == majority).float().mean().item()
                print(f"    Query {query_idx.item()}: "
                      f"{valid.sum().item()}/17 joints assigned, "
                      f"accuracy={accuracy:.2%}")
    
    # ── Plot ──────────────────────────────────────────────────────────
    plot_results(history, output_dir)
    
    return model, history


# ══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════

def plot_results(history: Dict, output_dir: Path):
    """Plot training curves and metrics."""
    
    epochs = history['epoch']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("DETR Decoder Standalone Test", fontsize=16, fontweight='bold')
    
    # 1. Total loss
    ax = axes[0, 0]
    ax.plot(epochs, history['total_loss'], 'b-', linewidth=1.5)
    ax.set_title('Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 2. Loss breakdown
    ax = axes[0, 1]
    ax.plot(epochs, history['existence_loss'], 'r-', label='Existence', linewidth=1.5)
    ax.plot(epochs, history['assignment_loss'], 'g-', label='Assignment', linewidth=1.5)
    ax.set_title('Loss Components')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. PGA
    ax = axes[0, 2]
    ax.plot(epochs, history['pga'], 'b-', linewidth=1.5)
    ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect')
    ax.set_title('Pose Grouping Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PGA')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Perfect count rate
    ax = axes[1, 0]
    ax.plot(epochs, history['perfect_count_rate'], 'm-', linewidth=1.5)
    ax.set_title('Perfect Count Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rate')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # 5. Mean count error
    ax = axes[1, 1]
    ax.plot(epochs, history['mean_count_error'], 'r-', linewidth=1.5)
    ax.axhline(y=0, color='g', linestyle='--', alpha=0.5)
    ax.set_title('Mean Count Error')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|pred - gt|')
    ax.grid(True, alpha=0.3)
    
    # 6. Avg predicted count vs GT
    ax = axes[1, 2]
    ax.plot(epochs, history['avg_pred_count'], 'b-', label='Predicted', linewidth=1.5)
    gt_mean = history['avg_pred_count'][0]  # roughly constant
    ax.axhline(y=3.5, color='g', linestyle='--', alpha=0.5, label='~GT mean (3.5)')
    ax.set_title('Avg Predicted Count')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = output_dir / "detr_test_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model, history = train_detr_standalone(
        epochs=300,
        batch_size=16,
        lr=1e-4,
        min_people=2,
        max_people=5,
        noise_std=0.05,
        embedding_dim=128,
        eval_every=10,
        device=device,
    )