"""
DETR Decoder Standalone Test (with Count Head)
================================================
Tests the DETR decoder in isolation by feeding it synthetic "ideal" GAT embeddings.

What this proves:
- The DETR can learn to count people (count head)
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

# ── Import DETR and Loss from your codebase ──────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

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
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Generate one synthetic sample: fake GAT embeddings with known grouping.
    """
    person_identities = torch.randn(num_people, embedding_dim, device=device)
    person_identities = F.normalize(person_identities, p=2, dim=-1)
    
    all_embeddings = []
    all_joint_types = []
    all_person_labels = []
    
    for person_idx in range(num_people):
        for joint_type in range(num_joint_types):
            embedding = person_identities[person_idx] + torch.randn(embedding_dim, device=device) * noise_std
            embedding = F.normalize(embedding, p=2, dim=-1)
            
            all_embeddings.append(embedding)
            all_joint_types.append(joint_type)
            all_person_labels.append(person_idx)
    
    return {
        'embeddings': torch.stack(all_embeddings),
        'joint_types': torch.tensor(all_joint_types, device=device),
        'person_labels': torch.tensor(all_person_labels, device=device),
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
    use_count_head: bool = True,
    existence_threshold: float = 0.5
) -> Dict[str, float]:
    """Compute existence and assignment metrics on a set of samples."""
    model.eval()
    
    total_correct_assignments = 0
    total_assignments = 0
    total_perfect_count = 0
    total_samples = 0
    
    pred_counts = []
    gt_counts = []
    count_preds_raw = []
    
    for sample in samples:
        result = model.predict(
            sample['embeddings'],
            sample['joint_types'],
            existence_threshold=existence_threshold,
            use_count_head=use_count_head
        )
        
        num_gt = sample['num_people']
        num_pred = result['num_people']
        
        gt_counts.append(num_gt)
        pred_counts.append(num_pred)
        count_preds_raw.append(result['count_pred'])
        
        if num_pred == num_gt:
            total_perfect_count += 1
        total_samples += 1
        
        # Assignment accuracy
        if num_pred > 0 and num_gt > 0:
            assignments = result['assignments']
            person_mask = result['person_mask']
            
            for query_idx in torch.where(person_mask)[0]:
                assigned_joints = assignments[query_idx]
                valid = assigned_joints >= 0
                
                if valid.any():
                    valid_indices = assigned_joints[valid]
                    gt_labels = sample['person_labels'][valid_indices]
                    
                    if len(gt_labels) > 0:
                        majority_label = gt_labels.mode().values.item()
                        correct = (gt_labels == majority_label).sum().item()
                        total_correct_assignments += correct
                        total_assignments += len(gt_labels)
    
    model.train()
    
    pga = total_correct_assignments / max(total_assignments, 1)
    perfect_count_rate = total_perfect_count / max(total_samples, 1)
    
    gt_arr = np.array(gt_counts)
    pred_arr = np.array(pred_counts)
    mean_count_error = np.mean(np.abs(gt_arr - pred_arr))
    
    return {
        'pga': pga,
        'perfect_count_rate': perfect_count_rate,
        'mean_count_error': mean_count_error,
        'avg_gt_count': gt_arr.mean(),
        'avg_pred_count': pred_arr.mean(),
        'avg_count_pred_raw': np.mean(count_preds_raw),
    }


# ══════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════

def train_detr_standalone(
    epochs: int = 300,
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
        max_people=max_people + 2,
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
        lambda_contrastive=0.0,  # No contrastive - testing DETR only
        lambda_count=2.0,        # NEW: count head loss
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
        'count_loss': [],
        'count_pred': [],
        'pga': [],
        'perfect_count_rate': [],
        'mean_count_error': [],
        'avg_pred_count': [],
    }
    
    # ── Training ──────────────────────────────────────────────────────
    print(f"\nTraining DETR decoder for {epochs} epochs")
    print(f"People range: {min_people}-{max_people}, noise_std: {noise_std}")
    print(f"Batch size: {batch_size}, LR: {lr}")
    print(f"Count head: ENABLED (lambda_count={loss_config.lambda_count})")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        model.train()
        
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
            'count': 0.0,
            'count_pred': 0.0,
        }
        
        for sample in batch:
            optimizer.zero_grad()
            
            outputs = model(sample['embeddings'], sample['joint_types'])
            outputs['embeddings'] = sample['embeddings']
            
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
            epoch_losses['count'] += loss_dict['count_loss'].item()
            epoch_losses['count_pred'] += loss_dict['count_pred']
        
        scheduler.step()
        
        for k in epoch_losses:
            epoch_losses[k] /= batch_size
        
        # ── Evaluate ──────────────────────────────────────────────────
        if epoch % eval_every == 0 or epoch == 1:
            metrics = compute_metrics(model, eval_samples, use_count_head=True)
            
            history['epoch'].append(epoch)
            history['total_loss'].append(epoch_losses['total'])
            history['existence_loss'].append(epoch_losses['existence'])
            history['assignment_loss'].append(epoch_losses['assignment'])
            history['count_loss'].append(epoch_losses['count'])
            history['count_pred'].append(epoch_losses['count_pred'])
            history['pga'].append(metrics['pga'])
            history['perfect_count_rate'].append(metrics['perfect_count_rate'])
            history['mean_count_error'].append(metrics['mean_count_error'])
            history['avg_pred_count'].append(metrics['avg_pred_count'])
            
            print(
                f"Epoch {epoch:>4d} | "
                f"Loss: {epoch_losses['total']:.4f} "
                f"(exist: {epoch_losses['existence']:.4f}, "
                f"assign: {epoch_losses['assignment']:.4f}, "
                f"count: {epoch_losses['count']:.4f}) | "
                f"PGA: {metrics['pga']:.4f} | "
                f"Count: {metrics['perfect_count_rate']:.2%} perfect, "
                f"err: {metrics['mean_count_error']:.2f} | "
                f"Pred/GT: {metrics['avg_pred_count']:.1f}/{metrics['avg_gt_count']:.1f} | "
                f"Raw: {epoch_losses['count_pred']:.2f}"
            )
    
    # ── Final evaluation ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL EVALUATION (count head)")
    print("=" * 70)
    
    final_metrics = compute_metrics(model, eval_samples, use_count_head=True)
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Also evaluate with threshold for comparison
    print("\n" + "=" * 70)
    print("COMPARISON: threshold-based (no count head)")
    print("=" * 70)
    
    for thresh in [0.5, 0.55, 0.6]:
        thresh_metrics = compute_metrics(model, eval_samples, use_count_head=False, existence_threshold=thresh)
        print(f"  threshold={thresh}: perfect_count={thresh_metrics['perfect_count_rate']:.2%}, "
              f"pga={thresh_metrics['pga']:.4f}, err={thresh_metrics['mean_count_error']:.2f}")
    
    # ── Example predictions ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)
    
    model.eval()
    for i in range(min(5, len(eval_samples))):
        sample = eval_samples[i]
        result = model.predict(sample['embeddings'], sample['joint_types'], use_count_head=True)
        
        print(f"\n  Sample {i}: GT={sample['num_people']} people, "
              f"Predicted={result['num_people']} people "
              f"(count_head={result['count_pred']:.2f})")
        
        probs = result['existence_probs']
        top_probs = probs.sort(descending=True).values[:sample['num_people'] + 2]
        print(f"  Top existence probs: {[f'{p:.3f}' for p in top_probs.tolist()]}")
        
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
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle("DETR Decoder Standalone Test (with Count Head)", fontsize=16, fontweight='bold')
    
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
    ax.plot(epochs, history['count_loss'], 'orange', label='Count', linewidth=1.5)
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
    
    # 4. Count loss specifically
    ax = axes[0, 3]
    ax.plot(epochs, history['count_loss'], 'orange', linewidth=1.5)
    ax.axhline(y=0, color='g', linestyle='--', alpha=0.5)
    ax.set_title('Count Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    
    # 5. Perfect count rate
    ax = axes[1, 0]
    ax.plot(epochs, history['perfect_count_rate'], 'm-', linewidth=1.5)
    ax.set_title('Perfect Count Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rate')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # 6. Mean count error
    ax = axes[1, 1]
    ax.plot(epochs, history['mean_count_error'], 'r-', linewidth=1.5)
    ax.axhline(y=0, color='g', linestyle='--', alpha=0.5)
    ax.set_title('Mean Count Error')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|pred - gt|')
    ax.grid(True, alpha=0.3)
    
    # 7. Avg predicted count vs GT
    ax = axes[1, 2]
    ax.plot(epochs, history['avg_pred_count'], 'b-', label='Predicted', linewidth=1.5)
    ax.axhline(y=3.5, color='g', linestyle='--', alpha=0.5, label='~GT mean (3.5)')
    ax.set_title('Avg Predicted Count')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 8. Raw count head prediction
    ax = axes[1, 3]
    ax.plot(epochs, history['count_pred'], 'purple', linewidth=1.5, label='Count head output')
    ax.axhline(y=3.5, color='g', linestyle='--', alpha=0.5, label='~GT mean (3.5)')
    ax.set_title('Count Head Raw Output')
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