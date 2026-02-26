"""
DETR Diagnostic: Where exactly is the bottleneck?

This script trains the DETR decoder normally, then runs a battery of 
diagnostic evaluations to isolate the failure mode:

1. Current method: count_head + existence ranking → PGA
2. Oracle count + existence ranking → PGA (isolates counting vs ranking)
3. Oracle count + assignment confidence ranking → PGA (tests alt ranking)
4. Oracle count + BEST possible query subset → PGA (ceiling for assignment)
5. Oracle count + oracle query selection → PGA (true ceiling)

If (4) >> (1): the problem is counting + query selection, not assignment
If (4) ≈ (1): the assignment itself is broken regardless of selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from detr_decoder import DETRDecoder, DETRConfig
from losses import PoseGroupingLoss, LossConfig


# ══════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA (same as test_detr_standalone.py)
# ══════════════════════════════════════════════════════════════════════════

def generate_batch(
    batch_size: int,
    min_people: int,
    max_people: int,
    embedding_dim: int,
    noise_std: float,
    device: str,
) -> List[Dict]:
    batch = []
    for _ in range(batch_size):
        num_people = torch.randint(min_people, max_people + 1, (1,)).item()
        num_joint_types = 17
        
        person_centers = torch.randn(num_people, embedding_dim, device=device)
        person_centers = F.normalize(person_centers, dim=1) * 2.0
        
        all_embeddings = []
        all_types = []
        all_labels = []
        
        for person_id in range(num_people):
            for jtype in range(num_joint_types):
                emb = person_centers[person_id] + torch.randn(embedding_dim, device=device) * noise_std
                all_embeddings.append(emb)
                all_types.append(jtype)
                all_labels.append(person_id)
        
        batch.append({
            'embeddings': torch.stack(all_embeddings),
            'joint_types': torch.tensor(all_types, device=device),
            'person_labels': torch.tensor(all_labels, device=device),
            'num_people': num_people,
        })
    return batch


# ══════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC EVALUATION
# ══════════════════════════════════════════════════════════════════════════

def compute_pga_for_query_subset(
    outputs: Dict,
    active_indices: torch.Tensor,
    sample: Dict,
) -> Tuple[float, int, int]:
    """
    Given DETR outputs and a set of active query indices,
    do per-type Hungarian assignment and compute PGA.
    
    Returns: (pga, correct_count, total_count)
    """
    from scipy.optimize import linear_sum_assignment
    
    M = outputs['existence_logits'].shape[0]
    device = outputs['existence_logits'].device
    assignments = torch.full((M, 17), -1, dtype=torch.long, device=device)
    
    if len(active_indices) == 0:
        return 0.0, 0, 0
    
    for joint_type, (scores, indices) in enumerate(
        zip(outputs['assignment_scores'], outputs['joint_indices_per_type'])
    ):
        if len(indices) == 0:
            continue
        
        active_scores = scores[active_indices]
        cost_matrix = -active_scores.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            query_idx = active_indices[r]
            global_joint_idx = indices[c]
            assignments[query_idx, joint_type] = global_joint_idx
    
    # Compute PGA
    total_correct = 0
    total_assigned = 0
    
    for query_idx in active_indices:
        assigned = assignments[query_idx]
        valid = assigned >= 0
        if valid.any():
            valid_indices = assigned[valid]
            gt_labels = sample['person_labels'][valid_indices]
            if len(gt_labels) > 0:
                majority = gt_labels.mode().values.item()
                total_correct += (gt_labels == majority).sum().item()
                total_assigned += len(gt_labels)
    
    pga = total_correct / max(total_assigned, 1)
    return pga, total_correct, total_assigned


def query_assignment_confidence(outputs: Dict) -> torch.Tensor:
    """
    Rank queries by how peaked their assignment distributions are.
    A query that's found a person has high-confidence (peaked) assignments.
    An unspecialized query has diffuse assignments.
    
    Returns: [M] confidence scores (higher = more confident)
    """
    M = outputs['existence_logits'].shape[0]
    device = outputs['existence_logits'].device
    confidences = torch.zeros(M, device=device)
    num_types_with_joints = 0
    
    for scores, indices in zip(outputs['assignment_scores'], outputs['joint_indices_per_type']):
        if len(indices) == 0:
            continue
        
        # Softmax over joints for each query: [M, K]
        probs = F.softmax(scores, dim=1)
        # Max probability = peakedness: [M]
        max_probs = probs.max(dim=1).values
        confidences += max_probs
        num_types_with_joints += 1
    
    if num_types_with_joints > 0:
        confidences /= num_types_with_joints
    
    return confidences


@torch.no_grad()
def run_diagnostics(
    model: DETRDecoder,
    samples: List[Dict],
) -> Dict[str, Dict]:
    """
    Run all diagnostic evaluation modes on eval samples.
    """
    model.eval()
    
    results = {
        'count_head': {'correct': 0, 'total': 0, 'perfect_count': 0, 'n': 0},
        'oracle_count_existence_rank': {'correct': 0, 'total': 0, 'perfect_count': 0, 'n': 0},
        'oracle_count_confidence_rank': {'correct': 0, 'total': 0, 'perfect_count': 0, 'n': 0},
        'oracle_count_best_subset': {'correct': 0, 'total': 0, 'perfect_count': 0, 'n': 0},
        'oracle_everything': {'correct': 0, 'total': 0, 'perfect_count': 0, 'n': 0},
        'n_over_17': {'correct': 0, 'total': 0, 'perfect_count': 0, 'n': 0},
    }
    
    # Track existence separation
    matched_probs = []
    unmatched_probs = []
    matched_confs = []
    unmatched_confs = []
    
    for sample in samples:
        outputs = model.forward(sample['embeddings'], sample['joint_types'])
        num_gt = sample['num_people']
        M = outputs['existence_logits'].shape[0]
        
        existence_probs = torch.sigmoid(outputs['existence_logits'])
        confidences = query_assignment_confidence(outputs)
        count_pred = outputs['count_pred']
        
        # ── Method 1: Current (count_head + existence rank) ──────────
        n_pred = int(round(count_pred.item()))
        n_pred = max(0, min(n_pred, M))
        if n_pred > 0:
            _, top_idx = existence_probs.topk(n_pred)
        else:
            top_idx = torch.tensor([], dtype=torch.long, device=existence_probs.device)
        pga, correct, total = compute_pga_for_query_subset(outputs, top_idx, sample)
        results['count_head']['correct'] += correct
        results['count_head']['total'] += total
        results['count_head']['perfect_count'] += int(n_pred == num_gt)
        results['count_head']['n'] += 1
        
        # ── Method 2: Oracle count + existence ranking ───────────────
        _, top_idx = existence_probs.topk(num_gt)
        pga, correct, total = compute_pga_for_query_subset(outputs, top_idx, sample)
        results['oracle_count_existence_rank']['correct'] += correct
        results['oracle_count_existence_rank']['total'] += total
        results['oracle_count_existence_rank']['perfect_count'] += 1  # always perfect
        results['oracle_count_existence_rank']['n'] += 1
        
        # ── Method 3: Oracle count + confidence ranking ──────────────
        _, top_idx = confidences.topk(num_gt)
        pga, correct, total = compute_pga_for_query_subset(outputs, top_idx, sample)
        results['oracle_count_confidence_rank']['correct'] += correct
        results['oracle_count_confidence_rank']['total'] += total
        results['oracle_count_confidence_rank']['perfect_count'] += 1
        results['oracle_count_confidence_rank']['n'] += 1
        
        # ── Method 4: Oracle count + BEST possible subset ────────────
        # Try all C(M, num_gt) combinations, keep the best PGA
        best_pga = 0.0
        best_correct = 0
        best_total = 0
        for combo in combinations(range(M), num_gt):
            idx = torch.tensor(combo, dtype=torch.long, device=existence_probs.device)
            p, c, t = compute_pga_for_query_subset(outputs, idx, sample)
            if p > best_pga:
                best_pga = p
                best_correct = c
                best_total = t
        results['oracle_count_best_subset']['correct'] += best_correct
        results['oracle_count_best_subset']['total'] += best_total
        results['oracle_count_best_subset']['perfect_count'] += 1
        results['oracle_count_best_subset']['n'] += 1
        
        # ── Method 5: Oracle everything (all M queries active) ───────
        # Use ALL queries — the Hungarian matching handles it
        all_idx = torch.arange(M, device=existence_probs.device)
        pga, correct, total = compute_pga_for_query_subset(outputs, all_idx, sample)
        results['oracle_everything']['correct'] += correct
        results['oracle_everything']['total'] += total
        results['oracle_everything']['n'] += 1
        
        # ── Method 6: N/17 count + confidence ranking ────────────────
        n_from_joints = sample['embeddings'].size(0) // 17
        n_from_joints = max(0, min(n_from_joints, M))
        _, top_idx = confidences.topk(n_from_joints)
        pga, correct, total = compute_pga_for_query_subset(outputs, top_idx, sample)
        results['n_over_17']['correct'] += correct
        results['n_over_17']['total'] += total
        results['n_over_17']['perfect_count'] += int(n_from_joints == num_gt)
        results['n_over_17']['n'] += 1
        
        # ── Existence/confidence separation tracking ─────────────────
        sorted_exist, sorted_exist_idx = existence_probs.sort(descending=True)
        sorted_conf, sorted_conf_idx = confidences.sort(descending=True)
        
        matched_probs.extend(sorted_exist[:num_gt].cpu().tolist())
        unmatched_probs.extend(sorted_exist[num_gt:].cpu().tolist())
        matched_confs.extend(sorted_conf[:num_gt].cpu().tolist())
        unmatched_confs.extend(sorted_conf[num_gt:].cpu().tolist())
    
    # ── Compile results ──────────────────────────────────────────────
    summary = {}
    for method, data in results.items():
        pga = data['correct'] / max(data['total'], 1)
        pct = data.get('perfect_count', 0) / max(data['n'], 1)
        summary[method] = {'pga': pga, 'perfect_count_rate': pct}
    
    summary['separation'] = {
        'exist_matched_mean': np.mean(matched_probs) if matched_probs else 0,
        'exist_unmatched_mean': np.mean(unmatched_probs) if unmatched_probs else 0,
        'exist_gap': np.mean(matched_probs) - np.mean(unmatched_probs) if matched_probs and unmatched_probs else 0,
        'conf_matched_mean': np.mean(matched_confs) if matched_confs else 0,
        'conf_unmatched_mean': np.mean(unmatched_confs) if unmatched_confs else 0,
        'conf_gap': np.mean(matched_confs) - np.mean(unmatched_confs) if matched_confs and unmatched_confs else 0,
    }
    
    model.train()
    return summary


# ══════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train_and_diagnose(
    epochs: int = 300,
    batch_size: int = 16,
    lr: float = 1e-4,
    query_lr_multiplier: float = 10.0,
    min_people: int = 2,
    max_people: int = 5,
    noise_std: float = 0.05,
    embedding_dim: int = 128,
    eval_every: int = 10,
    device: str = "cpu",
    output_dir: Path = Path("detr_test_outputs"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = DETRConfig(
        embedding_dim=embedding_dim,
        max_people=max_people + 2,  # 7 queries for 2-5 people
        num_decoder_layers=3,
        num_heads=8,
        ffn_dim=512,
        dropout=0.1,
        num_joint_types=17,
        num_null_tokens=8,
    )
    model = DETRDecoder(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {device}")
    print(f"DETR parameters: {num_params:,}")
    
    loss_config = LossConfig(
        lambda_existence=1.0,
        lambda_assignment=5.0,
        lambda_contrastive=0.0,
        lambda_count=2.0,
        lambda_auxiliary=1.0,
        contrastive_margin=0.5,
        label_smoothing=0.0,
    )
    criterion = PoseGroupingLoss(loss_config)
    
    optimizer = torch.optim.AdamW([
        {'params': model.get_non_query_params(), 'lr': lr},
        {'params': model.get_query_params(), 'lr': lr * query_lr_multiplier},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    eval_samples = generate_batch(100, min_people, max_people, embedding_dim, noise_std, device)
    
    print(f"\nTraining DETR decoder for {epochs} epochs")
    print(f"People range: {min_people}-{max_people}, noise_std: {noise_std}")
    print("=" * 90)
    
    # ── Training loop ────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        model.train()
        batch = generate_batch(batch_size, min_people, max_people, embedding_dim, noise_std, device)
        
        epoch_loss = 0.0
        for sample in batch:
            optimizer.zero_grad()
            
            outputs = model(sample['embeddings'], sample['joint_types'])
            outputs['embeddings'] = sample['embeddings']
            
            loss_dict = criterion(outputs, sample['person_labels'], sample['joint_types'], sample['num_people'])
            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss_dict['total_loss'].item()
        
        scheduler.step()
        epoch_loss /= batch_size
        
        if epoch % eval_every == 0 or epoch == 1:
            diag = run_diagnostics(model, eval_samples)
            
            print(
                f"Ep {epoch:>3d} | "
                f"Loss: {epoch_loss:.2f} | "
                f"Current: {diag['count_head']['pga']:.3f} | "
                f"Oracle+Exist: {diag['oracle_count_existence_rank']['pga']:.3f} | "
                f"Oracle+Conf: {diag['oracle_count_confidence_rank']['pga']:.3f} | "
                f"Oracle+Best: {diag['oracle_count_best_subset']['pga']:.3f} | "
                f"AllQueries: {diag['oracle_everything']['pga']:.3f} | "
                f"N/17+Conf: {diag['n_over_17']['pga']:.3f} | "
                f"ExGap: {diag['separation']['exist_gap']:.3f} "
                f"CfGap: {diag['separation']['conf_gap']:.3f}"
            )
    
    # ── Final detailed diagnostic ────────────────────────────────────
    print("\n" + "=" * 90)
    print("FINAL DIAGNOSTIC RESULTS")
    print("=" * 90)
    
    diag = run_diagnostics(model, eval_samples)
    
    print("\n  METHOD                              PGA     Perfect Count")
    print("  " + "-" * 60)
    
    methods = [
        ('count_head',                     'Current (count_head + exist rank)'),
        ('oracle_count_existence_rank',    'Oracle count + existence rank'),
        ('oracle_count_confidence_rank',   'Oracle count + confidence rank'),
        ('oracle_count_best_subset',       'Oracle count + BEST subset (ceiling)'),
        ('oracle_everything',              'ALL queries active'),
        ('n_over_17',                      'N/17 count + confidence rank'),
    ]
    
    for key, label in methods:
        d = diag[key]
        pct_str = f"{d['perfect_count_rate']:.0%}" if 'perfect_count_rate' in d else "N/A"
        print(f"  {label:<40s} {d['pga']:.4f}  {pct_str}")
    
    print(f"\n  EXISTENCE SEPARATION:")
    sep = diag['separation']
    print(f"    Existence prob - matched:   {sep['exist_matched_mean']:.4f}")
    print(f"    Existence prob - unmatched: {sep['exist_unmatched_mean']:.4f}")
    print(f"    Existence gap:              {sep['exist_gap']:.4f}")
    print(f"    Confidence - matched:       {sep['conf_matched_mean']:.4f}")
    print(f"    Confidence - unmatched:     {sep['conf_unmatched_mean']:.4f}")
    print(f"    Confidence gap:             {sep['conf_gap']:.4f}")
    
    # ── Interpretation ───────────────────────────────────────────────
    current_pga = diag['count_head']['pga']
    best_pga = diag['oracle_count_best_subset']['pga']
    oracle_exist = diag['oracle_count_existence_rank']['pga']
    oracle_conf = diag['oracle_count_confidence_rank']['pga']
    n17_pga = diag['n_over_17']['pga']
    
    print(f"\n  INTERPRETATION:")
    print(f"    Assignment ceiling (best possible): {best_pga:.4f}")
    print(f"    Current method achieves:            {current_pga:.4f} ({current_pga/best_pga:.0%} of ceiling)")
    
    if best_pga > 0.9:
        print(f"    → Assignment IS working well. Bottleneck is query selection.")
        if oracle_conf > oracle_exist:
            print(f"    → Confidence ranking ({oracle_conf:.4f}) > existence ranking ({oracle_exist:.4f})")
            print(f"    → Existence head is useless. Use confidence ranking instead.")
        if n17_pga > current_pga:
            print(f"    → N/17 count + confidence ({n17_pga:.4f}) > current ({current_pga:.4f})")
            print(f"    → Simple N/17 count BEATS the learned count head.")
    elif best_pga > 0.7:
        print(f"    → Assignment is partially working but has room to improve.")
        print(f"    → Both assignment quality AND query selection need work.")
    else:
        print(f"    → Assignment itself is broken. Query selection is not the bottleneck.")
    
    return model, diag


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_and_diagnose(
        epochs=300,
        device=device,
        eval_every=10,
    )