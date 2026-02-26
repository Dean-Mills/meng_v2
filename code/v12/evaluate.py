"""
Evaluate trained GAT + DETR model on test set with visualization.

Supports both:
- Full model (GAT + DETR): Uses learned assignment
- GAT-only model: Falls back to HDBSCAN clustering
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import os
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Literal

from gat import GATEmbedding, GATConfig
from detr_decoder import DETRConfig, PoseGroupingModel
from virtual_preprocess import VirtualKeypointPreprocessor
from virtual_dataloader import create_virtual_dataloader
from settings import settings


# COCO skeleton connections for visualization
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Torso
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]


@dataclass
class EvalConfig:
    # Model settings
    existence_threshold: float = 0.5
    use_count_head: bool = True
    
    # Visualization
    save_visualizations: bool = True
    max_visualizations: int = 50
    
    # Metrics
    compute_per_joint_metrics: bool = True


class Evaluator:
    """
    Evaluate trained GAT + DETR model on test set.
    
    Computes:
    - Pose Grouping Accuracy (PGA): % of joints assigned to correct person
    - NMI: Normalized Mutual Information (embedding quality)
    - Person Detection: Precision, Recall, F1
    - Per-joint accuracy breakdown
    """
    
    def __init__(
        self, 
        checkpoint_path: str,
        config: Optional[EvalConfig] = None,
        device: str = 'cuda'
    ):
        self.config = config if config is not None else EvalConfig()
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        
        # Create output directory
        self.output_dir = self.checkpoint_path.parent.parent / "evaluation"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model
        self._load_model()
        
        # Preprocessor
        self.preprocessor = VirtualKeypointPreprocessor(
            device=device,
            k_neighbors=8,
            image_size=512
        )
        
        print(f"Evaluator initialized")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Mode: {'Full (GAT + DETR)' if self.is_full_model else 'GAT-only'}")
    
    def _load_model(self):
        """Load model from checkpoint."""
        checkpoint = torch.load(
            self.checkpoint_path, 
            map_location=self.device, 
            weights_only=False
        )
        
        # Check if this is a full model or GAT-only
        state_dict = checkpoint['model_state_dict']
        self.is_full_model = any('detr' in key for key in state_dict.keys())
        
        if self.is_full_model:
            # Load full model (GAT + DETR)
            gat = GATEmbedding(GATConfig())
            self.model = PoseGroupingModel(gat, DETRConfig())
        else:
            # Load GAT-only model
            self.model = GATEmbedding(GATConfig())
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded model from epoch {self.epoch}")
    
    def predict_full_model(self, graph) -> Dict[str, Any]:
        """
        Get predictions from full GAT + DETR model.
        
        Returns:
            Dict with predicted_labels, existence_probs, num_predicted, embeddings
        """
        with torch.no_grad():
            predictions = self.model.predict(
                graph, 
                existence_threshold=self.config.existence_threshold,
                use_count_head=self.config.use_count_head
            )
            outputs = self.model(graph)
        
        # Convert assignments to per-joint labels
        # assignments: [M, 17] - for each person query, which joint index per type
        # We need to invert this to: [N] - for each joint, which person
        
        N = graph.x.size(0)
        predicted_labels = torch.full((N,), -1, dtype=torch.long, device=self.device)
        
        person_idx_counter = 0
        for query_idx in range(predictions['assignments'].size(0)):
            if not predictions['person_mask'][query_idx]:
                continue
            
            # This query represents a real person
            for joint_type in range(17):
                joint_idx = predictions['assignments'][query_idx, joint_type].item()
                if joint_idx != -1 and joint_idx < N:
                    predicted_labels[joint_idx] = person_idx_counter
            
            person_idx_counter += 1
        
        return {
            'predicted_labels': predicted_labels,
            'existence_probs': predictions['existence_probs'],
            'num_predicted': predictions['num_people'],
            'person_mask': predictions['person_mask'],
            'embeddings': outputs['embeddings']
        }
    
    def predict_gat_only(self, graph, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Get predictions from GAT-only model using k-means clustering.
        
        Args:
            graph: Input graph
            n_clusters: Number of clusters (if None, use ground truth count)
        
        Returns:
            Dict with predicted_labels, embeddings
        """
        with torch.no_grad():
            embeddings = self.model(graph)
        
        # Use ground truth number of people (oracle) or specified
        if n_clusters is None:
            n_clusters = graph.num_people
        
        # K-means clustering
        emb_np = embeddings.cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        predicted_labels = kmeans.fit_predict(emb_np)
        
        return {
            'predicted_labels': torch.tensor(predicted_labels, device=self.device),
            'num_predicted': n_clusters,
            'embeddings': embeddings
        }
    
    def compute_pga(
        self,
        predicted_labels: torch.Tensor,
        ground_truth_labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute Pose Grouping Accuracy with Hungarian matching.
        
        PGA = % of joints assigned to the correct person after optimal matching.
        """
        pred_np = predicted_labels.cpu().numpy()
        gt_np = ground_truth_labels.cpu().numpy()
        
        # Get unique labels (excluding -1 for unassigned)
        pred_people = set(pred_np) - {-1}
        gt_people = set(gt_np)
        
        if len(pred_people) == 0 or len(gt_people) == 0:
            return {'pga': 0.0, 'matched_pairs': []}
        
        # Build cost matrix [num_pred x num_gt]
        pred_list = sorted(pred_people)
        gt_list = sorted(gt_people)
        
        cost_matrix = np.zeros((len(pred_list), len(gt_list)))
        
        for i, pred_id in enumerate(pred_list):
            for j, gt_id in enumerate(gt_list):
                # Cost = negative overlap (we want to maximize overlap)
                pred_mask = (pred_np == pred_id)
                gt_mask = (gt_np == gt_id)
                overlap = np.sum(pred_mask & gt_mask)
                cost_matrix[i, j] = -overlap
        
        # Hungarian matching
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
        
        # Compute accuracy based on matched pairs
        correct = 0
        total = 0
        matched_pairs = []
        
        # Create mapping from predicted to GT
        pred_to_gt = {}
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            pred_id = pred_list[pred_idx]
            gt_id = gt_list[gt_idx]
            pred_to_gt[pred_id] = gt_id
            matched_pairs.append((pred_id, gt_id))
        
        # Count correct assignments
        for i, (pred, gt) in enumerate(zip(pred_np, gt_np)):
            if pred == -1:
                continue  # Skip unassigned joints
            total += 1
            if pred in pred_to_gt and pred_to_gt[pred] == gt:
                correct += 1
        
        pga = correct / total if total > 0 else 0.0
        
        return {
            'pga': pga,
            'correct': correct,
            'total': total,
            'matched_pairs': matched_pairs
        }
    
    def compute_per_joint_accuracy(
        self,
        predicted_labels: torch.Tensor,
        ground_truth_labels: torch.Tensor,
        joint_types: torch.Tensor,
        pred_to_gt_mapping: Dict[int, int]
    ) -> Dict[int, float]:
        """
        Compute accuracy per joint type.
        """
        pred_np = predicted_labels.cpu().numpy()
        gt_np = ground_truth_labels.cpu().numpy()
        joint_types_np = joint_types.cpu().numpy()
        
        per_joint_acc = {}
        
        for joint_type in range(17):
            mask = joint_types_np == joint_type
            if mask.sum() == 0:
                continue
            
            correct = 0
            total = mask.sum()
            
            for i in np.where(mask)[0]:
                pred = pred_np[i]
                gt = gt_np[i]
                
                if pred == -1:
                    continue
                
                if pred in pred_to_gt_mapping and pred_to_gt_mapping[pred] == gt:
                    correct += 1
            
            per_joint_acc[joint_type] = correct / total if total > 0 else 0.0
        
        return per_joint_acc
    
    def compute_detection_metrics(
        self,
        num_predicted: int,
        num_ground_truth: int
    ) -> Dict[str, float]:
        """
        Compute person detection precision, recall, F1.
        """
        tp = min(num_predicted, num_ground_truth)
        fp = max(0, num_predicted - num_ground_truth)
        fn = max(0, num_ground_truth - num_predicted)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def visualize_predictions(
        self,
        image: torch.Tensor,
        keypoints_list: List[torch.Tensor],
        predicted_labels: torch.Tensor,
        ground_truth_labels: torch.Tensor,
        joint_types: torch.Tensor,
        img_id: int,
        metrics: Dict[str, Any],
        save_path: Optional[Path] = None
    ):
        """
        Visualize predicted vs ground truth poses.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Convert image
        img_np = image.permute(1, 2, 0).cpu().numpy()
        if img_np.max() > 1:
            img_np = img_np / 255.0
        img_np = np.clip(img_np, 0, 1)
        
        pred_np = predicted_labels.cpu().numpy()
        gt_np = ground_truth_labels.cpu().numpy()
        
        # Get unique predicted people (excluding -1)
        unique_pred = sorted(set(pred_np) - {-1})
        unique_gt = sorted(set(gt_np))
        
        # Color maps
        colors_pred = plt.cm.tab10(np.linspace(0, 1, max(len(unique_pred), 1)))
        colors_gt = plt.cm.Set1(np.linspace(0, 1, max(len(unique_gt), 1)))
        
        pred_color_map = {-1: [0.5, 0.5, 0.5, 1.0]}  # Gray for unassigned
        for i, pred_id in enumerate(unique_pred):
            pred_color_map[pred_id] = colors_pred[i % len(colors_pred)]
        
        gt_color_map = {}
        for i, gt_id in enumerate(unique_gt):
            gt_color_map[gt_id] = colors_gt[i % len(colors_gt)]
        
        # Build joint index to (person_idx, joint_idx) mapping
        joint_idx = 0
        joint_to_coords = {}
        
        for person_idx, keypoints in enumerate(keypoints_list):
            for joint_type_idx in range(keypoints.shape[0]):
                x, y, visibility = keypoints[joint_type_idx]
                if visibility > 0:
                    joint_to_coords[joint_idx] = {
                        'x': x.item(),
                        'y': y.item(),
                        'person_idx': person_idx,
                        'joint_type': joint_type_idx
                    }
                    joint_idx += 1
        
        # --- Left plot: Predicted ---
        axes[0].imshow(img_np)
        pga_str = f"{metrics['pga']:.1%}" if 'pga' in metrics else "N/A"
        axes[0].set_title(f"Predicted (DETR)\nPGA: {pga_str}, Found: {len(unique_pred)} people")
        axes[0].axis('off')
        
        # Draw predicted
        for idx, info in joint_to_coords.items():
            pred_label = pred_np[idx]
            color = pred_color_map.get(pred_label, [0.5, 0.5, 0.5, 1.0])
            axes[0].scatter(info['x'], info['y'], c=[color], s=60,
                           edgecolors='black', linewidth=0.5, zorder=3)
        
        # Draw skeleton connections (predicted)
        self._draw_skeletons(axes[0], keypoints_list, pred_np, joint_to_coords, pred_color_map)
        
        # Legend for predicted
        legend_pred = [Patch(facecolor=pred_color_map.get(p, [0.5]*4), label=f'Person {p}') 
                      for p in unique_pred]
        if -1 in pred_np:
            legend_pred.append(Patch(facecolor=[0.5, 0.5, 0.5, 1.0], label='Unassigned'))
        axes[0].legend(handles=legend_pred, loc='upper right', fontsize=8)
        
        # --- Right plot: Ground Truth ---
        axes[1].imshow(img_np)
        axes[1].set_title(f"Ground Truth\n{len(keypoints_list)} people")
        axes[1].axis('off')
        
        # Draw ground truth
        for idx, info in joint_to_coords.items():
            gt_label = gt_np[idx]
            color = gt_color_map.get(gt_label, [0.5, 0.5, 0.5, 1.0])
            axes[1].scatter(info['x'], info['y'], c=[color], s=60,
                           edgecolors='black', linewidth=0.5, zorder=3)
        
        # Draw skeleton connections (ground truth)
        self._draw_skeletons(axes[1], keypoints_list, gt_np, joint_to_coords, gt_color_map)
        
        # Legend for ground truth
        legend_gt = [Patch(facecolor=gt_color_map[g], label=f'Person {g}') 
                    for g in unique_gt]
        axes[1].legend(handles=legend_gt, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def _draw_skeletons(self, ax, keypoints_list, labels_np, joint_to_coords, color_map):
        """Helper to draw skeleton connections."""
        # Build reverse mapping: (person_idx, joint_type) -> joint_idx
        coord_to_joint_idx = {}
        for joint_idx, info in joint_to_coords.items():
            key = (info['person_idx'], info['joint_type'])
            coord_to_joint_idx[key] = joint_idx
        
        for person_idx, keypoints in enumerate(keypoints_list):
            for (j1, j2) in COCO_SKELETON:
                if keypoints[j1, 2] > 0 and keypoints[j2, 2] > 0:
                    # Get joint indices
                    idx1 = coord_to_joint_idx.get((person_idx, j1))
                    idx2 = coord_to_joint_idx.get((person_idx, j2))
                    
                    if idx1 is None or idx2 is None:
                        continue
                    
                    # Use color of first joint
                    label = labels_np[idx1]
                    color = color_map.get(label, [0.5, 0.5, 0.5, 1.0])
                    
                    ax.plot(
                        [keypoints[j1, 0].item(), keypoints[j2, 0].item()],
                        [keypoints[j1, 1].item(), keypoints[j2, 1].item()],
                        color=color, linewidth=2, alpha=0.7, zorder=2
                    )
    
    def evaluate_single(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        visualize: bool = True,
        vis_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a single image."""
        image = batch['image'][batch_idx]
        keypoints_list = batch['keypoints'][batch_idx]
        distances = batch['distances'][batch_idx]
        img_id = batch['img_id'][batch_idx]
        
        # Create graph
        graph = self.preprocessor.create_graph(keypoints_list, distances)
        
        if graph is None:
            return None
        
        graph = graph.to(self.device)
        
        # Get predictions
        if self.is_full_model:
            pred_output = self.predict_full_model(graph)
        else:
            pred_output = self.predict_gat_only(graph)
        
        predicted_labels = pred_output['predicted_labels']
        ground_truth_labels = graph.person_labels
        num_predicted = pred_output['num_predicted']
        num_gt = graph.num_people
        
        # Compute metrics
        pga_results = self.compute_pga(predicted_labels, ground_truth_labels)
        detection_metrics = self.compute_detection_metrics(num_predicted, num_gt)
        
        # NMI on embeddings
        emb_np = pred_output['embeddings'].cpu().numpy()
        gt_np = ground_truth_labels.cpu().numpy()
        
        try:
            kmeans = KMeans(n_clusters=num_gt, random_state=42, n_init=10)
            pred_clusters = kmeans.fit_predict(emb_np)
            nmi = normalized_mutual_info_score(gt_np, pred_clusters)
            ari = adjusted_rand_score(gt_np, pred_clusters)
        except:
            nmi, ari = 0.0, 0.0
        
        # Per-joint accuracy
        per_joint_acc = {}
        if self.config.compute_per_joint_metrics and pga_results['matched_pairs']:
            pred_to_gt = {p: g for p, g in pga_results['matched_pairs']}
            per_joint_acc = self.compute_per_joint_accuracy(
                predicted_labels, ground_truth_labels, graph.joint_types, pred_to_gt
            )
        
        results = {
            'img_id': img_id,
            'num_predicted': num_predicted,
            'num_gt': num_gt,
            'pga': pga_results['pga'],
            'nmi': nmi,
            'ari': ari,
            'precision': detection_metrics['precision'],
            'recall': detection_metrics['recall'],
            'f1': detection_metrics['f1'],
            'per_joint_acc': per_joint_acc,
        }
        
        # Visualize
        if visualize and vis_count < self.config.max_visualizations:
            save_path = self.output_dir / f"eval_{img_id}.png"
            self.visualize_predictions(
                image, keypoints_list,
                predicted_labels, ground_truth_labels,
                graph.joint_types, img_id,
                results, save_path
            )
        
        return results
    
    def evaluate_dataset(
        self,
        split: str = "four_persons",
        max_images: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate on entire dataset split.
        """
        print(f"\nEvaluating on '{split}' split...")
        print("-" * 60)
        
        dataloader = create_virtual_dataloader(split=split, shuffle=False, num_workers=0)
        
        all_results = []
        image_count = 0
        vis_count = 0
        
        for batch in dataloader:
            batch_size = batch['image'].shape[0]
            
            for i in range(batch_size):
                if max_images and image_count >= max_images:
                    break
                
                result = self.evaluate_single(
                    batch, i, 
                    visualize=self.config.save_visualizations,
                    vis_count=vis_count
                )
                
                if result:
                    all_results.append(result)
                    vis_count += 1
                    
                    print(
                        f"Image {result['img_id']:4d} | "
                        f"Pred: {result['num_predicted']} GT: {result['num_gt']} | "
                        f"PGA: {result['pga']:.3f} | "
                        f"NMI: {result['nmi']:.3f} | "
                        f"F1: {result['f1']:.3f}"
                    )
                
                image_count += 1
            
            if max_images and image_count >= max_images:
                break
        
        # Summary
        summary = self._compute_summary(all_results, split)
        self._print_summary(summary)
        self._save_summary(summary)
        
        return summary
    
    def _compute_summary(self, results: List[Dict], split: str) -> Dict[str, Any]:
        """Compute summary statistics."""
        if not results:
            return {}
        
        summary = {
            'split': split,
            'n_images': len(results),
            'avg_pga': np.mean([r['pga'] for r in results]),
            'std_pga': np.std([r['pga'] for r in results]),
            'avg_nmi': np.mean([r['nmi'] for r in results]),
            'avg_ari': np.mean([r['ari'] for r in results]),
            'avg_precision': np.mean([r['precision'] for r in results]),
            'avg_recall': np.mean([r['recall'] for r in results]),
            'avg_f1': np.mean([r['f1'] for r in results]),
            'perfect_detection': sum(1 for r in results if r['num_predicted'] == r['num_gt']) / len(results),
        }
        
        # Per-joint accuracy
        per_joint_accs = {j: [] for j in range(17)}
        for r in results:
            for j, acc in r.get('per_joint_acc', {}).items():
                per_joint_accs[j].append(acc)
        
        summary['per_joint_avg'] = {
            COCO_KEYPOINT_NAMES[j]: np.mean(accs) if accs else 0.0
            for j, accs in per_joint_accs.items()
        }
        
        summary['individual_results'] = results
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print summary to console."""
        if not summary:
            print("No results to summarize.")
            return
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Split: {summary['split']}")
        print(f"Images evaluated: {summary['n_images']}")
        print("-" * 60)
        print(f"Pose Grouping Accuracy (PGA): {summary['avg_pga']:.4f} ± {summary['std_pga']:.4f}")
        print(f"NMI (embedding quality):      {summary['avg_nmi']:.4f}")
        print(f"ARI:                          {summary['avg_ari']:.4f}")
        print("-" * 60)
        print("Person Detection:")
        print(f"  Precision:                  {summary['avg_precision']:.4f}")
        print(f"  Recall:                     {summary['avg_recall']:.4f}")
        print(f"  F1:                         {summary['avg_f1']:.4f}")
        print(f"  Perfect Count Rate:         {summary['perfect_detection']:.2%}")
        print("-" * 60)
        
        if summary.get('per_joint_avg'):
            print("Per-Joint Accuracy:")
            for joint_name, acc in sorted(summary['per_joint_avg'].items(), 
                                         key=lambda x: -x[1]):
                print(f"  {joint_name:20s}: {acc:.4f}")
        
        print("=" * 60)
        print(f"Results saved to: {self.output_dir}")
    
    def _save_summary(self, summary: Dict[str, Any]):
        """Save summary to file."""
        import json
        
        # Remove non-serializable items
        summary_save = {k: v for k, v in summary.items() if k != 'individual_results'}
        
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_save, f, indent=2)
        
        # Save detailed results
        results_path = self.output_dir / "detailed_results.json"
        with open(results_path, 'w') as f:
            json.dump(summary.get('individual_results', []), f, indent=2)
    
    def plot_per_joint_accuracy(self, summary: Dict[str, Any]):
        """Create bar chart of per-joint accuracy."""
        if not summary.get('per_joint_avg'):
            return
        
        joint_names = list(summary['per_joint_avg'].keys())
        accuracies = list(summary['per_joint_avg'].values())
        
        # Sort by accuracy
        sorted_pairs = sorted(zip(joint_names, accuracies), key=lambda x: x[1])
        joint_names, accuracies = zip(*sorted_pairs)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.RdYlGn(np.array(accuracies))
        bars = ax.barh(joint_names, accuracies, color=colors)
        
        ax.set_xlabel('Accuracy')
        ax.set_title('Per-Joint Grouping Accuracy')
        ax.set_xlim([0, 1])
        ax.axvline(x=np.mean(accuracies), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(accuracies):.3f}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "per_joint_accuracy.png", dpi=150)
        plt.close()


def main():
    """Main evaluation script."""
    
    # ===========================================
    # CONFIGURATION
    # ===========================================
    
    checkpoint_path = "/home/dean/projects/mills_ds/outputs/pipeline/9ddaeac7/checkpoints/model_final.pt"
    split = "mixed_test"
    max_images = None  # Set to a number to limit
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ===========================================
    
    config = EvalConfig(
        existence_threshold=0.3,
        save_visualizations=True,
        max_visualizations=20,
        compute_per_joint_metrics=True
    )
    
    evaluator = Evaluator(checkpoint_path, config=config, device=device)
    summary = evaluator.evaluate_dataset(split=split, max_images=max_images)
    
    # Plot per-joint accuracy
    if summary:
        evaluator.plot_per_joint_accuracy(summary)


if __name__ == '__main__':
    main()