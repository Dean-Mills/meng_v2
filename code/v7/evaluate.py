# evaluate.py
"""
Evaluate trained GAT model on test set with HDBSCAN clustering and skeleton visualization.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from hdbscan import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import os

from gat import TinyGAT
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


class Evaluator:
    """Evaluate trained model on test set with clustering and visualization."""
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize evaluator with a trained model checkpoint.
        
        Args:
            checkpoint_path: Path to the saved model checkpoint
            device: Device to use for inference
        """
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        
        # Create output directory for evaluation results
        self.output_dir = self.checkpoint_path.parent.parent / "evaluation"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load model
        self.model = TinyGAT().to(device)
        self.preprocessor = VirtualKeypointPreprocessor(device=device)
        
        # Load checkpoint
        self._load_checkpoint()
        
        # Set model to evaluation mode
        self.model.eval()
        
        print(f"Evaluator initialized")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Output directory: {self.output_dir}")
    
    def _load_checkpoint(self):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device,weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded model from epoch {self.epoch}")
    
    def cluster_embeddings(self, embeddings, min_cluster_size=5, min_samples=3):
        """
        Cluster embeddings using HDBSCAN.
        
        Args:
            embeddings: numpy array of shape [N, embedding_dim]
            min_cluster_size: Minimum size of clusters
            min_samples: Number of samples in neighborhood for core points
            
        Returns:
            cluster_labels: Array of cluster assignments (-1 for noise)
        """
        # Use cosine distance since we trained with cosine similarity
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',  # On L2-normalized vectors, euclidean ~ cosine
            cluster_selection_method='eom'
        )
        
        cluster_labels = clusterer.fit_predict(embeddings)
        
        return cluster_labels
    
    def visualize_skeleton(self, image, keypoints_list, cluster_labels, ground_truth_labels,
                          img_id, save_path=None):
        """
        Visualize skeletons colored by predicted cluster.
        
        Args:
            image: Image tensor [C, H, W]
            keypoints_list: List of keypoint tensors [17, 3] for each person
            cluster_labels: Predicted cluster for each keypoint (flattened)
            ground_truth_labels: Ground truth person labels
            img_id: Image identifier
            save_path: Path to save the visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Convert image for display
        img_np = image.permute(1, 2, 0).cpu().numpy()
        if img_np.max() > 1:
            img_np = img_np / 255.0
        img_np = np.clip(img_np, 0, 1)
        
        # Get unique clusters (excluding noise label -1)
        unique_clusters = sorted(set(cluster_labels) - {-1})
        n_clusters = len(unique_clusters)
        
        # Color maps
        colors_predicted = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
        colors_gt = plt.cm.Set1(np.linspace(0, 1, len(keypoints_list)))
        noise_color = [0.5, 0.5, 0.5, 1.0]  # Gray for noise
        
        # Create cluster to color mapping
        cluster_color_map = {-1: noise_color}
        for i, cluster_id in enumerate(unique_clusters):
            cluster_color_map[cluster_id] = colors_predicted[i % len(colors_predicted)]
        
        # --- Left plot: Predicted clusters ---
        axes[0].imshow(img_np)
        axes[0].set_title(f'Predicted Clusters (HDBSCAN)\n{n_clusters} clusters found')
        axes[0].axis('off')
        
        # Build a mapping from (person_idx, joint_idx) to cluster label
        keypoint_idx = 0
        keypoint_to_cluster = {}
        
        for person_idx, keypoints in enumerate(keypoints_list):
            for joint_idx in range(keypoints.shape[0]):
                x, y, visibility = keypoints[joint_idx]
                if visibility > 0:
                    keypoint_to_cluster[(person_idx, joint_idx)] = cluster_labels[keypoint_idx]
                    keypoint_idx += 1
        
        # Draw predicted skeletons
        for person_idx, keypoints in enumerate(keypoints_list):
            # Draw keypoints
            for joint_idx in range(keypoints.shape[0]):
                x, y, visibility = keypoints[joint_idx]
                if visibility > 0:
                    cluster_id = keypoint_to_cluster.get((person_idx, joint_idx), -1)
                    color = cluster_color_map.get(cluster_id, noise_color)
                    axes[0].scatter(x.item(), y.item(), c=[color], s=50, 
                                   edgecolors='black', linewidth=0.5, zorder=3)
            
            # Draw skeleton connections
            for (j1, j2) in COCO_SKELETON:
                if keypoints[j1, 2] > 0 and keypoints[j2, 2] > 0:
                    # Use the cluster of the first joint for line color
                    cluster_id = keypoint_to_cluster.get((person_idx, j1), -1)
                    color = cluster_color_map.get(cluster_id, noise_color)
                    axes[0].plot(
                        [keypoints[j1, 0].item(), keypoints[j2, 0].item()],
                        [keypoints[j1, 1].item(), keypoints[j2, 1].item()],
                        color=color, linewidth=2, alpha=0.7, zorder=2
                    )
        
        # Add legend for predicted clusters
        legend_elements = []
        for cluster_id in sorted(cluster_color_map.keys()):
            if cluster_id == -1:
                label = 'Noise'
            else:
                label = f'Cluster {cluster_id}'
            legend_elements.append(Patch(facecolor=cluster_color_map[cluster_id], label=label))
        axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # --- Right plot: Ground truth ---
        axes[1].imshow(img_np)
        axes[1].set_title(f'Ground Truth\n{len(keypoints_list)} people')
        axes[1].axis('off')
        
        # Draw ground truth skeletons
        for person_idx, keypoints in enumerate(keypoints_list):
            color = colors_gt[person_idx % len(colors_gt)]
            
            # Draw keypoints
            for joint_idx in range(keypoints.shape[0]):
                x, y, visibility = keypoints[joint_idx]
                if visibility > 0:
                    axes[1].scatter(x.item(), y.item(), c=[color], s=50,
                                   edgecolors='black', linewidth=0.5, zorder=3)
            
            # Draw skeleton connections
            for (j1, j2) in COCO_SKELETON:
                if keypoints[j1, 2] > 0 and keypoints[j2, 2] > 0:
                    axes[1].plot(
                        [keypoints[j1, 0].item(), keypoints[j2, 0].item()],
                        [keypoints[j1, 1].item(), keypoints[j2, 1].item()],
                        color=color, linewidth=2, alpha=0.7, zorder=2
                    )
        
        # Add legend for ground truth
        gt_legend = [Patch(facecolor=colors_gt[i], label=f'Person {i}') 
                     for i in range(len(keypoints_list))]
        axes[1].legend(handles=gt_legend, loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.close()
    
    def evaluate_single(self, batch, batch_idx, visualize=True):
        """
        Evaluate a single batch item.
        
        Args:
            batch: Batch from dataloader
            batch_idx: Index within batch
            visualize: Whether to save visualization
            
        Returns:
            Dictionary with evaluation results
        """
        image = batch['image'][batch_idx]
        keypoints_list = batch['keypoints'][batch_idx]
        distances = batch['distances'][batch_idx]
        img_id = batch['img_id'][batch_idx]
        
        # Create graph
        graph = self.preprocessor.create_mixed_graph(image, keypoints_list, distances)
        
        if graph is None:
            return None
        
        graph = graph.to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.model(graph)
        
        embeddings_np = embeddings.cpu().numpy()
        ground_truth = graph.person_labels.cpu().numpy()
        
        # Cluster
        predicted_clusters = self.cluster_embeddings(embeddings_np)
        
        # Calculate metrics (excluding noise points)
        valid_mask = predicted_clusters != -1
        if valid_mask.sum() > 0:
            ari = adjusted_rand_score(ground_truth[valid_mask], predicted_clusters[valid_mask])
            nmi = normalized_mutual_info_score(ground_truth[valid_mask], predicted_clusters[valid_mask])
        else:
            ari = 0.0
            nmi = 0.0
        
        n_clusters = len(set(predicted_clusters) - {-1})
        n_noise = (predicted_clusters == -1).sum()
        n_people_gt = len(keypoints_list)
        
        results = {
            'img_id': img_id,
            'n_clusters': n_clusters,
            'n_people_gt': n_people_gt,
            'n_noise': n_noise,
            'ari': ari,
            'nmi': nmi,
        }
        
        # Visualize
        if visualize:
            save_path = self.output_dir / f"test_image_{img_id}.png"
            self.visualize_skeleton(
                image, keypoints_list, predicted_clusters, ground_truth,
                img_id, save_path
            )
        
        return results
    
    def evaluate_dataset(self, split="three_persons_test", max_images=None):
        """
        Evaluate on entire test dataset.
        
        Args:
            split: Dataset split to evaluate on
            max_images: Maximum number of images to evaluate (None for all)
            
        Returns:
            Summary statistics
        """
        print(f"\nEvaluating on '{split}' split...")
        print("-" * 50)
        
        dataloader = create_virtual_dataloader(split=split, shuffle=False, num_workers=0)
        
        all_results = []
        image_count = 0
        
        for batch in dataloader:
            batch_size = batch['image'].shape[0]
            
            for i in range(batch_size):
                if max_images and image_count >= max_images:
                    break
                
                result = self.evaluate_single(batch, i, visualize=True)
                
                if result:
                    all_results.append(result)
                    print(f"Image {result['img_id']}: "
                          f"Clusters={result['n_clusters']} (GT={result['n_people_gt']}), "
                          f"Noise={result['n_noise']}, "
                          f"ARI={result['ari']:.3f}, NMI={result['nmi']:.3f}")
                
                image_count += 1
            
            if max_images and image_count >= max_images:
                break
        
        # Summary statistics
        if all_results:
            avg_ari = np.mean([r['ari'] for r in all_results])
            avg_nmi = np.mean([r['nmi'] for r in all_results])
            avg_clusters = np.mean([r['n_clusters'] for r in all_results])
            avg_noise = np.mean([r['n_noise'] for r in all_results])
            
            # How often did we get the right number of clusters?
            correct_count = sum(1 for r in all_results if r['n_clusters'] == r['n_people_gt'])
            accuracy = correct_count / len(all_results)
            
            print("\n" + "=" * 50)
            print("EVALUATION SUMMARY")
            print("=" * 50)
            print(f"Total images evaluated: {len(all_results)}")
            print(f"Average clusters found: {avg_clusters:.2f}")
            print(f"Average noise points: {avg_noise:.2f}")
            print(f"Cluster count accuracy: {accuracy:.2%}")
            print(f"Average ARI: {avg_ari:.4f}")
            print(f"Average NMI: {avg_nmi:.4f}")
            print(f"\nResults saved to: {self.output_dir}")
            
            # Save summary
            summary = {
                'split': split,
                'n_images': len(all_results),
                'avg_ari': avg_ari,
                'avg_nmi': avg_nmi,
                'avg_clusters': avg_clusters,
                'avg_noise': avg_noise,
                'cluster_count_accuracy': accuracy,
                'individual_results': all_results
            }
            
            return summary
        
        return None


def main():
    """Main evaluation script."""
    
    # ===========================================
    # CONFIGURATION - Edit these parameters
    # ===========================================
    
    # checkpoint_path = "/home/dean/projects/mills_ds/outputs/pipeline/b1482612-89d8-479d-8faa-4fcc80aabf48/checkpoints/model_final.pt"
    checkpoint_path = "/home/dean/projects/mills_ds/outputs/pipeline/b1482612-89d8-479d-8faa-4fcc80aabf48/checkpoints/model_epoch_100.pt"
    split = "four_persons_test"
    max_images = None  # Set to a number to limit, or None for all
    device = "cuda"
    # ===========================================
    
    evaluator = Evaluator(checkpoint_path, device=device)
    evaluator.evaluate_dataset(split=split, max_images=max_images)


if __name__ == '__main__':
    main()