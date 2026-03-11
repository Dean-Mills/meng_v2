"""
Evaluator — compares all grouping heads on the test set.

Loads a trained GAT checkpoint and runs three grouping heads:
  - kNN          (no learned head, baseline)
  - Slot attention
  - Graph partitioning

All three use the same GAT embeddings so the comparison is fair.

Metrics:
  - PGA  (Pose Grouping Accuracy) — Hungarian-matched joint assignment accuracy
  - NMI  (Normalized Mutual Information) — embedding quality
  - ARI  (Adjusted Rand Index)
  - Per-joint accuracy
  - Person detection F1

Usage:
    python evaluator.py --checkpoint outputs/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from config import ExperimentConfig
from gat import GATEmbedding
from preprocessor import PosePreprocessor
from dataset import PoseDataset
from dataloader import create_dataloader
from virtual_adapter import VirtualAdapter


COCO_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers  (also imported by trainer.py for validation)
# ─────────────────────────────────────────────────────────────────────────────

def predict_knn(embeddings: torch.Tensor, k: int) -> torch.Tensor:
    """K-means on L2-normalised embeddings. Returns [N] label tensor."""
    from sklearn.cluster import KMeans
    emb_np = embeddings.cpu().numpy()
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(emb_np)
    return torch.tensor(labels, device=embeddings.device, dtype=torch.long)


def predict_slot(
    head,
    embeddings: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Run slot attention head, return [N] label tensor (argmax assignment)."""
    logits, _ = head(embeddings, k)
    return logits.argmax(dim=1)


def predict_partition(
    head,
    embeddings: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Run edge classifier, threshold affinities, connected components → labels.
    Returns [N] label tensor.
    """
    logits, pairs = head(embeddings)
    same          = (logits.sigmoid() > threshold).cpu().numpy()
    pairs_np      = pairs.cpu().numpy()
    n             = embeddings.size(0)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for (i, j), s in zip(pairs_np, same):
        if s:
            union(int(i), int(j))

    labels = torch.tensor(
        [find(i) for i in range(n)],
        device=embeddings.device, dtype=torch.long,
    )

    # Remap to 0..C-1
    unique = torch.unique(labels)
    remap  = {v.item(): i for i, v in enumerate(unique)}
    labels = torch.tensor([remap[l.item()] for l in labels],
                          device=embeddings.device, dtype=torch.long)
    return labels


def compute_pga(
    pred: torch.Tensor,
    gt:   torch.Tensor,
) -> float:
    """Hungarian-matched joint assignment accuracy."""
    pred_np = pred.cpu().numpy()
    gt_np   = gt.cpu().numpy()

    unique_pred = np.unique(pred_np)
    unique_gt   = np.unique(gt_np)
    k           = max(len(unique_pred), len(unique_gt))

    confusion = np.zeros((k, k), dtype=np.int64)
    pred_remap = {v: i for i, v in enumerate(np.unique(pred_np))}
    gt_remap   = {v: i for i, v in enumerate(np.unique(gt_np))}

    for p, g in zip(pred_np, gt_np):
        confusion[pred_remap[p], gt_remap[g]] += 1

    row, col = linear_sum_assignment(-confusion)
    return confusion[row, col].sum() / len(gt_np)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, checkpoint_path: Path, device: str = "cuda"):
        self.device = device
        ckpt        = torch.load(checkpoint_path, map_location=device,
                                 weights_only=False)
        cfg_dict    = ckpt["config"]
        self.cfg    = ExperimentConfig(**cfg_dict)

        # GAT
        self.gat = GATEmbedding(self.cfg.gat).to(device)
        self.gat.load_state_dict(ckpt["gat_state"])
        self.gat.eval()

        # Head (may be None for knn-only checkpoint)
        self.head      = None
        self.head_name = "knn_only"

        if ckpt.get("head_state") is not None:
            if self.cfg.slot_attention is not None:
                from slot_attention import SlotAttention
                self.head = SlotAttention(
                    self.cfg.slot_attention,
                    embedding_dim=self.cfg.gat.output_dim
                ).to(device)
                self.head_name = "slot_attention"

            elif self.cfg.graph_partitioning is not None:
                from graph_partitioning import EdgeClassifier
                self.head = EdgeClassifier(
                    self.cfg.graph_partitioning,
                    embedding_dim=self.cfg.gat.output_dim
                ).to(device)
                self.head_name = "graph_partitioning"

            if self.head is not None:
                self.head.load_state_dict(ckpt["head_state"])
                self.head.eval()

        self.preprocessor = PosePreprocessor(device=device)
        print(f"Loaded checkpoint (epoch {ckpt.get('epoch','?')}, "
              f"val PGA {ckpt.get('val_pga', 0):.4f})")
        print(f"Head: {self.head_name}")

    # ── Per-scene evaluation ──────────────────────────────────────────────────

    def _eval_graph(self, graph) -> Dict:
        graph      = graph.to(self.device)
        embeddings = self.gat(graph)
        k          = int(graph.num_people)
        gt         = graph.person_labels

        results = {}

        # kNN baseline
        knn_pred         = predict_knn(embeddings, k)
        results["knn"]   = self._scene_metrics(knn_pred, gt, embeddings, k,
                                                graph.joint_types)

        # Trained head
        if self.head_name == "slot_attention":
            head_pred = predict_slot(self.head, embeddings, k)
        elif self.head_name == "graph_partitioning":
            head_pred = predict_partition(
                self.head, embeddings,
                self.cfg.graph_partitioning.threshold
            )
        else:
            head_pred = knn_pred

        results[self.head_name] = self._scene_metrics(
            head_pred, gt, embeddings, k, graph.joint_types
        )

        return results

    def _scene_metrics(
        self,
        pred:        torch.Tensor,
        gt:          torch.Tensor,
        embeddings:  torch.Tensor,
        k:           int,
        joint_types: torch.Tensor,
    ) -> Dict:
        pga  = compute_pga(pred, gt)

        # NMI / ARI from embeddings
        emb_np = embeddings.detach().cpu().numpy()
        gt_np  = gt.cpu().numpy()
        try:
            from sklearn.cluster import KMeans
            km      = KMeans(n_clusters=k, random_state=42, n_init=10)
            emb_cl  = km.fit_predict(emb_np)
            nmi     = normalized_mutual_info_score(gt_np, emb_cl)
            ari     = adjusted_rand_score(gt_np, emb_cl)
        except Exception:
            nmi, ari = 0.0, 0.0

        # Person count F1
        n_pred    = len(torch.unique(pred))
        n_gt      = k
        tp        = min(n_pred, n_gt)
        fp        = max(0, n_pred - n_gt)
        fn        = max(0, n_gt - n_pred)
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        det_f1    = 2 * precision * recall / (precision + recall + 1e-8)

        # Per-joint accuracy
        pred_np  = pred.cpu().numpy()
        gt_np_   = gt.cpu().numpy()
        jt_np    = joint_types.cpu().numpy()

        # Build optimal pred→gt mapping
        unique_pred = np.unique(pred_np)
        unique_gt   = np.unique(gt_np_)
        sz          = max(len(unique_pred), len(unique_gt))
        conf        = np.zeros((sz, sz), dtype=np.int64)
        pr          = {v: i for i, v in enumerate(unique_pred)}
        gr          = {v: i for i, v in enumerate(unique_gt)}
        for p, g in zip(pred_np, gt_np_):
            conf[pr[p], gr[g]] += 1
        row, col    = linear_sum_assignment(-conf)
        pred_to_gt  = {unique_pred[r]: unique_gt[c]
                       for r, c in zip(row, col)
                       if r < len(unique_pred) and c < len(unique_gt)}

        per_joint = {}
        for jt in range(17):
            mask = jt_np == jt
            if mask.sum() == 0:
                continue
            correct = sum(
                1 for p, g in zip(pred_np[mask], gt_np_[mask])
                if pred_to_gt.get(p) == g
            )
            per_joint[jt] = correct / mask.sum()

        return {
            "pga":       pga,
            "nmi":       nmi,
            "ari":       ari,
            "det_f1":    det_f1,
            "n_pred":    n_pred,
            "n_gt":      n_gt,
            "per_joint": per_joint,
        }

    # ── Dataset evaluation ────────────────────────────────────────────────────

    def evaluate(
        self,
        virtual_dir: Path,
        split:       str  = "test",
        batch_size:  int  = 4,
    ) -> Dict:
        assert self.cfg.training is not None or True  # training cfg optional here
        adapter  = VirtualAdapter(virtual_dir / split)
        dataset  = PoseDataset(adapter)
        loader   = create_dataloader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0)

        agg: Dict[str, List] = {}

        with torch.no_grad():
            for batch in loader:
                graphs = self.preprocessor.process_batch(batch)
                for graph in graphs:
                    scene_results = self._eval_graph(graph)
                    for method, metrics in scene_results.items():
                        if method not in agg:
                            agg[method] = []
                        agg[method].append(metrics)

        return self._summarise(agg)

    def _summarise(self, agg: Dict[str, List[Dict]]) -> Dict:
        summary = {}
        for method, scenes in agg.items():
            pga_list    = [s["pga"]    for s in scenes]
            nmi_list    = [s["nmi"]    for s in scenes]
            ari_list    = [s["ari"]    for s in scenes]
            det_f1_list = [s["det_f1"] for s in scenes]

            # Per-joint average across scenes
            per_joint_acc: Dict[int, List] = {j: [] for j in range(17)}
            for s in scenes:
                for j, acc in s["per_joint"].items():
                    per_joint_acc[j].append(acc)
            per_joint_avg = {
                COCO_JOINT_NAMES[j]: float(np.mean(v)) if v else 0.0
                for j, v in per_joint_acc.items()
            }

            summary[method] = {
                "n_scenes":     len(scenes),
                "pga_mean":     float(np.mean(pga_list)),
                "pga_std":      float(np.std(pga_list)),
                "nmi_mean":     float(np.mean(nmi_list)),
                "ari_mean":     float(np.mean(ari_list)),
                "det_f1_mean":  float(np.mean(det_f1_list)),
                "per_joint":    per_joint_avg,
            }

        return summary

    def print_comparison(self, summary: Dict):
        methods = list(summary.keys())
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"{'Metric':<25}" + "".join(f"{m:>15}" for m in methods))
        print("-" * 70)

        for metric, label in [
            ("pga_mean",    "PGA (mean)"),
            ("pga_std",     "PGA (std)"),
            ("nmi_mean",    "NMI"),
            ("ari_mean",    "ARI"),
            ("det_f1_mean", "Detection F1"),
        ]:
            row = f"{label:<25}"
            for m in methods:
                row += f"{summary[m][metric]:>15.4f}"
            print(row)

        print("-" * 70)
        print("Per-joint accuracy:")
        for j in range(17):
            name = COCO_JOINT_NAMES[j]
            row  = f"  {name:<23}"
            for m in methods:
                val = summary[m]["per_joint"].get(name, 0.0)
                row += f"{val:>15.4f}"
            print(row)

        print("=" * 70)

    def save_results(self, summary: Dict, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--virtual_dir", type=Path, default=Path("data/virtual"))
    parser.add_argument("--split",      type=str,  default="test")
    parser.add_argument("--save",       type=Path,
                        default=Path("outputs/eval_results.json"))
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    evaluator = Evaluator(args.checkpoint, device=args.device)
    summary   = evaluator.evaluate(args.virtual_dir, split=args.split)

    evaluator.print_comparison(summary)
    evaluator.save_results(summary, args.save)


if __name__ == "__main__":
    main()