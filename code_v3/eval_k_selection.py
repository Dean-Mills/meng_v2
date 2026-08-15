"""
K-selection for the autonomous pipeline — feasibility clamp + candidate search.

Test-time compute applied to the K estimate instead of the assignment.
The K-head predicts K per scene (76% exact / 92% off-by-1 on detections);
this experiment asks whether cheap per-scene K search recovers the
misestimates.

Selectors compared (each scored with kNN (k-means) and THA grouping):

    oracle   K = ground-truth person count            (upper reference)
    pred     K = K-head prediction                    (current autonomous baseline)
    clamp    K = max(pred, K_lb)                      (the feasibility freebie)
    select   K = argmax silhouette over candidates    (candidate search)
    ceiling  per-scene best candidate by PGA          (headroom — oracle selection,
                                                       not a deployable method)

where K_lb = max count of any single joint type in the scene — a hard
lower bound on the number of people among the *observed* keypoints: with
K < K_lb, per-type Hungarian assignment is structurally forced to violate
the type constraint. Candidates for `select` are {pred-1, pred, pred+1},
each raised to K_lb; k=1 is scored as silhouette 0.0 by convention.

Modes:
    GT keypoints (control — pred-K is known to HURT here):
        python eval_k_selection.py \
            --checkpoint outputs/pg_gat_finetune_sweep/3362f752/best.pt \
            --k_head outputs/k_head_meng_headline_coco/latest/best.pt \
            --coco_img_dir data/coco2017/val2017 \
            --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json

    End-to-end HigherHRNet detections (pred-K is known to HELP here):
        python eval_k_selection.py --e2e \
            --checkpoint outputs/pg_gat_finetune_sweep/3362f752/best.pt \
            --k_head outputs/k_head_meng_headline_coco/latest/best.pt \
            --coco_img_dir data/coco2017/val2017 \
            --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json

This file is a pure addition — it imports the existing detection/matching/
grouping code unchanged and modifies no existing module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from config import ExperimentConfig
from gat import GATEmbedding
from preprocessor import PosePreprocessor
from dataset import PoseDataset
from dataloader import create_dataloader
from evaluator import compute_pga, predict_knn
from eval_hungarian_grouping import predict_tha

NUM_JOINT_TYPES = 17

SELECTORS = ["oracle", "pred", "clamp", "select", "ceiling"]


# ─────────────────────────────────────────────────────────────────────────────
# K bounds and selection
# ─────────────────────────────────────────────────────────────────────────────

def k_lower_bound(joint_types: torch.Tensor) -> int:
    """
    Hard lower bound on the number of people among the observed keypoints:
    the max count of any single joint type. With K below this, no
    type-constrained grouping can avoid duplicate types in a cluster.
    """
    jt = joint_types.cpu().numpy()
    counts = np.bincount(jt, minlength=NUM_JOINT_TYPES)
    return int(counts.max())


def _silhouette(emb_np: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette score with the k=1 / degenerate conventions documented above."""
    n_labels = len(np.unique(labels))
    n = len(emb_np)
    if n_labels < 2:
        return 0.0  # k=1 convention — neutral score
    if n_labels >= n:
        return -1.0  # every point its own cluster — degenerate
    from sklearn.metrics import silhouette_score
    try:
        return float(silhouette_score(emb_np, labels))
    except ValueError:
        return -1.0


def evaluate_scene(
    embeddings: torch.Tensor,
    joint_types: torch.Tensor,
    gt_labels: torch.Tensor,
    k_gt: int,
    k_pred: int,
) -> Dict:
    """
    Run all K selectors on one scene.

    Returns:
        {
          "k_lb": int,
          "<selector>": {"k": int, "knn_pga": float, "tha_pga": float},
          ...
        }
    ceiling reports per-method best PGA over the candidate set (its two
    PGAs may come from different K values — it is a headroom bound, not
    a method).
    """
    n = embeddings.size(0)
    emb_np = embeddings.cpu().numpy()
    k_lb = k_lower_bound(joint_types)

    def clip(k: int) -> int:
        return int(min(max(k, 1), n))

    # Candidate set for `select`: {pred-1, pred, pred+1} raised to K_lb
    candidates = sorted({clip(max(k, k_lb)) for k in (k_pred - 1, k_pred, k_pred + 1)})

    # All K values needed by any selector
    k_values = sorted(set(candidates) | {clip(k_pred), clip(k_gt), clip(max(k_pred, k_lb))})

    # Group once per distinct K
    per_k: Dict[int, Dict] = {}
    for k in k_values:
        knn_labels = predict_knn(embeddings, k)
        tha_labels = predict_tha(embeddings, k, joint_types)
        per_k[k] = {
            "knn_pga": compute_pga(knn_labels, gt_labels),
            "tha_pga": compute_pga(tha_labels, gt_labels),
            "silhouette": _silhouette(emb_np, tha_labels.cpu().numpy()),
        }

    def row(k: int) -> Dict:
        return {"k": k, "knn_pga": per_k[k]["knn_pga"], "tha_pga": per_k[k]["tha_pga"]}

    out = {"k_lb": k_lb}
    out["oracle"] = row(clip(k_gt))
    out["pred"] = row(clip(k_pred))
    out["clamp"] = row(clip(max(k_pred, k_lb)))

    # select: best silhouette among candidates (ties → smallest K)
    k_sel = max(candidates, key=lambda k: (per_k[k]["silhouette"], -k))
    out["select"] = row(k_sel)

    # ceiling: per-method best over candidates (headroom, not deployable)
    out["ceiling"] = {
        "k": -1,
        "knn_pga": max(per_k[k]["knn_pga"] for k in candidates),
        "tha_pga": max(per_k[k]["tha_pga"] for k in candidates),
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (mirrors the eval_hungarian_grouping branching)
# ─────────────────────────────────────────────────────────────────────────────

def _load_models(checkpoint_path: Path, k_head_path: Path, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ExperimentConfig(**ckpt["config"])

    is_hyperbolic = cfg.sa_gat_hyperbolic is not None
    if is_hyperbolic:
        from sa_gat_hyperbolic import SAGATHyperbolicEmbedding
        gat = SAGATHyperbolicEmbedding(cfg.sa_gat_hyperbolic).to(device)
        embedding_dim = cfg.sa_gat_hyperbolic.output_dim
        use_depth = cfg.sa_gat_hyperbolic.use_depth
    elif cfg.sa_gat is not None:
        from sa_gat import SAGATEmbedding
        gat = SAGATEmbedding(cfg.sa_gat).to(device)
        embedding_dim = cfg.sa_gat.output_dim
        use_depth = cfg.sa_gat.use_depth
    else:
        gat = GATEmbedding(cfg.gat).to(device)
        embedding_dim = cfg.gat.output_dim
        use_depth = cfg.gat.use_depth
    gat.load_state_dict(ckpt["gat_state"])
    gat.eval()

    from k_head import KEstimationHead
    k_head_ckpt = torch.load(k_head_path, map_location=device, weights_only=False)
    k_head = KEstimationHead(embedding_dim=embedding_dim).to(device)
    k_head.load_state_dict(k_head_ckpt["k_head_state"])
    k_head.eval()

    k_neighbors = 16 if embedding_dim >= 256 else 8
    preprocessor = PosePreprocessor(
        device=device, k_neighbors=k_neighbors, use_depth=use_depth,
    )

    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}) + K-head")
    return gat, k_head, preprocessor, use_depth, is_hyperbolic


def _to_euclidean(gat, embeddings: torch.Tensor, is_hyperbolic: bool) -> torch.Tensor:
    if is_hyperbolic:
        import torch.nn.functional as F
        logmap = gat.manifold.logmap0(embeddings)
        embeddings = F.normalize(logmap[..., 1:], p=2, dim=-1)
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Accumulation / reporting
# ─────────────────────────────────────────────────────────────────────────────

def _new_accumulator() -> Dict:
    acc = {s: {"knn": [], "tha": [], "k": []} for s in SELECTORS}
    acc["k_gt"] = []
    acc["k_eff"] = []
    acc["k_lb"] = []
    return acc


def _accumulate(acc: Dict, scene: Dict, k_gt: int, k_eff: Optional[int]):
    for s in SELECTORS:
        acc[s]["knn"].append(scene[s]["knn_pga"])
        acc[s]["tha"].append(scene[s]["tha_pga"])
        acc[s]["k"].append(scene[s]["k"])
    acc["k_gt"].append(k_gt)
    acc["k_eff"].append(k_eff if k_eff is not None else k_gt)
    acc["k_lb"].append(scene["k_lb"])


def _summarise(acc: Dict, n_restarts_note: str = "") -> Dict:
    n = len(acc["k_gt"])
    k_gt = np.array(acc["k_gt"])
    k_eff = np.array(acc["k_eff"])

    summary = {"n_images": n}
    for s in SELECTORS:
        knn = acc[s]["knn"]
        tha = acc[s]["tha"]
        ks = np.array(acc[s]["k"])
        entry = {
            "knn_pga": sum(knn) / n,
            "tha_pga": sum(tha) / n,
        }
        if s != "ceiling":
            entry.update({
                "k_mean": float(ks.mean()),
                "k_exact_vs_gt": float((ks == k_gt).mean()),
                "k_off_by_1_vs_gt": float((np.abs(ks - k_gt) <= 1).mean()),
                "k_exact_vs_eff": float((ks == k_eff).mean()),
            })
        summary[s] = entry
    summary["k_gt_mean"] = float(k_gt.mean())
    summary["k_eff_mean"] = float(k_eff.mean())
    summary["k_lb_mean"] = float(np.mean(acc["k_lb"]))
    return summary


def _print_summary(summary: Dict, title: str):
    n = summary["n_images"]
    print(f"\n{'='*72}")
    print(f"K-SELECTION — {title} ({n} images)")
    print(f"{'='*72}")
    print(f"{'Selector':<12}{'K mean':>8}{'K=gt':>8}{'|Δ|≤1':>8}{'K=eff':>8}"
          f"{'kNN PGA':>10}{'THA PGA':>10}")
    print("-" * 72)
    for s in SELECTORS:
        e = summary[s]
        if s == "ceiling":
            print(f"{s:<12}{'—':>8}{'—':>8}{'—':>8}{'—':>8}"
                  f"{e['knn_pga']:>10.4f}{e['tha_pga']:>10.4f}")
        else:
            print(f"{s:<12}{e['k_mean']:>8.2f}{e['k_exact_vs_gt']:>8.1%}"
                  f"{e['k_off_by_1_vs_gt']:>8.1%}{e['k_exact_vs_eff']:>8.1%}"
                  f"{e['knn_pga']:>10.4f}{e['tha_pga']:>10.4f}")
    print("-" * 72)
    print(f"K_gt mean {summary['k_gt_mean']:.2f} | K_eff mean {summary['k_eff_mean']:.2f}"
          f" | K_lb mean {summary['k_lb_mean']:.2f}")
    print(f"{'='*72}")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: GT keypoints (control)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_gt_kps(
    checkpoint_path: Path,
    k_head_path: Path,
    device: str,
    coco_img_dir: Path,
    coco_ann_file: Path,
    max_images: Optional[int] = None,
):
    gat, k_head, preprocessor, use_depth, is_hyperbolic = _load_models(
        checkpoint_path, k_head_path, device,
    )

    from coco_adapter import CocoAdapter
    adapter = CocoAdapter(
        img_dir=coco_img_dir, ann_file=coco_ann_file, device=device,
        use_depth=use_depth,
    )
    print(f"Evaluating on COCO GT keypoints: {coco_ann_file.name}")

    dataset = PoseDataset(adapter)
    loader = create_dataloader(dataset, batch_size=4, shuffle=False, num_workers=0)

    acc = _new_accumulator()

    with torch.no_grad():
        for batch in loader:
            graphs = preprocessor.process_batch(batch)
            for graph in graphs:
                if max_images and len(acc["k_gt"]) >= max_images:
                    break
                graph = graph.to(device)
                embeddings = _to_euclidean(gat, gat(graph), is_hyperbolic)

                k_gt = int(graph.num_people)
                k_pred = k_head.predict(embeddings)

                scene = evaluate_scene(
                    embeddings, graph.joint_types, graph.person_labels,
                    k_gt=k_gt, k_pred=k_pred,
                )
                _accumulate(acc, scene, k_gt, k_eff=None)

    summary = _summarise(acc)
    _print_summary(summary, "COCO GT keypoints (control)")

    save_path = checkpoint_path.parent / "eval_k_selection_coco.json"
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {save_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: end-to-end on HigherHRNet detections
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_e2e(
    checkpoint_path: Path,
    k_head_path: Path,
    device: str,
    coco_img_dir: Path,
    coco_ann_file: Path,
    hrnet_weights: Path,
    hrnet_device: str = "cpu",
    max_images: Optional[int] = None,
):
    import cv2
    # Importing eval_end_to_end also puts vendors/simple-HigherHRNet on sys.path
    from eval_end_to_end import (
        match_detections_to_gt,
        hrnet_joints_to_detections,
        build_graph_from_detections,
    )

    gat, k_head, preprocessor, use_depth, is_hyperbolic = _load_models(
        checkpoint_path, k_head_path, device,
    )

    from SimpleHigherHRNet import SimpleHigherHRNet
    hrnet = SimpleHigherHRNet(
        c=32, nof_joints=17,
        checkpoint_path=str(hrnet_weights),
        resolution=512,
        device=torch.device(hrnet_device),
    )
    print(f"Loaded HigherHRNet w32-512 on {hrnet_device}")

    from pycocotools.coco import COCO
    print("Loading COCO annotations...")
    coco = COCO(str(coco_ann_file))

    cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = sorted(coco.getImgIds(catIds=cat_ids))

    # Same image filter as eval_end_to_end
    valid_img_ids = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        n_people = sum(
            1 for ann in anns
            if (np.array(ann["keypoints"]).reshape(17, 3)[:, 2] > 0).sum() >= 3
        )
        if n_people >= 1:
            valid_img_ids.append(img_id)
    print(f"COCO: {len(valid_img_ids)} valid images")

    if max_images:
        valid_img_ids = valid_img_ids[:max_images]

    acc = _new_accumulator()

    with torch.no_grad():
        for i, img_id in enumerate(valid_img_ids):
            img_info = coco.loadImgs(img_id)[0]
            image = cv2.imread(str(coco_img_dir / img_info["file_name"]))
            if image is None:
                continue

            joints = hrnet.predict(image)
            det_pos, det_types, _ = hrnet_joints_to_detections(joints)
            if len(det_pos) < 2:
                continue

            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
            anns = coco.loadAnns(ann_ids)
            gt_kps_list = []
            for ann in anns:
                kps = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)
                if (kps[:, 2] > 0).sum() >= 3:
                    kps_4 = np.zeros((17, 4), dtype=np.float32)
                    kps_4[:, 0] = kps[:, 0]
                    kps_4[:, 1] = kps[:, 1]
                    kps_4[:, 3] = kps[:, 2]
                    gt_kps_list.append(kps_4)
            if len(gt_kps_list) < 1:
                continue

            matched_det_idx, matched_gt_person, matched_gt_type = \
                match_detections_to_gt(det_pos, det_types, gt_kps_list)
            if len(matched_det_idx) < 2:
                continue

            graph = build_graph_from_detections(
                det_pos[matched_det_idx], det_types[matched_det_idx],
                preprocessor, device,
            )
            if graph is None:
                continue
            graph = graph.to(device)
            embeddings = _to_euclidean(gat, gat(graph), is_hyperbolic)

            gt_labels = torch.tensor(matched_gt_person, dtype=torch.long)
            k_gt = len(gt_kps_list)
            k_eff = len(np.unique(matched_gt_person))  # people with >=1 detection
            k_pred = max(1, k_head.predict(embeddings))

            scene = evaluate_scene(
                embeddings, graph.joint_types, gt_labels,
                k_gt=k_gt, k_pred=k_pred,
            )
            _accumulate(acc, scene, k_gt, k_eff=k_eff)

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(valid_img_ids)} images "
                      f"({len(acc['k_gt'])} scored)")

    summary = _summarise(acc)
    _print_summary(summary, "End-to-end HigherHRNet detections")

    save_path = checkpoint_path.parent / "eval_k_selection_e2e.json"
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {save_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--k_head", type=Path, required=True,
                        help="K-head checkpoint (k_head_state)")
    parser.add_argument("--coco_img_dir", type=Path, required=True)
    parser.add_argument("--coco_ann_file", type=Path, required=True)
    parser.add_argument("--e2e", action="store_true",
                        help="Evaluate on HigherHRNet detections instead of GT keypoints")
    parser.add_argument("--hrnet_weights", type=Path,
                        default=Path("vendors/simple-HigherHRNet/weights/pose_higher_hrnet_w32_512.pth"))
    parser.add_argument("--hrnet_device", type=str, default="cpu")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Attach eval metrics to this W&B run id. "
                             "Default: read from checkpoint metadata. "
                             "Pass 'none' to skip.")
    args = parser.parse_args()

    if args.e2e:
        summary = evaluate_e2e(
            args.checkpoint, args.k_head, args.device,
            coco_img_dir=args.coco_img_dir, coco_ann_file=args.coco_ann_file,
            hrnet_weights=args.hrnet_weights, hrnet_device=args.hrnet_device,
            max_images=args.max_images,
        )
        dataset = "e2e"
    else:
        summary = evaluate_gt_kps(
            args.checkpoint, args.k_head, args.device,
            coco_img_dir=args.coco_img_dir, coco_ann_file=args.coco_ann_file,
            max_images=args.max_images,
        )
        dataset = "coco"

    run_id = args.wandb_run_id
    if run_id is None:
        from wandb_helpers import get_wandb_run_id_from_ckpt
        run_id = get_wandb_run_id_from_ckpt(args.checkpoint)
    if run_id and run_id.lower() != "none":
        from wandb_helpers import attach_eval_metrics
        flat = {"n_images": summary["n_images"]}
        for s in SELECTORS:
            for key, val in summary[s].items():
                flat[f"{s}_{key}"] = val
        url = attach_eval_metrics(run_id, f"eval/k_selection/{dataset}", flat)
        if url: print(f"W&B summary updated: {url}")


if __name__ == "__main__":
    main()
