"""
Generic end-to-end evaluation: ANY keypoint detector → PG-GAT → grouping → PGA.

The detector-swap test for the modularity claim. `eval_end_to_end.py` is
hard-wired to HigherHRNet; this script accepts any detector that emits
per-person COCO-17 keypoints, pools the keypoints into an ungrouped set
(discarding the detector's own person assignment), and re-groups them with a
frozen PG-GAT checkpoint that has never seen the detector.

Two rows come out of every run:
  1. "native" — the detector's own person assignment scored against GT
     (its internal grouping, whatever mechanism produced it)
  2. "knn"    — PG-GAT embeddings + k-means on the same pooled detections
plus COP-Kmeans / THA rows and an optional predicted-K row (--k_head),
mirroring eval_end_to_end.py so numbers are directly comparable.

Matching protocol is identical to eval_end_to_end.py: per-type Hungarian
matching of detections to GT keypoints at a 10-pixel threshold; PGA is
computed on the matched subset only.

Supported detectors (all pip-installable, no mmcv):
  yolo  — ultralytics YOLO pose family (default yolo11x-pose.pt, auto-downloads)
  rtmo  — RTMO one-stage bottom-up via rtmlib (default RTMO-L, auto-downloads)

Usage:
    python eval_e2e_generic.py \
        --detector yolo \
        --checkpoint outputs/pg_gat_finetune_sweep/3362f752/best.pt \
        --coco_img_dir data/coco2017/val2017 \
        --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json \
        --k_head outputs/k_head_meng_headline_coco/latest/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from config import ExperimentConfig
from gat import GATEmbedding
from preprocessor import PosePreprocessor
from evaluator import compute_pga
from eval_end_to_end import (
    MATCH_THRESHOLD,
    match_detections_to_gt,
    build_graph_from_detections,
    predict_knn,
    predict_cop_kmeans,
    predict_tha,
)


# ─────────────────────────────────────────────────────────────────────────────
# Detector adapters — each returns (positions [N,2] xy px, types [N], person_ids [N])
# ─────────────────────────────────────────────────────────────────────────────

class YoloPoseDetector:
    """ultralytics YOLO pose models (yolo11n/s/m/l/x-pose)."""

    def __init__(self, model_name: Optional[str], device: str):
        from ultralytics import YOLO
        model_name = model_name or "yolo11x-pose.pt"
        self.model = YOLO(model_name)
        self.device = device
        self.name = model_name

    def __call__(self, image_bgr: np.ndarray, conf_threshold: float):
        res = self.model(image_bgr, verbose=False, device=self.device)[0]
        kp = res.keypoints
        if kp is None or kp.xy is None or len(kp.xy) == 0:
            return np.zeros((0, 2)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        xy = kp.xy.cpu().numpy()                      # [n_people, 17, 2]
        conf = (kp.conf.cpu().numpy() if kp.conf is not None
                else np.ones(xy.shape[:2]))           # [n_people, 17]
        return _pool_person_keypoints(xy, conf, conf_threshold)


RTMO_L_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
              "rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip")


class RtmoDetector:
    """RTMO one-stage bottom-up detector via rtmlib (ONNX Runtime)."""

    def __init__(self, model_name: Optional[str], device: str):
        from rtmlib import RTMO
        model_name = model_name or RTMO_L_URL
        self.model = RTMO(onnx_model=model_name, model_input_size=(640, 640),
                          backend="onnxruntime", device=device)
        self.name = model_name.rsplit("/", 1)[-1]

    def __call__(self, image_bgr: np.ndarray, conf_threshold: float):
        keypoints, scores = self.model(image_bgr)     # [n,17,2], [n,17]
        if keypoints is None or len(keypoints) == 0:
            return np.zeros((0, 2)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        return _pool_person_keypoints(np.asarray(keypoints), np.asarray(scores),
                                      conf_threshold)


def _pool_person_keypoints(
    xy: np.ndarray, conf: np.ndarray, conf_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten per-person [n,17,2] keypoints into an ungrouped detection pool,
    keeping the detector's person index so its native grouping can be scored."""
    positions, types, person_ids = [], [], []
    for p in range(xy.shape[0]):
        for j in range(17):
            if conf[p, j] > conf_threshold:
                x, y = xy[p, j]
                if x <= 0 and y <= 0:   # ultralytics pads absent joints with (0,0)
                    continue
                positions.append([x, y])
                types.append(j)
                person_ids.append(p)
    if not positions:
        return np.zeros((0, 2)), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return (np.array(positions), np.array(types, dtype=int),
            np.array(person_ids, dtype=int))


DETECTORS = {"yolo": YoloPoseDetector, "rtmo": RtmoDetector}


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation (mirrors eval_end_to_end.evaluate with a pluggable detector)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    detector_kind: str,
    detector_model: Optional[str],
    detector_device: str,
    checkpoint_path: Path,
    coco_img_dir: Path,
    coco_ann_file: Path,
    device: str,
    max_images: Optional[int] = None,
    conf_threshold: float = 0.1,
    k_head_path: Optional[Path] = None,
):
    # ── Load PG-GAT/GAT (same branch logic as eval_end_to_end) ───────────
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ExperimentConfig(**ckpt["config"])

    is_hyperbolic = False
    if cfg.sa_gat_hyperbolic is not None:
        from sa_gat_hyperbolic import SAGATHyperbolicEmbedding
        gat = SAGATHyperbolicEmbedding(cfg.sa_gat_hyperbolic).to(device)
        embedding_dim = cfg.sa_gat_hyperbolic.output_dim
        use_depth = cfg.sa_gat_hyperbolic.use_depth
        is_hyperbolic = True
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

    k_neighbors = 16 if embedding_dim >= 256 else 8
    preprocessor = PosePreprocessor(
        device=device, k_neighbors=k_neighbors, use_depth=use_depth,
    )
    print(f"Loaded PG-GAT/GAT checkpoint (epoch {ckpt.get('epoch', '?')})")

    # ── Optional K-head ──────────────────────────────────────────────────
    k_head = None
    if k_head_path is not None:
        from k_head import KEstimationHead
        k_head_ckpt = torch.load(k_head_path, map_location=device, weights_only=False)
        k_head = KEstimationHead(embedding_dim=embedding_dim).to(device)
        k_head.load_state_dict(k_head_ckpt["k_head_state"])
        k_head.eval()
        print(f"Loaded K-head from {k_head_path}")

    # ── Load detector ────────────────────────────────────────────────────
    detector = DETECTORS[detector_kind](detector_model, detector_device)
    print(f"Loaded detector: {detector_kind} ({detector.name}) on {detector_device}")

    # ── COCO iteration (identical filter to eval_end_to_end) ─────────────
    from pycocotools.coco import COCO
    print("Loading COCO annotations...")
    coco = COCO(str(coco_ann_file))
    cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = sorted(coco.getImgIds(catIds=cat_ids))

    valid_img_ids = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        people_with_kps = [
            ann for ann in anns
            if (np.array(ann["keypoints"]).reshape(17, 3)[:, 2] > 0).sum() >= 3
        ]
        if len(people_with_kps) >= 1:
            valid_img_ids.append(img_id)
    print(f"COCO: {len(valid_img_ids)} valid images")
    if max_images:
        valid_img_ids = valid_img_ids[:max_images]

    results = {"native": [], "knn": [], "cop_kmeans": [], "tha": []}
    if k_head is not None:
        results["knn_pred_k"] = []
        results["k_pred"] = []
        results["k_gt"] = []
    detection_stats = {"total_gt": 0, "total_matched": 0, "total_detected": 0}

    with torch.no_grad():
        for i, img_id in enumerate(valid_img_ids):
            img_info = coco.loadImgs(img_id)[0]
            image = cv2.imread(str(coco_img_dir / img_info["file_name"]))
            if image is None:
                continue

            # ── Detection ─────────────────────────────────────────────
            det_pos, det_types, native_person_ids = detector(image, conf_threshold)
            if len(det_pos) < 2:
                continue

            # ── GT keypoints ──────────────────────────────────────────
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
            anns = coco.loadAnns(ann_ids)
            gt_kps_list = []
            for ann in anns:
                kps = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)
                if (kps[:, 2] > 0).sum() >= 3:
                    kps_4 = np.zeros((17, 4), dtype=np.float32)
                    kps_4[:, 0], kps_4[:, 1], kps_4[:, 3] = kps[:, 0], kps[:, 1], kps[:, 2]
                    gt_kps_list.append(kps_4)
            if len(gt_kps_list) < 1:
                continue

            # ── Match detections to GT ────────────────────────────────
            matched_det_idx, matched_gt_person, matched_gt_type = \
                match_detections_to_gt(det_pos, det_types, gt_kps_list)
            if len(matched_det_idx) < 2:
                continue

            n_gt_kps = sum((kps[:, 3] > 0).sum() for kps in gt_kps_list)
            detection_stats["total_gt"] += n_gt_kps
            detection_stats["total_matched"] += len(matched_det_idx)
            detection_stats["total_detected"] += len(det_pos)

            n_gt_people = len(gt_kps_list)
            gt_labels = torch.tensor(matched_gt_person, dtype=torch.long)

            # ── 1. Detector's native grouping ─────────────────────────
            native_labels = torch.tensor(
                native_person_ids[matched_det_idx], dtype=torch.long
            )
            results["native"].append(compute_pga(native_labels, gt_labels))

            # ── 2. PG-GAT grouping on the same pooled detections ──────
            graph = build_graph_from_detections(
                det_pos[matched_det_idx], det_types[matched_det_idx],
                preprocessor, device,
            )
            if graph is None:
                continue
            graph = graph.to(device)
            embeddings = gat(graph)
            if is_hyperbolic:
                import torch.nn.functional as F
                embeddings = F.normalize(
                    gat.manifold.logmap0(embeddings)[..., 1:], p=2, dim=-1)

            if len(matched_det_idx) < n_gt_people:
                continue

            knn_pred = predict_knn(embeddings, n_gt_people)
            results["knn"].append(compute_pga(knn_pred, gt_labels.to(device)))

            if k_head is not None:
                k_pred = max(1, k_head.predict(embeddings))
                results["k_pred"].append(k_pred)
                results["k_gt"].append(n_gt_people)
                if len(matched_det_idx) >= k_pred:
                    pred_labels = predict_knn(embeddings, k_pred)
                    results["knn_pred_k"].append(
                        compute_pga(pred_labels, gt_labels.to(device)))

            cop_pred = predict_cop_kmeans(embeddings, n_gt_people, graph.joint_types)
            results["cop_kmeans"].append(compute_pga(cop_pred, gt_labels.to(device)))
            tha_pred = predict_tha(embeddings, n_gt_people, graph.joint_types)
            results["tha"].append(compute_pga(tha_pred, gt_labels.to(device)))

            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(valid_img_ids)} images processed...")

    # ── Report ───────────────────────────────────────────────────────────
    n = len(results["native"])
    if n == 0:
        print("No valid images processed!")
        return {}

    recall = detection_stats["total_matched"] / max(detection_stats["total_gt"], 1)
    precision = detection_stats["total_matched"] / max(detection_stats["total_detected"], 1)

    print(f"\n{'='*65}")
    print(f"GENERIC E2E EVALUATION — detector: {detector_kind} ({n} images)")
    print(f"{'='*65}")
    print(f"  detection recall:    {recall:.4f}")
    print(f"  detection precision: {precision:.4f}")
    print(f"\n{'Method':<28}{'PGA':>10}{'Std':>10}")
    print("-" * 48)
    method_labels = [
        ("native", f"{detector_kind} native grouping"),
        ("knn", "PG-GAT + k-means"),
        ("cop_kmeans", "PG-GAT + COP-Kmeans"),
        ("tha", "PG-GAT + THA"),
    ]
    if results.get("knn_pred_k"):
        method_labels.append(("knn_pred_k", "PG-GAT + k-means (pred K)"))
        k_preds, k_gts = results["k_pred"], results["k_gt"]
        k_exact = sum(1 for p, g in zip(k_preds, k_gts) if p == g) / max(len(k_gts), 1)
        k_off1 = sum(1 for p, g in zip(k_preds, k_gts) if abs(p - g) <= 1) / max(len(k_gts), 1)
        print(f"  [K-head] exact {k_exact:.3f}, off-by-1 {k_off1:.3f}, "
              f"mean pred {sum(k_preds)/len(k_preds):.2f} vs GT {sum(k_gts)/len(k_gts):.2f}")
    for method, label in method_labels:
        vals = results[method]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        print(f"  {label:<26}{mean:>10.4f}{std:>10.4f}")
    print(f"{'='*65}")

    # ── Save next to checkpoint ──────────────────────────────────────────
    save_path = checkpoint_path.parent / f"eval_e2e_{detector_kind}_coco.json"
    save_data = {
        "detector": {"kind": detector_kind, "model": detector.name,
                     "conf_threshold": conf_threshold,
                     "match_threshold_px": MATCH_THRESHOLD},
        "n_images": n,
        "detection": {k: int(v) for k, v in detection_stats.items()},
        "pga": {
            m: {"mean": sum(v) / len(v),
                "std": (sum((x - sum(v)/len(v)) ** 2 for x in v) / len(v)) ** 0.5}
            for m, v in results.items() if v
        },
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {save_path}")

    out = {
        f"{m}_pga": sum(v) / len(v)
        for m, v in results.items()
        if v and m not in ("k_pred", "k_gt")
    }
    if results.get("k_pred"):
        k_preds, k_gts = results["k_pred"], results["k_gt"]
        out["k_exact"] = sum(1 for p, g in zip(k_preds, k_gts) if p == g) / max(len(k_gts), 1)
        out["k_off_by_1"] = sum(1 for p, g in zip(k_preds, k_gts) if abs(p - g) <= 1) / max(len(k_gts), 1)
        out["k_pred_mean"] = sum(k_preds) / max(len(k_preds), 1)
        out["k_gt_mean"] = sum(k_gts) / max(len(k_gts), 1)
    out["n_images"] = n
    out["detection_recall"] = recall
    out["detection_precision"] = precision
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generic E2E evaluation: any COCO-17 detector → PG-GAT → PGA"
    )
    parser.add_argument("--detector", type=str, required=True, choices=sorted(DETECTORS),
                        help="Which detector family to run")
    parser.add_argument("--detector_model", type=str, default=None,
                        help="Model name/path override. yolo: an ultralytics pose model "
                             "(default yolo11x-pose.pt). rtmo: a .onnx path/URL "
                             "(default rtmlib's RTMO-L).")
    parser.add_argument("--detector_device", type=str, default=None,
                        help="Device for the detector (default: cuda for yolo if "
                             "available, cpu for rtmo).")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="PG-GAT or GAT checkpoint")
    parser.add_argument("--coco_img_dir", type=Path, required=True)
    parser.add_argument("--coco_ann_file", type=Path, required=True)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--conf_threshold", type=float, default=0.1,
                        help="Per-keypoint confidence threshold (same default as "
                             "the HigherHRNet protocol in eval_end_to_end.py)")
    parser.add_argument("--k_head", type=Path, default=None,
                        help="Optional K-head checkpoint for the predicted-K row")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Attach eval metrics to this W&B run id. "
                             "Default: read from checkpoint metadata. "
                             "Pass 'none' to skip.")
    args = parser.parse_args()

    if args.detector_device is None:
        args.detector_device = (
            "cuda" if (args.detector == "yolo" and torch.cuda.is_available()) else "cpu"
        )

    metrics = evaluate(
        detector_kind=args.detector,
        detector_model=args.detector_model,
        detector_device=args.detector_device,
        checkpoint_path=args.checkpoint,
        coco_img_dir=args.coco_img_dir,
        coco_ann_file=args.coco_ann_file,
        device=args.device,
        max_images=args.max_images,
        conf_threshold=args.conf_threshold,
        k_head_path=args.k_head,
    )

    # ── Attach to W&B run summary (optional) ──────────────────────────────
    run_id = args.wandb_run_id
    if run_id is None:
        from wandb_helpers import get_wandb_run_id_from_ckpt
        run_id = get_wandb_run_id_from_ckpt(args.checkpoint)
    if run_id and run_id.lower() != "none" and metrics:
        from wandb_helpers import attach_eval_metrics
        url = attach_eval_metrics(run_id, f"eval/e2e_{args.detector}", metrics)
        if url: print(f"W&B summary updated: {url}")


if __name__ == "__main__":
    main()
