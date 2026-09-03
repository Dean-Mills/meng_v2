"""Per-image end-to-end dump for the dissertation figures (k-means read-out only).

Reuses eval_end_to_end.py verbatim for HigherHRNet detection, 10 px Hungarian matching,
graph construction and the K-head, so the per-image numbers are the ones behind the
documented aggregates (AE 0.9847 / oracle-K 0.9591 / predicted-K 0.9644 on 2,165 images).

Writes:
  <out_dir>/per_image.jsonl   one record per image reaching the matching stage
  <out_dir>/arrays/<id>.npz   (with --save_arrays) per-detection data to redraw the scene
  <out_dir>/summary.json      aggregates, for comparison against docs/results.md

Usage (from code_v3/):
    python eval_e2e_per_image.py \
        --checkpoint outputs/pg_gat_finetune_sweep/3362f752/best.pt \
        --k_head outputs/k_head_meng_headline_coco/a636b1c4/best.pt \
        --coco_img_dir data/coco2017/val2017 \
        --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json \
        --out_dir outputs/e2e_per_image --hrnet_device cuda --save_arrays
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_end_to_end import (  # noqa: E402
    hrnet_joints_to_detections,
    match_detections_to_gt,
    build_graph_from_detections,
    predict_knn,
)
from config import ExperimentConfig  # noqa: E402
from preprocessor import PosePreprocessor  # noqa: E402
from evaluator import compute_pga  # noqa: E402


def load_models(checkpoint, k_head_path, hrnet_weights, device, hrnet_device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ExperimentConfig(**ckpt["config"])
    assert cfg.sa_gat is not None and cfg.sa_gat_hyperbolic is None and cfg.sa_gat_v2 is None, \
        "this dump is for the Euclidean PG-GAT headline checkpoint"
    from sa_gat import SAGATEmbedding
    gat = SAGATEmbedding(cfg.sa_gat).to(device)
    gat.load_state_dict(ckpt["gat_state"])
    gat.eval()
    embedding_dim = cfg.sa_gat.output_dim
    k_neighbors = 16 if embedding_dim >= 256 else 8   # same rule as trainer/evaluator
    pre = PosePreprocessor(device=device, k_neighbors=k_neighbors, use_depth=cfg.sa_gat.use_depth)

    from k_head import KEstimationHead
    kh = torch.load(k_head_path, map_location=device, weights_only=False)
    k_head = KEstimationHead(embedding_dim=embedding_dim).to(device)
    k_head.load_state_dict(kh["k_head_state"])
    k_head.eval()

    sys.path.insert(0, str(Path(__file__).parent / "vendors" / "simple-HigherHRNet"))
    from SimpleHigherHRNet import SimpleHigherHRNet
    hrnet = SimpleHigherHRNet(c=32, nof_joints=17, checkpoint_path=str(hrnet_weights),
                              resolution=512, device=torch.device(hrnet_device))
    print(f"loaded PG-GAT (epoch {ckpt.get('epoch', '?')}, d={embedding_dim}, k={k_neighbors}), "
          f"K-head, HigherHRNet-W32-512 on {hrnet_device}")
    return gat, pre, k_head, hrnet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--k_head", type=Path, required=True)
    ap.add_argument("--coco_img_dir", type=Path, required=True)
    ap.add_argument("--coco_ann_file", type=Path, required=True)
    ap.add_argument("--hrnet_weights", type=Path,
                    default=Path("vendors/simple-HigherHRNet/weights/pose_higher_hrnet_w32_512.pth"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hrnet_device", default="cpu")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--save_arrays", action="store_true")
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.save_arrays:
        (a.out_dir / "arrays").mkdir(exist_ok=True)
    gat, pre, k_head, hrnet = load_models(a.checkpoint, a.k_head, a.hrnet_weights, a.device, a.hrnet_device)

    from pycocotools.coco import COCO
    coco = COCO(str(a.coco_ann_file))
    cat_ids = coco.getCatIds(catNms=["person"])
    valid = []
    for img_id in sorted(coco.getImgIds(catIds=cat_ids)):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False))
        if any((np.array(x["keypoints"]).reshape(17, 3)[:, 2] > 0).sum() >= 3 for x in anns):
            valid.append(img_id)
    print(f"COCO: {len(valid)} valid images")
    if a.max_images:
        valid = valid[:a.max_images]

    agg = {"ae": [], "oracle": [], "pred": [], "k_pred": [], "k_gt": [], "gt": 0, "matched": 0, "det": 0}
    fh = open(a.out_dir / "per_image.jsonl", "w")
    with torch.no_grad():
        for i, img_id in enumerate(valid):
            info = coco.loadImgs(img_id)[0]
            image = cv2.imread(str(a.coco_img_dir / info["file_name"]))
            if image is None:
                continue
            joints = hrnet.predict(image)
            det_pos, det_types, ae_ids = hrnet_joints_to_detections(joints)
            if len(det_pos) < 2:
                continue
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False))
            gt_list = []
            for ann in anns:
                k = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)
                k4 = np.zeros((17, 4), dtype=np.float32)
                k4[:, 0], k4[:, 1], k4[:, 3] = k[:, 0], k[:, 1], k[:, 2]
                if (k[:, 2] > 0).sum() >= 3:
                    gt_list.append(k4)
            if len(gt_list) < 1:
                continue
            m_idx, m_person, _ = match_detections_to_gt(det_pos, det_types, gt_list)
            if len(m_idx) < 2:
                continue
            n_gt_kps = int(sum((k[:, 3] > 0).sum() for k in gt_list))
            agg["gt"] += n_gt_kps; agg["matched"] += len(m_idx); agg["det"] += len(det_pos)
            K_gt = len(gt_list)
            gt_labels = torch.tensor(m_person, dtype=torch.long)

            pga_ae = float(compute_pga(torch.tensor(ae_ids[m_idx], dtype=torch.long), gt_labels))
            agg["ae"].append(pga_ae)

            graph = build_graph_from_detections(det_pos[m_idx], det_types[m_idx], pre, a.device)
            rec = {"image_id": int(img_id), "file_name": info["file_name"], "width": info["width"],
                   "height": info["height"], "n_det": int(len(det_pos)), "n_matched": int(len(m_idx)),
                   "n_gt_kps": n_gt_kps, "K_gt": K_gt, "K_ae": int(len(np.unique(ae_ids))),
                   "pga_ae": pga_ae, "K_pred": None, "pga_oracle": None, "pga_pred": None}
            lab_o = lab_p = None
            if graph is not None:
                graph = graph.to(a.device)
                emb = gat(graph)
                K_pred = max(1, int(k_head.predict(emb)))
                rec["K_pred"] = K_pred
                if len(m_idx) >= K_gt:          # same skip rule as eval_end_to_end.py
                    lab_o = predict_knn(emb, K_gt)
                    rec["pga_oracle"] = float(compute_pga(lab_o, gt_labels.to(a.device)))
                    agg["oracle"].append(rec["pga_oracle"])
                    agg["k_pred"].append(K_pred); agg["k_gt"].append(K_gt)
                    if len(m_idx) >= K_pred:
                        lab_p = predict_knn(emb, K_pred)
                        rec["pga_pred"] = float(compute_pga(lab_p, gt_labels.to(a.device)))
                        agg["pred"].append(rec["pga_pred"])
            fh.write(json.dumps(rec) + "\n")
            if a.save_arrays:
                np.savez_compressed(
                    a.out_dir / "arrays" / f"{img_id}.npz",
                    det_pos=det_pos, det_types=det_types, ae_ids=ae_ids, matched_idx=m_idx,
                    gt_person=m_person,
                    labels_oracle=(lab_o.cpu().numpy() if lab_o is not None else np.zeros(0, int)),
                    labels_pred=(lab_p.cpu().numpy() if lab_p is not None else np.zeros(0, int)),
                    gt_kps=np.stack(gt_list), K_gt=K_gt, K_pred=rec["K_pred"] or -1,
                    file_name=info["file_name"])
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(valid)}", flush=True)
    fh.close()

    kp, kg = np.array(agg["k_pred"]), np.array(agg["k_gt"])
    summary = {
        "n_ae": len(agg["ae"]), "n_oracle": len(agg["oracle"]), "n_pred": len(agg["pred"]),
        "pga_ae": float(np.mean(agg["ae"])), "pga_oracle": float(np.mean(agg["oracle"])),
        "pga_pred": float(np.mean(agg["pred"])),
        "k_exact": float((kp == kg).mean()), "k_off_by_1": float((np.abs(kp - kg) <= 1).mean()),
        "k_pred_mean": float(kp.mean()), "k_gt_mean": float(kg.mean()),
        "recall": agg["matched"] / max(agg["gt"], 1), "matched": agg["matched"], "gt_kps": agg["gt"],
    }
    json.dump(summary, open(a.out_dir / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
