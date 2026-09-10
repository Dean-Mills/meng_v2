"""COCO AP@OKS for the modular pipeline: HigherHRNet detections re-grouped by PG-GAT + k-means.

Purpose (journal paper, 2026-09-09): one system-level number next to PGA. Both grouping mechanisms
are scored on IDENTICAL HigherHRNet-W32-512 keypoints; only the person assignment differs.

Differences from the PGA protocol of eval_end_to_end.py / eval_e2e_per_image.py, all deliberate:
  * every COCO val2017 image is processed (5,000), not only scenes with matched detections;
  * PG-GAT receives EVERY pooled detection above the 0.1 confidence threshold, including
    unmatched (false-positive) keypoints - the PGA protocol built the graph from GT-matched
    detections only. This is the deployable setting; no ground truth enters inference;
  * instances are assembled per k-means cluster: one keypoint per joint type (highest
    confidence), missing types left at (0, 0, 0); instance score = mean confidence over the
    17 joint slots (zeros for missing joints), i.e. the HigherHRNet parser's convention,
    applied identically to every method.

Methods written as COCO results files and scored with pycocotools (iouType="keypoints"):
  native_full   HigherHRNet's own grouping, every joint it output (its standard result)
  native_pooled HigherHRNet's own grouping restricted to joints with conf > 0.1
                (the exact keypoint set PG-GAT re-groups)
  pggat_pred    PG-GAT + K-head + k-means, K = max(1, min(K_head, n_det))   <- autonomous
  pggat_oracle  PG-GAT + k-means with K = number of GT people with >= 1 labelled keypoint
                (the COCOeval keypoint GT set; diagnostic only - annotation-level K, never GT identities)
  pggat_kdet    PG-GAT + k-means with K = the detector's own instance count (diagnostic: isolates the
                embedding + k-means assignment from count estimation; uses no ground truth)

--cache_dir stores the detector output per image so grouping variants (--conf_threshold) can be
re-scored without re-running HigherHRNet.

Usage (from code_v3/):
    python eval_e2e_ap.py \
        --checkpoint outputs/pg_gat_finetune_sweep/3362f752/best.pt \
        --k_head outputs/k_head_meng_headline_coco/a636b1c4/best.pt \
        --coco_img_dir data/coco2017/val2017 \
        --coco_ann_file data/coco2017/annotations/person_keypoints_val2017.json \
        --out_dir outputs/e2e_ap --cache_dir outputs/e2e_ap/hrnet_cache --hrnet_device cuda
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_end_to_end import build_graph_from_detections  # noqa: E402
from eval_e2e_per_image import load_models  # noqa: E402
from evaluator import predict_knn  # noqa: E402

CONF_THRESHOLD = 0.1      # same pooling threshold as eval_end_to_end.hrnet_joints_to_detections
NOF_JOINTS = 17
METHODS = ("native_full", "native_pooled", "pggat_pred", "pggat_oracle", "pggat_kdet")
COCO_STAT_NAMES = ("AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl")


def pool_detections(joints: np.ndarray, conf_threshold: float = CONF_THRESHOLD):
    """Flatten HigherHRNet output [n_people, 17, 3] (y, x, conf) into pooled arrays.

    Same rule as eval_end_to_end.hrnet_joints_to_detections, plus the confidences.
    Returns positions [N, 2] (x, y), types [N], confs [N], native person ids [N].
    """
    if len(joints) == 0:
        return (np.zeros((0, 2), np.float32), np.zeros(0, int), np.zeros(0, np.float32), np.zeros(0, int))
    pos, typ, conf, pid = [], [], [], []
    for p in range(joints.shape[0]):
        for j in range(NOF_JOINTS):
            y, x, c = joints[p, j]
            if c > conf_threshold:
                pos.append([x, y]); typ.append(j); conf.append(c); pid.append(p)
    if not pos:
        return (np.zeros((0, 2), np.float32), np.zeros(0, int), np.zeros(0, np.float32), np.zeros(0, int))
    return (np.array(pos, np.float32), np.array(typ, int), np.array(conf, np.float32), np.array(pid, int))


def instance_score(kps17: np.ndarray) -> float:
    """HigherHRNet parser convention: mean confidence over all 17 joint slots (missing = 0)."""
    return float(kps17[:, 2].mean())


def coco_result(image_id: int, kps17: np.ndarray) -> dict:
    return {
        "image_id": int(image_id),
        "category_id": 1,
        "keypoints": [float(v) for v in kps17.reshape(-1)],
        "score": instance_score(kps17),
    }


def native_instances(joints: np.ndarray, image_id: int, conf_threshold: float | None):
    """HigherHRNet's own grouping. conf_threshold=None keeps every joint (standard output);
    otherwise joints at or below the threshold are zeroed (the pooled keypoint set)."""
    out = []
    for p in range(len(joints)):
        k = np.zeros((NOF_JOINTS, 3), np.float32)
        k[:, 0] = joints[p, :, 1]   # x
        k[:, 1] = joints[p, :, 0]   # y
        k[:, 2] = joints[p, :, 2]
        if conf_threshold is not None:
            k[k[:, 2] <= conf_threshold] = 0.0
        if (k[:, 2] > 0).any():
            out.append(coco_result(image_id, k))
    return out


def assemble_instances(pos, typ, conf, labels: np.ndarray, image_id: int):
    """One instance per k-means cluster; per joint type keep the highest-confidence detection."""
    out = []
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        k = np.zeros((NOF_JOINTS, 3), np.float32)
        for j in range(NOF_JOINTS):
            cand = members[typ[members] == j]
            if len(cand) == 0:
                continue
            best = cand[np.argmax(conf[cand])]
            k[j] = (pos[best, 0], pos[best, 1], conf[best])
        if (k[:, 2] > 0).any():
            out.append(coco_result(image_id, k))
    return out


def coco_eval(coco_gt, results: list, img_ids: list) -> dict:
    """Run COCOeval (keypoints) on the given images; returns the 10 standard stats.

    An empty results list scores 0 on every stat (pycocotools cannot load it)."""
    from pycocotools.cocoeval import COCOeval
    if not results:
        return {n: 0.0 for n in COCO_STAT_NAMES}
    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, iouType="keypoints")
    ev.params.imgIds = img_ids
    buf = io.StringIO()
    with redirect_stdout(buf):            # keep the 12-line summary out of the loop output
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {n: float(v) for n, v in zip(COCO_STAT_NAMES, ev.stats)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--k_head", type=Path, required=True)
    ap.add_argument("--coco_img_dir", type=Path, required=True)
    ap.add_argument("--coco_ann_file", type=Path, required=True)
    ap.add_argument("--hrnet_weights", type=Path,
                    default=Path("vendors/simple-HigherHRNet/weights/pose_higher_hrnet_w32_512.pth"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hrnet_device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache_dir", type=Path, default=None,
                    help="cache HigherHRNet joints per image as <id>.npy; re-runs then skip the detector")
    ap.add_argument("--conf_threshold", type=float, default=CONF_THRESHOLD,
                    help="pooling threshold for the keypoints PG-GAT re-groups (PGA protocol: 0.1)")
    ap.add_argument("--img_set", choices=["all", "person"], default="all",
                    help="all = every val2017 image (standard AP); person = images with a person annotation")
    ap.add_argument("--max_images", type=int, default=None, help="smoke test: first N images only")
    ap.add_argument("--wandb_run_id", type=str, default=None,
                    help="W&B run to attach eval/e2e_ap metrics to (default: the checkpoint's run; 'none' disables)")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if a.cache_dir:
        a.cache_dir.mkdir(parents=True, exist_ok=True)

    gat, pre, k_head, hrnet = load_models(a.checkpoint, a.k_head, a.hrnet_weights, a.device, a.hrnet_device)

    from pycocotools.coco import COCO
    coco = COCO(str(a.coco_ann_file))
    cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = sorted(coco.getImgIds(catIds=cat_ids) if a.img_set == "person" else coco.getImgIds())
    if a.max_images:
        img_ids = img_ids[:a.max_images]
    print(f"COCO val2017: scoring {len(img_ids)} images ({a.img_set})")

    results = {m: [] for m in METHODS}
    per_image = open(a.out_dir / "per_image.jsonl", "w")
    counts = {"images": 0, "images_with_dets": 0, "n_det": 0, "k_pred": [], "k_oracle": [], "k_det": [], "n_native": 0}
    t0 = time.time()
    with torch.no_grad():
        for i, img_id in enumerate(img_ids):
            info = coco.loadImgs(img_id)[0]
            cache_file = (a.cache_dir / f"{img_id}.npy") if a.cache_dir else None
            if cache_file is not None and cache_file.exists():
                joints = np.load(cache_file)
            else:
                image = cv2.imread(str(a.coco_img_dir / info["file_name"]))
                if image is None:
                    print(f"  skip unreadable {info['file_name']}")
                    continue
                joints = hrnet.predict(image)                  # [n_people, 17, 3] (y, x, conf)
                joints = np.asarray(joints, dtype=np.float32).reshape(-1, NOF_JOINTS, 3) if len(joints) else np.zeros((0, NOF_JOINTS, 3), np.float32)
                if cache_file is not None:
                    np.save(cache_file, joints)
            counts["images"] += 1
            counts["n_native"] += len(joints)

            results["native_full"] += native_instances(joints, img_id, None)
            results["native_pooled"] += native_instances(joints, img_id, a.conf_threshold)

            pos, typ, conf, _ = pool_detections(joints, a.conf_threshold)
            n_det = len(pos)
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False))
            n_gt = sum(1 for x in anns if x.get("num_keypoints", 0) > 0)   # the COCOeval keypoint GT set
            rec = {"image_id": int(img_id), "n_native": int(len(joints)), "n_det": int(n_det),
                   "n_gt": int(n_gt), "K_pred": None, "K_oracle": None, "K_det": None}
            if n_det > 0:
                counts["images_with_dets"] += 1
                counts["n_det"] += n_det
                if n_det == 1:
                    K_pred = K_oracle = K_det = 1
                    labels = {1: np.zeros(1, int)}
                else:
                    graph = build_graph_from_detections(pos, typ, pre, a.device).to(a.device)
                    emb = gat(graph)
                    K_pred = max(1, min(int(k_head.predict(emb)), n_det))
                    K_oracle = max(1, min(n_gt, n_det))
                    K_det = max(1, min(int(len(joints)), n_det))
                    labels = {}
                    for K in {K_pred, K_oracle, K_det}:          # k-means once per distinct K
                        labels[K] = predict_knn(emb, K).cpu().numpy()
                rec["K_pred"], rec["K_oracle"], rec["K_det"] = K_pred, K_oracle, K_det
                counts["k_pred"].append(K_pred); counts["k_oracle"].append(K_oracle); counts["k_det"].append(K_det)
                results["pggat_pred"] += assemble_instances(pos, typ, conf, labels[K_pred], img_id)
                results["pggat_oracle"] += assemble_instances(pos, typ, conf, labels[K_oracle], img_id)
                results["pggat_kdet"] += assemble_instances(pos, typ, conf, labels[K_det], img_id)
            per_image.write(json.dumps(rec) + "\n")
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(img_ids)}  ({time.time() - t0:.0f} s)", flush=True)
    per_image.close()

    summary = {
        "protocol": {
            "detector": "HigherHRNet-W32-512 via simple-HigherHRNet (single scale, no flip)",
            "img_set": a.img_set, "n_images": counts["images"], "conf_threshold": a.conf_threshold,
            "pggat_input": "all pooled detections above conf_threshold (no GT matching)",
            "instance_score": "mean confidence over 17 joint slots, missing joints = 0",
            "checkpoint": str(a.checkpoint), "k_head": str(a.k_head),
        },
        "counts": {
            "images_with_dets": counts["images_with_dets"], "pooled_detections": counts["n_det"],
            "native_instances": counts["n_native"],
            **{f"instances_{m}": len(results[m]) for m in METHODS},
            "k_pred_mean": float(np.mean(counts["k_pred"])) if counts["k_pred"] else None,
            "k_oracle_mean": float(np.mean(counts["k_oracle"])) if counts["k_oracle"] else None,
            "k_det_mean": float(np.mean(counts["k_det"])) if counts["k_det"] else None,
            "k_exact_vs_oracle": float(np.mean(np.array(counts["k_pred"]) == np.array(counts["k_oracle"]))) if counts["k_pred"] else None,
        },
        "coco": {},
    }
    scored_ids = [int(x) for x in img_ids]
    for m in METHODS:
        json.dump(results[m], open(a.out_dir / f"results_{m}.json", "w"))
        summary["coco"][m] = coco_eval(coco, results[m], scored_ids)
    json.dump(summary, open(a.out_dir / "summary.json", "w"), indent=2)

    print("\n{:<14s}".format("method") + "".join(f"{n:>7s}" for n in COCO_STAT_NAMES))
    for m in METHODS:
        print(f"{m:<14s}" + "".join(f"{summary['coco'][m][n]:7.3f}" for n in COCO_STAT_NAMES))
    print(f"\nimages {counts['images']}, with detections {counts['images_with_dets']}, "
          f"K_pred mean {summary['counts']['k_pred_mean']}, K_oracle mean {summary['counts']['k_oracle_mean']}, "
          f"K_det mean {summary['counts']['k_det_mean']}")

    # W&B: same pattern as eval_e2e_generic.py (summary fields on the checkpoint's run)
    run_id = a.wandb_run_id
    if run_id is None:
        from wandb_helpers import get_wandb_run_id_from_ckpt
        run_id = get_wandb_run_id_from_ckpt(a.checkpoint)
    if run_id and run_id.lower() != "none" and not a.max_images:
        from wandb_helpers import attach_eval_metrics
        metrics = {f"{m}/{n}": summary["coco"][m][n] for m in METHODS for n in ("AP", "AP50", "AP75", "AR")}
        metrics["n_images"] = counts["images"]
        prefix = "eval/e2e_ap" if a.conf_threshold == CONF_THRESHOLD else f"eval/e2e_ap_t{a.conf_threshold:g}"
        url = attach_eval_metrics(run_id, prefix, metrics)
        print(f"W&B: {url}" if url else f"W&B: run {run_id} not found - metrics not attached")


if __name__ == "__main__":
    main()
