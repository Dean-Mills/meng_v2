"""
Cache a detector's raw person outputs on ALL COCO val2017 images for AP@OKS (PEng).

Writes <cache_dir>/<image_id>.npy = [n_people, 17, 3] float32 in (y, x, conf) order --
the layout of the HigherHRNet AP cache written by eval_e2e_ap.py -- so eval_e2e_ap_peng.py
--cache_dir <cache_dir> scores PG-GAT pipelines and the detector's native grouping on it
unchanged. Unlike cache_e2e_detector.py (PGA: GT-matched scenes only, no confidences),
every image is kept, every person and every joint with its confidence.

Detectors are the eval_e2e_generic.py adapters' models (YOLO11x-pose, RTMO-L body7), called
directly for their raw [n, 17, 2] keypoints and [n, 17] scores.

    python cache_ap_detector.py --detector yolo --cache_dir outputs/e2e_ap_yolo/cache
    python cache_ap_detector.py --detector rtmo --cache_dir outputs/e2e_ap_rtmo/cache --detector_device cpu
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from eval_e2e_generic import DETECTORS


def raw_people(kind: str, det, image_bgr: np.ndarray) -> np.ndarray:
    """[n, 17, 3] (y, x, conf) for every person the detector outputs."""
    if kind == "yolo":
        res = det.model(image_bgr, verbose=False, device=det.device)[0]
        kp = res.keypoints
        if kp is None or kp.xy is None or len(kp.xy) == 0:
            return np.zeros((0, 17, 3), np.float32)
        xy = kp.xy.cpu().numpy()
        conf = kp.conf.cpu().numpy() if kp.conf is not None else np.ones(xy.shape[:2])
    else:
        xy, conf = det.model(image_bgr)
        if xy is None or len(xy) == 0:
            return np.zeros((0, 17, 3), np.float32)
        xy, conf = np.asarray(xy), np.asarray(conf)
    out = np.zeros((len(xy), 17, 3), np.float32)
    out[..., 0], out[..., 1], out[..., 2] = xy[..., 1], xy[..., 0], conf
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=sorted(DETECTORS))
    ap.add_argument("--detector_model", default=None)
    ap.add_argument("--detector_device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--coco_img_dir", type=Path, default=Path("data/coco2017/val2017"))
    ap.add_argument("--coco_ann_file", type=Path,
                    default=Path("data/coco2017/annotations/person_keypoints_val2017.json"))
    ap.add_argument("--cache_dir", type=Path, required=True)
    a = ap.parse_args()
    a.cache_dir.mkdir(parents=True, exist_ok=True)

    det = DETECTORS[a.detector](a.detector_model, a.detector_device)
    print(f"detector: {a.detector} ({det.name}) on {a.detector_device}", flush=True)
    from pycocotools.coco import COCO
    coco = COCO(str(a.coco_ann_file))
    img_ids = sorted(coco.getImgIds())
    t0, n_people = time.time(), 0
    for i, img_id in enumerate(img_ids):
        f = a.cache_dir / f"{img_id}.npy"
        if f.exists():
            continue
        image = cv2.imread(str(a.coco_img_dir / coco.loadImgs(img_id)[0]["file_name"]))
        people = raw_people(a.detector, det, image) if image is not None else np.zeros((0, 17, 3), np.float32)
        n_people += len(people)
        np.save(f, people)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(img_ids)} ({time.time() - t0:.0f} s, {n_people} people)", flush=True)
    print(f"done: {len(img_ids)} images in {a.cache_dir}")


if __name__ == "__main__":
    main()
