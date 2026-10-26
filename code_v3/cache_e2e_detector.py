"""
Cache any detector's COCO val2017 output in the eval_e2e_per_image.py array format (PEng).

The detector-swap suite (eval_e2e_generic.py) re-runs the detector for every
grouping variant and cannot feed the image-evidence model its image. This
script runs a detector ONCE per image and writes, per scene,

    <out_dir>/arrays/<image_id>.npz   det_pos, det_types, ae_ids (= the detector's
                                      own person index: its native grouping),
                                      matched_idx, gt_person, gt_kps, K_gt, K_pred=-1,
                                      file_name
    <out_dir>/per_image.jsonl         image_id, file_name, width, height, counts, native PGA
    <out_dir>/summary.json            native PGA, recall

so eval_peng_gap.py --arrays <out_dir>/arrays scores every backbone (incl. image
evidence, letterbox frame) on identical scenes per detector; its "AE" column is then
that detector's native grouping. Detector adapters, the valid-image filter, GT parsing,
the 0.1 pooling threshold and the 10 px Hungarian matching are imported unchanged from
eval_e2e_generic.py / eval_end_to_end.py.

    python cache_e2e_detector.py --detector yolo --out_dir outputs/e2e_cache_yolo
    python cache_e2e_detector.py --detector rtmo --out_dir outputs/e2e_cache_rtmo --detector_device cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from eval_e2e_generic import DETECTORS
from eval_end_to_end import match_detections_to_gt
from evaluator import compute_pga


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=sorted(DETECTORS))
    ap.add_argument("--detector_model", default=None)
    ap.add_argument("--detector_device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--coco_img_dir", type=Path, default=Path("data/coco2017/val2017"))
    ap.add_argument("--coco_ann_file", type=Path,
                    default=Path("data/coco2017/annotations/person_keypoints_val2017.json"))
    ap.add_argument("--conf_threshold", type=float, default=0.1)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_images", type=int, default=None)
    a = ap.parse_args()

    (a.out_dir / "arrays").mkdir(parents=True, exist_ok=True)
    detector = DETECTORS[a.detector](a.detector_model, a.detector_device)
    print(f"detector: {a.detector} ({detector.name}) on {a.detector_device}", flush=True)

    from pycocotools.coco import COCO
    coco = COCO(str(a.coco_ann_file))
    cat_ids = coco.getCatIds(catNms=["person"])
    valid = []
    for img_id in sorted(coco.getImgIds(catIds=cat_ids)):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False))
        if any((np.array(x["keypoints"]).reshape(17, 3)[:, 2] > 0).sum() >= 3 for x in anns):
            valid.append(img_id)
    if a.max_images:
        valid = valid[:a.max_images]
    print(f"COCO: {len(valid)} valid images", flush=True)

    native, n_gt, n_matched = [], 0, 0
    with open(a.out_dir / "per_image.jsonl", "w") as fh, torch.no_grad():
        for i, img_id in enumerate(valid):
            info = coco.loadImgs(img_id)[0]
            image = cv2.imread(str(a.coco_img_dir / info["file_name"]))
            if image is None:
                continue
            det_pos, det_types, native_ids = detector(image, a.conf_threshold)
            if len(det_pos) < 2:
                continue
            gt_list = []
            for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)):
                k = np.array(ann["keypoints"], dtype=np.float32).reshape(17, 3)
                if (k[:, 2] > 0).sum() >= 3:
                    k4 = np.zeros((17, 4), dtype=np.float32)
                    k4[:, 0], k4[:, 1], k4[:, 3] = k[:, 0], k[:, 1], k[:, 2]
                    gt_list.append(k4)
            if not gt_list:
                continue
            m_idx, m_person, _ = match_detections_to_gt(det_pos, det_types, gt_list)
            if len(m_idx) < 2:
                continue
            m_idx = np.asarray(m_idx)
            pga = float(compute_pga(torch.tensor(native_ids[m_idx], dtype=torch.long),
                                    torch.tensor(m_person, dtype=torch.long)))
            native.append(pga)
            kps = int(sum((k[:, 3] > 0).sum() for k in gt_list))
            n_gt += kps
            n_matched += len(m_idx)
            np.savez_compressed(
                a.out_dir / "arrays" / f"{img_id}.npz",
                det_pos=det_pos.astype(np.float32), det_types=det_types, ae_ids=native_ids,
                matched_idx=m_idx, gt_person=np.asarray(m_person), gt_kps=np.stack(gt_list),
                K_gt=len(gt_list), K_pred=-1, file_name=info["file_name"])
            fh.write(json.dumps({
                "image_id": int(img_id), "file_name": info["file_name"], "width": info["width"],
                "height": info["height"], "n_det": int(len(det_pos)), "n_matched": int(len(m_idx)),
                "n_gt_kps": kps, "K_gt": len(gt_list), "K_native": int(len(np.unique(native_ids))),
                "pga_native": pga}) + "\n")
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(valid)}", flush=True)

    summary = {"detector": a.detector, "model": detector.name, "conf_threshold": a.conf_threshold,
               "n_scenes": len(native), "pga_native": float(np.mean(native)),
               "recall": n_matched / max(n_gt, 1), "matched": n_matched, "gt_kps": n_gt}
    json.dump(summary, open(a.out_dir / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
