"""Perfect-grouping AP ceiling per detector (diagnostics pass, 2026-10-26).

How much AP can ANY grouping of a detector's joints reach? On the all-image AP cache (cache_ap_detector.py /
eval_e2e_ap.py format), each pooled detection above the threshold is assigned to the COCOeval GT person
(num_keypoints > 0) whose same-type labelled keypoint gives the highest per-keypoint OKS term
exp(-d^2 / (2 * area * k_t^2)) (COCO sigmas, k = 2 sigma); detections whose best term is below --min_kp_oks are
unmatched. (A first version used the PGA protocol's fixed 10 px matching; it discarded correct joints of large
people and produced a "ceiling" below the native grouping, so it was replaced.)
  oracle_drop_unmatched   matched joints grouped by GT person, unmatched joints removed
  oracle_native_unmatched matched joints grouped by GT person, unmatched joints keep the detector's grouping
Native rows (detector's own grouping) for reference. Same instance assembly and score as eval_e2e_ap.py.

    python -m analysis.ap_ceiling --cache_dir outputs/e2e_ap_yolo/cache --conf_threshold 0 --out <json>
"""
import argparse, json
from pathlib import Path
import numpy as np
from eval_e2e_ap import assemble_instances, coco_eval, native_instances, pool_detections
SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0


def oks_assign(pos, typ, gt, areas, min_kp_oks):
    """Per detection: index of the GT person with the highest per-keypoint OKS term, or -1."""
    lab = np.full(len(pos), -1)
    if not gt:
        return lab
    G = np.stack(gt)                                   # [P, 17, 3] (x, y, v)
    k2 = (2 * SIGMAS[typ]) ** 2                          # [N]
    d2 = ((G[:, typ, :2] - pos[None]) ** 2).sum(-1)     # [P, N]
    term = np.exp(-d2 / (2 * np.maximum(areas, 1.0)[:, None] * k2[None]))
    term[G[:, typ, 2] <= 0] = 0.0                      # unlabelled GT joints do not attract
    best = term.argmax(0)
    ok = term[best, np.arange(len(pos))] >= min_kp_oks
    lab[ok] = best[ok]
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=Path, required=True)
    ap.add_argument("--conf_threshold", type=float, default=0.0)
    ap.add_argument("--coco_ann_file", type=Path, default=Path("data/coco2017/annotations/person_keypoints_val2017.json"))
    ap.add_argument("--min_kp_oks", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from pycocotools.coco import COCO
    coco = COCO(str(a.coco_ann_file)); cat = coco.getCatIds(catNms=["person"])
    img_ids = sorted(coco.getImgIds())
    rows = {k: [] for k in ("native_full", "native_pooled", "oracle_drop_unmatched", "oracle_native_unmatched")}
    n_gt_of = {}
    for img_id in img_ids:
        joints = np.load(a.cache_dir / f"{img_id}.npy")
        rows["native_full"] += native_instances(joints, img_id, None)
        rows["native_pooled"] += native_instances(joints, img_id, a.conf_threshold)
        pos, typ, conf, pid = pool_detections(joints, a.conf_threshold)
        anns = [x for x in coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat, iscrowd=False)) if x.get("num_keypoints", 0) > 0]
        n_gt_of[img_id] = len(anns)
        if len(pos) == 0:
            continue
        gt = [np.array(x["keypoints"], dtype=np.float32).reshape(17, 3) for x in anns]
        areas = np.array([x["area"] for x in anns], dtype=np.float64)
        lab = oks_assign(pos, typ, gt, areas, a.min_kp_oks)
        matched = lab >= 0
        if matched.any():
            rows["oracle_drop_unmatched"] += assemble_instances(pos[matched], typ[matched], conf[matched], lab[matched], img_id)
        lab2 = lab.copy(); lab2[~matched] = 10_000 + pid[~matched]
        rows["oracle_native_unmatched"] += assemble_instances(pos, typ, conf, lab2, img_id)
    subsets = {"all": img_ids, "n_gt >=6": [i for i in img_ids if n_gt_of[i] >= 6]}
    out = {"cache": str(a.cache_dir), "conf_threshold": a.conf_threshold, "min_kp_oks": a.min_kp_oks,
           "coco": {s: {r: coco_eval(coco, rows[r], ids) for r in rows} for s, ids in subsets.items()}}
    json.dump(out, open(a.out, "w"), indent=2)
    for s in subsets:
        print(f"[{s}] " + "  ".join(f"{r} AP {out['coco'][s][r]['AP']:.4f}" for r in rows))


if __name__ == "__main__":
    main()
