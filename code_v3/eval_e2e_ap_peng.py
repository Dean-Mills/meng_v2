"""
COCO AP@OKS for the PEng pipelines on HigherHRNet detections (2026-10-22).

eval_e2e_ap.py (journal paper, 2026-09-09) scored the MEng pipeline (meng-headline +
K-head + k-means) against HigherHRNet's own grouping. This script re-uses its protocol and
helpers unchanged -- every val2017 image, every pooled detection above 0.1 (false
positives included, no GT matching), one instance per cluster with the highest-confidence
detection per joint type, instance score = mean confidence over 17 slots -- and its cached
HigherHRNet output (--cache_dir, no detector run), but scores several backbones and K / grouping
rules in one pass, so every row sees identical pooled joints:

    --model NAME=CHECKPOINT[,K_HEAD]     repeatable; any backbone eval_peng_gap.load_models loads
                                         (image-evidence models get the letterboxed image and
                                         letterboxed coordinates; instances keep pixel coordinates)
Rows per model: <NAME>/pred_knn (MEng rule, needs a K-head), <NAME>/clamp_tha
(K = max(K-head, K_lb), the PEng autonomous rule), <NAME>/lb_tha (K_lb only),
<NAME>/oracle_tha (K = annotated people with keypoints; diagnostic).
Plus native_full / native_pooled. Crowd splits by annotated people: 1-5, 2-5, >= 6.

    python eval_e2e_ap_peng.py \
        --model headline=outputs/pg_gat_finetune_sweep/3362f752/best.pt,outputs/k_head_meng_headline_coco/a636b1c4/best.pt \
        --model peng=outputs/finetune_pg_gat_hier_img_coco/372658c6/best.pt,outputs/k_head_aug_372658c6/4579ce0c/best.pt \
        --out_dir outputs/e2e_ap_peng
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval_e2e_ap import (CONF_THRESHOLD, COCO_STAT_NAMES, NOF_JOINTS, assemble_instances,
                         coco_eval, native_instances, pool_detections)
from eval_end_to_end import build_graph_from_detections
from eval_hungarian_grouping import predict_tha
from eval_k_selection import k_lower_bound
from eval_peng_gap import letterbox, load_models
from evaluator import predict_knn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="NAME=CHECKPOINT[,K_HEAD]")
    ap.add_argument("--cache_dir", type=Path, default=Path("outputs/e2e_ap/hrnet_cache"))
    ap.add_argument("--coco_img_dir", type=Path, default=Path("data/coco2017/val2017"))
    ap.add_argument("--coco_ann_file", type=Path,
                    default=Path("data/coco2017/annotations/person_keypoints_val2017.json"))
    ap.add_argument("--conf_threshold", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed numpy / torch so COP-Kmeans restarts are repeatable (verification runs)")
    a = ap.parse_args()
    if a.seed is not None:
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    for spec in a.model:
        name, paths = spec.split("=", 1)
        ck, *kh = paths.split(",")
        gat, k_head, pre = load_models(Path(ck), Path(kh[0]) if kh else None, a.device)
        models[name] = (gat, k_head, pre, getattr(gat, "needs_image", False))

    from pycocotools.coco import COCO
    from torchvision.io import read_image
    from image_evidence import letterbox_image
    coco = COCO(str(a.coco_ann_file))
    cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = sorted(coco.getImgIds())
    if a.max_images:
        img_ids = img_ids[:a.max_images]

    rows = ["native_full", "native_pooled"]
    for name, (_, k_head, _, _) in models.items():
        rows += ([f"{name}/pred_knn"] if k_head is not None else []) + \
                ([f"{name}/clamp_tha"] if k_head is not None else []) + [f"{name}/lb_tha", f"{name}/oracle_tha"]
    results = {r: [] for r in rows}
    n_gt_of, kstats = {}, {r: [] for r in rows if "/" in r}
    t0 = time.time()
    with torch.no_grad():
        for i, img_id in enumerate(img_ids):
            joints = np.load(a.cache_dir / f"{img_id}.npy")
            info = coco.loadImgs(img_id)[0]
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False))
            n_gt = sum(1 for x in anns if x.get("num_keypoints", 0) > 0)
            n_gt_of[img_id] = n_gt
            results["native_full"] += native_instances(joints, img_id, None)
            results["native_pooled"] += native_instances(joints, img_id, a.conf_threshold)
            pos, typ, conf, _ = pool_detections(joints, a.conf_threshold)
            n = len(pos)
            if n == 0:
                continue
            image = None
            for name, (gat, k_head, pre, needs_image) in models.items():
                if n == 1:
                    ks = {"pred_knn": 1, "clamp_tha": 1, "lb_tha": 1, "oracle_tha": 1}
                    labels = {r: np.zeros(1, int) for r in ks}
                else:
                    gpos = pos
                    if needs_image:
                        gpos = letterbox(pos, info["width"], info["height"], pre.image_size)
                        if image is None:
                            img = read_image(str(a.coco_img_dir / info["file_name"]))
                            img = img.repeat(3, 1, 1) if img.shape[0] == 1 else img[:3]
                            image = letterbox_image(img, pre.image_size)
                    graph = build_graph_from_detections(gpos, typ, pre, a.device)
                    if needs_image:
                        graph.image = image
                    graph = graph.to(a.device)
                    emb = gat(graph)
                    jt = graph.joint_types
                    k_lb = min(k_lower_bound(jt), n)
                    ks = {"lb_tha": k_lb, "oracle_tha": max(1, min(n_gt, n))}
                    if k_head is not None:
                        k_pred = max(1, min(int(k_head.predict(emb)), n))
                        ks.update(pred_knn=k_pred, clamp_tha=max(k_pred, k_lb))
                    labels, cache = {}, {}
                    for r, k in ks.items():
                        rule = "knn" if r.endswith("knn") else "tha"
                        if (rule, k) not in cache:
                            cache[(rule, k)] = (predict_knn(emb, k) if rule == "knn"
                                                else predict_tha(emb, k, jt)).cpu().numpy()
                        labels[r] = cache[(rule, k)]
                for r, lab in labels.items():
                    key = f"{name}/{r}"
                    if key in results:
                        results[key] += assemble_instances(pos, typ, conf, lab, img_id)
                        kstats[key].append(ks[r])
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(img_ids)} ({time.time() - t0:.0f} s)", flush=True)

    subsets = {"all": list(img_ids),
               "n_gt 1-5": [x for x in img_ids if 1 <= n_gt_of[x] <= 5],
               "n_gt 2-5": [x for x in img_ids if 2 <= n_gt_of[x] <= 5],
               "n_gt >=6": [x for x in img_ids if n_gt_of[x] >= 6]}
    summary = {"protocol": "eval_e2e_ap.py protocol on its HigherHRNet cache; pooled joints conf > "
                           f"{a.conf_threshold}; models: {a.model}",
               "k_mean": {r: float(np.mean(v)) for r, v in kstats.items() if v},
               "coco": {s: {r: coco_eval(coco, results[r], ids) for r in rows} for s, ids in subsets.items()},
               "n_images": {s: len(ids) for s, ids in subsets.items()}}
    json.dump(summary, open(a.out_dir / "summary.json", "w"), indent=2)
    for r in rows:
        json.dump(results[r], open(a.out_dir / f"results_{r.replace('/', '__')}.json", "w"))

    for s, ids in subsets.items():
        print(f"\n[{s}] {len(ids)} images")
        print(f"{'method':<22s}" + "".join(f"{n:>7s}" for n in COCO_STAT_NAMES) + "   K mean")
        for r in rows:
            km = summary["k_mean"].get(r)
            print(f"{r:<22s}" + "".join(f"{summary['coco'][s][r][n]:7.3f}" for n in COCO_STAT_NAMES)
                  + (f"   {km:.2f}" if km is not None else ""))
    print(f"\nSaved {a.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
