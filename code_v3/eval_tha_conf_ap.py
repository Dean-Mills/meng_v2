"""
AP@OKS for confidence-aware THA (tha_conf.py) over a grid of lam (PEng, 2026-10-29).

eval_e2e_ap_peng.py protocol and helpers unchanged (every val2017 image, pooled detections above --conf_threshold,
one instance per cluster with the highest-confidence detection per type, instance score = mean confidence over 17
slots), one model, rows K_lb THA (lb) and, with a K-head, clamp THA (K = max(K-head, K_lb)), each at every lam.
lam = 0 must reproduce eval_e2e_ap_peng's <name>__{lb,clamp}_tha files exactly (checked by --check_dir).

lam is chosen without the reported numbers: images are split into two folds by position in the sorted id list
(fold = index % 2); summary.json holds AP per lam on each fold, and analysis/tha_conf_cv.py picks lam on one fold,
scores the other, and assembles the cross-fitted all-image results.

    python eval_tha_conf_ap.py --model crowd=outputs/finetune_pg_gat_hier_img_crowd_coco/d4422c37/best.pt,outputs/k_head_aug_d4422c37/88389153/best.pt \
        --cache_dir outputs/e2e_ap/hrnet_cache --conf_threshold 0 --out_dir outputs/tha_conf/crowd_hrnet_t0 \
        --check_dir outputs/verify_20261026/ap/hrnet_t0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval_e2e_ap import CONF_THRESHOLD, assemble_instances, coco_eval, pool_detections
from eval_end_to_end import build_graph_from_detections
from eval_k_selection import k_lower_bound
from eval_peng_gap import letterbox, load_models
from k_head_hist import predict_k
from tha_conf import predict_tha_conf

LAMS = (0.0, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="NAME=CHECKPOINT[,K_HEAD]")
    ap.add_argument("--cache_dir", type=Path, required=True)
    ap.add_argument("--coco_img_dir", type=Path, default=Path("data/coco2017/val2017"))
    ap.add_argument("--coco_ann_file", type=Path,
                    default=Path("data/coco2017/annotations/person_keypoints_val2017.json"))
    ap.add_argument("--conf_threshold", type=float, default=CONF_THRESHOLD)
    ap.add_argument("--lams", type=float, nargs="+", default=list(LAMS))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--check_dir", type=Path, default=None,
                    help="eval_e2e_ap_peng output dir; lam 0 must equal its results_<name>__<row>_tha.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    name, paths = a.model.split("=", 1)
    ck, *kh = paths.split(",")
    gat, k_head, pre = load_models(Path(ck), Path(kh[0]) if kh else None, a.device)
    needs_image = getattr(gat, "needs_image", False)

    from pycocotools.coco import COCO
    from torchvision.io import read_image
    from image_evidence import letterbox_image
    coco = COCO(str(a.coco_ann_file))
    cat_ids = coco.getCatIds(catNms=["person"])
    img_ids = sorted(coco.getImgIds())
    if a.max_images:
        img_ids = img_ids[:a.max_images]

    rows = ["lb"] + (["clamp"] if k_head is not None else [])
    results = {(r, lam): [] for r in rows for lam in a.lams}
    n_gt_of = {}
    t0 = time.time()
    with torch.no_grad():
        for i, img_id in enumerate(img_ids):
            joints = np.load(a.cache_dir / f"{img_id}.npy")
            info = coco.loadImgs(img_id)[0]
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False))
            n_gt_of[img_id] = sum(1 for x in anns if x.get("num_keypoints", 0) > 0)
            pos, typ, conf, _ = pool_detections(joints, a.conf_threshold)
            n = len(pos)
            if n == 0:
                continue
            if n == 1:
                labels = {(r, lam): np.zeros(1, int) for r in rows for lam in a.lams}
            else:
                gpos = pos
                image = None
                if needs_image:
                    gpos = letterbox(pos, info["width"], info["height"], pre.image_size)
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
                ks = {"lb": k_lb}
                if k_head is not None:
                    k_pred = max(1, min(int(predict_k(k_head, emb, jt)), n))
                    ks["clamp"] = max(k_pred, k_lb)
                emb_np, jt_np = emb.cpu().numpy(), jt.cpu().numpy()
                labels, cache = {}, {}
                for r, k in ks.items():
                    if k not in cache:
                        cache[k] = predict_tha_conf(emb_np, k, jt_np, conf, a.lams)
                    for lam in a.lams:
                        labels[(r, lam)] = cache[k][lam]
            for key, lab in labels.items():
                results[key] += assemble_instances(pos, typ, conf, lab, img_id)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(img_ids)} ({time.time() - t0:.0f} s)", flush=True)

    folds = {"fold0": img_ids[0::2], "fold1": img_ids[1::2]}
    subsets = {"all": img_ids, **folds, "ge6": [x for x in img_ids if n_gt_of[x] >= 6],
               "1-5": [x for x in img_ids if 1 <= n_gt_of[x] <= 5]}
    summary = {"model": a.model, "cache": str(a.cache_dir), "conf_threshold": a.conf_threshold, "lams": a.lams,
               "ap": {r: {str(lam): {s: coco_eval(coco, results[(r, lam)], ids)["AP"] for s, ids in subsets.items()}
                          for lam in a.lams} for r in rows},
               "n_images": {s: len(ids) for s, ids in subsets.items()}}
    json.dump(summary, open(a.out_dir / "summary.json", "w"), indent=2)
    for (r, lam), res in results.items():
        json.dump(res, open(a.out_dir / f"results_{r}_lam{lam:g}.json", "w"))
    if a.check_dir is not None:
        keep = set(img_ids)
        for r in rows:
            core = lambda xs: [(x["image_id"], x["keypoints"], x["score"]) for x in xs if x["image_id"] in keep]
            ref = json.load(open(a.check_dir / f"results_{name}__{r}_tha.json"))
            assert core(ref) == core(results[(r, 0.0)]), f"lam 0 differs from {a.check_dir} ({r})"
        print(f"check: lam 0 equals {a.check_dir} on {rows}")
    for r in rows:
        print(f"\n[{r}] AP by lam:   " + "  ".join(f"{s:>7s}" for s in subsets))
        for lam in a.lams:
            print(f"  lam {lam:<7g}" + "  ".join(f"{summary['ap'][r][str(lam)][s]:7.4f}" for s in subsets))
    print(f"\nSaved {a.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
