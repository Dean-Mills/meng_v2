"""
Error anatomy of the PEng pipeline vs a detector's native grouping (diagnostics pass, 2026-10-26).

For every matched detection in an eval_peng_gap-style cache, records whether it is grouped correctly
(Hungarian cluster-to-person mapping of evaluator.compute_pga, as analyze_gap_sources.correct_mask) by
  native   the detector's own grouping (ae_ids)
  <model>/eff    THA at the true K of the graph (embedding + assignment errors only)
  <model>/auto   THA at the autonomous K (clamp with a K-head, else K_lb)
together with joint attributes:
  joint type; GT visibility of the matched GT keypoint (1 occluded / 2 visible); person scale
  (sqrt of the person's labelled-keypoint bounding-box area, px); crowding = distance to the nearest
  detection of the same type belonging to another GT person / person scale; localisation error =
  detection-to-GT distance / person scale; scene K_gt and whether the autonomous K equals the true K.
Error rates are joint-weighted (not the per-scene PGA mean). CPU, seeded.

    python -m analysis.error_anatomy --arrays outputs/e2e_cache_yolo/arrays --det yolo \
        --model best=outputs/finetune_pg_gat_hier_img_crowd_coco/d4422c37/best.pt,outputs/k_head_aug_d4422c37/88389153/best.pt \
        --model noimg=outputs/finetune_pg_gat_hier_aug_coco/b59d602d/best.pt --out_dir outputs/diagnostics/anatomy_yolo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_gap_sources import correct_mask
from eval_end_to_end import build_graph_from_detections
from eval_hungarian_grouping import predict_tha
from eval_k_selection import k_lower_bound
from eval_peng_gap import letterbox, load_models
from k_head_hist import predict_k

JOINTS = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri",
          "l_hip", "r_hip", "l_knee", "r_knee", "l_ank", "r_ank"]
BUCKETS = {
    "scale_px": [0, 64, 128, 256, 1e9],
    "crowding": [0, 0.1, 0.25, 0.5, 1e9],
    "loc_err": [0, 0.02, 0.05, 1e9],
    "k_gt": [1, 2, 4, 6, 9, 1e9],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--model", action="append", required=True, help="NAME=CKPT[,K_HEAD]")
    ap.add_argument("--coco_img_dir", type=Path, default=Path("data/coco2017/val2017"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_images", type=int, default=None)
    a = ap.parse_args()
    np.random.seed(0); torch.manual_seed(0)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    models = {}
    for spec in a.model:
        name, paths = spec.split("=", 1)
        ck, *kh = paths.split(",")
        models[name] = load_models(Path(ck), Path(kh[0]) if kh else None, a.device)
    sizes = {}
    with open(a.arrays.parent / "per_image.jsonl") as fh:
        for line in fh:
            r = json.loads(line); sizes[r["image_id"]] = (r["width"], r["height"])
    from torchvision.io import read_image
    from image_evidence import letterbox_image

    cols = {k: [] for k in ("image_id", "type", "vis", "scale_px", "crowding", "loc_err", "k_gt", "k_eff", "native")}
    for name in models:
        cols[f"{name}/eff"] = []; cols[f"{name}/auto"] = []; cols[f"{name}/k_auto_ok"] = []
    files = sorted(a.arrays.glob("*.npz"), key=lambda p: int(p.stem))[: a.max_images]
    with torch.no_grad():
        for i, f in enumerate(files):
            d = np.load(f)
            m_idx, gt = d["matched_idx"], d["gt_person"]
            pos_px, typ = d["det_pos"][m_idx], d["det_types"][m_idx]
            gk = d["gt_kps"]                                   # [P, 17, 4] (x, y, z, v)
            n = len(m_idx)
            if n < 2:
                continue
            k_eff = len(np.unique(gt))
            # joint attributes
            scale = np.zeros(len(gk))
            for p in range(len(gk)):
                lab = gk[p, :, 3] > 0
                xy = gk[p, lab, :2]
                scale[p] = np.sqrt(max((np.ptp(xy[:, 0]) + 1) * (np.ptp(xy[:, 1]) + 1), 1.0)) if lab.sum() >= 2 else 1.0
            s = scale[gt]
            vis = gk[gt, typ, 3]
            loc = np.linalg.norm(pos_px - gk[gt, typ, :2], axis=1) / s
            crowd = np.full(n, np.inf)
            for t in np.unique(typ):
                ii = np.where(typ == t)[0]
                if len(ii) < 2:
                    continue
                dd = np.linalg.norm(pos_px[ii][:, None] - pos_px[ii][None], axis=2)
                same = gt[ii][:, None] == gt[ii][None]
                dd[same] = np.inf
                crowd[ii] = dd.min(axis=1)
            crowd = crowd / s
            per = {"native": correct_mask(d["ae_ids"][m_idx], gt)}
            image = None
            for name, (gat, k_head, pre) in models.items():
                needs = getattr(gat, "needs_image", False)
                gpos = letterbox(pos_px, *sizes[int(f.stem)], pre.image_size) if needs else pos_px
                g = build_graph_from_detections(gpos, typ, pre, a.device)
                if needs:
                    if image is None:
                        img = read_image(str(a.coco_img_dir / str(d["file_name"])))
                        image = letterbox_image(img.repeat(3, 1, 1) if img.shape[0] == 1 else img[:3], pre.image_size)
                    g.image = image
                g = g.to(a.device)
                emb = gat(g); jt = g.joint_types
                k_lb = k_lower_bound(jt)
                k_auto = max(max(1, predict_k(k_head, emb, jt)), k_lb) if k_head is not None else k_lb
                k_auto = int(min(max(k_auto, 1), n))
                lab_eff = predict_tha(emb, min(k_eff, n), jt).cpu().numpy()
                lab_auto = lab_eff if k_auto == k_eff else predict_tha(emb, k_auto, jt).cpu().numpy()
                per[f"{name}/eff"] = correct_mask(lab_eff, gt)
                per[f"{name}/auto"] = correct_mask(lab_auto, gt)
                per[f"{name}/k_auto_ok"] = np.full(n, k_auto == k_eff)
            for k, v in (("image_id", np.full(n, int(f.stem))), ("type", typ), ("vis", vis), ("scale_px", s),
                         ("crowding", crowd), ("loc_err", loc), ("k_gt", np.full(n, int(d["K_gt"]))), ("k_eff", np.full(n, k_eff))):
                cols[k].append(v)
            for k, v in per.items():
                cols[k].append(v)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(files)}", flush=True)
    C = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez_compressed(a.out_dir / "joints.npz", **C)

    methods = ["native"] + [k for k in C if k.endswith("/eff") or k.endswith("/auto")]
    err = lambda mask: {m: float(1 - C[m][mask].mean()) for m in methods} | {"n": int(mask.sum())}
    out = {"det": a.det, "n_joints": int(len(C["type"])), "overall": err(np.ones_like(C["type"], bool))}
    out["by_type"] = {JOINTS[t]: err(C["type"] == t) for t in range(17)}
    out["by_vis"] = {"occluded (v=1)": err(C["vis"] == 1), "visible (v=2)": err(C["vis"] == 2)}
    for key, edges in BUCKETS.items():
        out[f"by_{key}"] = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (C[key] >= lo) & (C[key] < hi)
            if mask.sum():
                out[f"by_{key}"][f"[{lo:g}, {hi:g})"] = err(mask)
    out["by_crowding"]["no same-type neighbour"] = err(~np.isfinite(C["crowding"]))
    for name in models:
        ok = C[f"{name}/k_auto_ok"]
        out[f"{name}/by_k_correct"] = {"K right": err(ok), "K wrong": err(~ok)}
    json.dump(out, open(a.out_dir / "summary.json", "w"), indent=2)

    def show(title, d):
        print(f"\n{title}")
        print(f"  {'bucket':<26}{'n':>8}" + "".join(f"{m:>16}" for m in methods))
        for b, r in d.items():
            print(f"  {b:<26}{r['n']:>8}" + "".join(f"{r[m]:>16.4f}" for m in methods))
    show("overall error rate", {"all": out["overall"]})
    for k in ("by_type", "by_vis", "by_scale_px", "by_crowding", "by_loc_err", "by_k_gt") + tuple(f"{n}/by_k_correct" for n in models):
        show(k, out[k])


if __name__ == "__main__":
    main()
