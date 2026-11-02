"""
PEng gap table for confidence-aware THA (tha_conf.py) on the fixed PGA scene set (2026-10-29).

eval_peng_gap.py's scenes, frame, model loading, K selectors and PGA, THA only, at each --lams value. The cached scenes
hold no detection confidence, so each matched detection's confidence is looked up in the all-image AP cache of the
same detector run (--ap_cache, eval_e2e_ap format [n_people, 17, (y, x, conf)]) by exact (type, x, y); every
detection matches on all three detectors (checked 2026-10-29). per_image.jsonl keeps eval_peng_gap's layout with
methods tha_l<lam> (tha_l0 = predict_tha), so analysis/paired_ci.py reads it, e.g.
    --a P --b P --mode diff --rows clamp:tha_l0.2=clamp:tha_l0      (effect of the confidence term)
    --a P --mode gap --rows clamp:tha_l0.2                          (gap to the native grouping)
--check: tha_l0 must equal the "tha" values of an eval_peng_gap per_image.jsonl on the same scenes.

    python eval_peng_gap_conf.py --checkpoint outputs/finetune_pg_gat_hier_img_crowd_coco/d4422c37/best.pt \
        --k_head outputs/k_head_aug_d4422c37/88389153/best.pt --arrays outputs/e2e_per_image/arrays \
        --ap_cache outputs/e2e_ap/hrnet_cache --out_dir outputs/peng_gap_conf/hrnet_d4422c37 \
        --check outputs/verify_20261026/gap/hrnet_d4422c37_khaug/per_image.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_end_to_end import build_graph_from_detections
from eval_k_selection import k_lower_bound
from eval_peng_gap import letterbox, load_models
from evaluator import compute_pga
from k_head_hist import predict_k
from tha_conf import predict_tha_conf

SELECTORS = ["oracle", "eff", "lb", "clamp"]


def lookup_conf(pos, typ, cache_joints):
    """Confidence of each detection = the AP-cache joint with the same type and (x, y)."""
    table = {}
    for p in range(len(cache_joints)):
        for j in range(17):
            y, x, c = cache_joints[p, j]
            table.setdefault((j, float(x), float(y)), float(c))
    return np.array([table[(int(t), float(x), float(y))] for (x, y), t in zip(pos, typ)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--k_head", type=Path, default=None)
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--ap_cache", type=Path, required=True)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.025, 0.05, 0.1, 0.2])
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--check", type=Path, default=None)
    ap.add_argument("--coco_img_dir", type=Path, default=Path("data/coco2017/val2017"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    gat, k_head, pre = load_models(a.checkpoint, a.k_head, a.device)
    needs_image = getattr(gat, "needs_image", False)
    if needs_image:
        from torchvision.io import read_image
        from image_evidence import letterbox_image
    selectors = [s for s in SELECTORS if k_head is not None or s != "clamp"]
    names = {lam: f"tha_l{lam:g}" for lam in a.lams}

    files = sorted(a.arrays.glob("*.npz"), key=lambda p: int(p.stem))
    if a.max_images:
        files = files[:a.max_images]
    sizes = {}
    with open(a.arrays.parent / "per_image.jsonl") as fm:
        for line in fm:
            r = json.loads(line)
            sizes[r["image_id"]] = (r["width"], r["height"])

    rows = []
    fh = open(a.out_dir / "per_image.jsonl", "w")
    with torch.no_grad():
        for i, f in enumerate(files):
            d = np.load(f)
            m_idx, m_person = d["matched_idx"], d["gt_person"]
            det_pos, det_types, ae_ids = d["det_pos"], d["det_types"], d["ae_ids"]
            raw = det_pos[m_idx]
            conf = lookup_conf(raw, det_types[m_idx], np.load(a.ap_cache / f"{f.stem}.npy"))
            pos = letterbox(raw, *sizes[int(f.stem)], pre.image_size) if needs_image else raw
            graph = build_graph_from_detections(pos, det_types[m_idx], pre, a.device)
            if graph is None:
                continue
            if needs_image:
                img = read_image(str(a.coco_img_dir / str(d["file_name"])))
                img = img.repeat(3, 1, 1) if img.shape[0] == 1 else img[:3]
                graph.image = letterbox_image(img, pre.image_size)
            graph = graph.to(a.device)
            emb = gat(graph)
            jt = graph.joint_types
            gt_labels = torch.tensor(m_person, dtype=torch.long)
            k_lb = k_lower_bound(jt)
            ks = {"oracle": int(d["K_gt"]), "eff": len(np.unique(m_person)), "lb": k_lb}
            if k_head is not None:
                ks["clamp"] = max(max(1, predict_k(k_head, emb, jt)), k_lb)
            n = emb.size(0)
            ks = {s: int(min(max(k, 1), n)) for s, k in ks.items()}
            emb_np, jt_np = emb.cpu().numpy(), jt.cpu().numpy()
            per_k = {}
            for k in sorted(set(ks.values())):
                labs = predict_tha_conf(emb_np, k, jt_np, conf, a.lams)
                per_k[k] = {names[l]: float(compute_pga(torch.tensor(labs[l], dtype=torch.long), gt_labels))
                            for l in a.lams}
            rec = {"image_id": int(f.stem), "n": int(len(m_idx)), "K_gt": int(d["K_gt"]),
                   "ae": float(compute_pga(torch.tensor(ae_ids[m_idx], dtype=torch.long), gt_labels)),
                   **{s: {"k": k, **per_k[k]} for s, k in ks.items()}}
            rows.append(rec)
            fh.write(json.dumps(rec) + "\n")
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(files)}", flush=True)
    fh.close()

    summary = {"n": len(rows), "checkpoint": str(a.checkpoint), "k_head": str(a.k_head), "lams": a.lams,
               "ae": float(np.mean([r["ae"] for r in rows])),
               **{s: {m: float(np.mean([r[s][m] for r in rows])) for m in names.values()} for s in selectors}}
    sel = [r for r in rows if r["K_gt"] >= 6]
    summary["k_gt_ge6"] = {"n": len(sel), "ae": float(np.mean([r["ae"] for r in sel])) if sel else None,
                           **{s: {m: float(np.mean([r[s][m] for r in sel])) if sel else None for m in names.values()}
                              for s in selectors}}
    json.dump(summary, open(a.out_dir / "summary.json", "w"), indent=2)
    if a.check is not None:
        ref = {json.loads(l)["image_id"]: json.loads(l) for l in open(a.check)}
        bad = [(r["image_id"], s) for r in rows for s in selectors
               if r["image_id"] in ref and abs(r[s]["tha_l0"] - ref[r["image_id"]][s]["tha"]) > 1e-12]
        missing = sum(r["image_id"] not in ref for r in rows)
        assert not bad and not missing, f"tha_l0 differs from {a.check}: {len(bad)} cells, {missing} scenes missing"
        print(f"check: tha_l0 equals {a.check} on {len(rows)} scenes x {selectors}")
    print(f"\nn = {len(rows)}   native AE {summary['ae']:.4f}")
    for s in selectors:
        print(f"  {s:<7}" + "  ".join(f"{m} {summary[s][m]:.4f}" for m in names.values()))
    print(f"K_gt >= 6 (n = {summary['k_gt_ge6']['n']}, AE {summary['k_gt_ge6']['ae']:.4f})")
    for s in selectors:
        print(f"  {s:<7}" + "  ".join(f"{m} {summary['k_gt_ge6'][s][m]:.4f}" for m in names.values()))


if __name__ == "__main__":
    main()
