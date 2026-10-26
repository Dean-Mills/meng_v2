"""
Where does the PG-GAT vs HigherHRNet-AE gap sit? Split by joint origin.

HigherHRNet's parser (simple-HigherHRNet, refine=True) outputs two kinds of joint:
  * peak   — a heatmap peak assigned to a person by AE tag matching;
  * refine — a joint ADDED by `refine` for a person who lacked that type: the
             heatmap is searched near the person's mean tag. Refine never moves
             existing joints, so these are exactly the refine-on joints with no
             refine-off detection of the same type at the same position.
Refine-added joints are placed by AE grouping, so AE's label for them is
self-consistent by construction; PG-GAT has to infer it from geometry.

Inputs: the refine-on cache (eval_e2e_per_image.py) and the refine-off cache
(eval_e2e_per_image_norefine.py). Every scene of the refine-on cache is scored
(same 2,166-scene set as eval_peng_gap.py). Per-joint correctness uses the
same Hungarian cluster-to-person mapping as evaluator.compute_pga, so the
per-origin contributions add up to the scene PGA exactly.

    python analyze_gap_sources.py \
        --checkpoint outputs/pg_gat_finetune_sweep/3362f752/best.pt \
        --out outputs/peng_gap/gap_sources_3362f752.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from eval_end_to_end import build_graph_from_detections
from eval_hungarian_grouping import predict_tha
from eval_k_selection import k_lower_bound
from eval_peng_gap import load_models

MATCH_PX = 1.0  # refine-on vs refine-off position tolerance (identical peaks up to GPU noise)


def correct_mask(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-joint correctness under compute_pga's Hungarian mapping."""
    up, ug = np.unique(pred), np.unique(gt)
    k = max(len(up), len(ug))
    pr = {v: i for i, v in enumerate(up)}
    gr = {v: i for i, v in enumerate(ug)}
    conf = np.zeros((k, k), dtype=np.int64)
    for p, g in zip(pred, gt):
        conf[pr[p], gr[g]] += 1
    row, col = linear_sum_assignment(-conf)
    to_gt = dict(zip(row, col))
    return np.array([to_gt.get(pr[p], -1) == gr[g] for p, g in zip(pred, gt)])


def joint_origin(on: dict, off: dict) -> np.ndarray:
    """True where a matched refine-on joint is a refine-off peak (same type, same position)."""
    pos, typ = on["det_pos"][on["matched_idx"]], on["det_types"][on["matched_idx"]]
    is_peak = np.zeros(len(pos), dtype=bool)
    if off is None or len(off["det_pos"]) == 0:
        return is_peak
    for t in np.unique(typ):
        a = np.where(typ == t)[0]
        b = off["det_pos"][off["det_types"] == t]
        if len(b) == 0:
            continue
        d = np.linalg.norm(pos[a][:, None, :] - b[None, :, :], axis=2)
        is_peak[a] = d.min(axis=1) <= MATCH_PX
    return is_peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--arrays", type=Path, default=Path("outputs/e2e_per_image/arrays"))
    ap.add_argument("--norefine_arrays", type=Path, default=Path("outputs/e2e_per_image_norefine/arrays"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    gat, _, pre = load_models(a.checkpoint, None, a.device)
    methods = ["ae", "pg_lb", "pg_eff"]  # AE; PG-GAT THA at K_lb (autonomous) and at true K of the graph
    # per-image contributions: sum over joints of a subset / n joints in the image
    contrib = {m: {"peak": [], "refine": []} for m in methods}
    n_peak = n_refine = 0
    acc = {m: {"peak": [0, 0], "refine": [0, 0]} for m in methods}  # [correct, total] joint-level
    multi = []
    files = sorted(a.arrays.glob("*.npz"), key=lambda p: int(p.stem))
    with torch.no_grad():
        for i, f in enumerate(files):
            on = np.load(f)
            off_f = a.norefine_arrays / f.name
            off = dict(np.load(off_f)) if off_f.exists() else None
            m_idx, gt = on["matched_idx"], on["gt_person"]
            graph = build_graph_from_detections(on["det_pos"][m_idx], on["det_types"][m_idx], pre, a.device)
            if graph is None:
                continue
            graph = graph.to(a.device)
            emb = gat(graph)
            n = len(m_idx)
            preds = {
                "ae": on["ae_ids"][m_idx],
                "pg_lb": predict_tha(emb, min(k_lower_bound(graph.joint_types), n), graph.joint_types).cpu().numpy(),
                "pg_eff": predict_tha(emb, min(len(np.unique(gt)), n), graph.joint_types).cpu().numpy(),
            }
            is_peak = joint_origin(dict(on), off)
            n_peak += int(is_peak.sum())
            n_refine += int((~is_peak).sum())
            multi.append(len(np.unique(gt)) >= 2)
            for m, p in preds.items():
                c = correct_mask(p, gt)
                for name, mask in (("peak", is_peak), ("refine", ~is_peak)):
                    contrib[m][name].append(float(c[mask].sum()) / n)
                    acc[m][name][0] += int(c[mask].sum())
                    acc[m][name][1] += int(mask.sum())
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)

    multi = np.array(multi)
    out = {"n_images": len(multi), "n_multi": int(multi.sum()),
           "joints_peak": n_peak, "joints_refine": n_refine,
           "refine_fraction": n_refine / max(n_peak + n_refine, 1)}
    for subset, sel in (("all", np.ones_like(multi)), ("multi", multi)):
        for m in methods:
            pk = np.array(contrib[m]["peak"])[sel]
            rf = np.array(contrib[m]["refine"])[sel]
            out[f"{subset}/{m}/pga"] = float((pk + rf).mean())
            out[f"{subset}/{m}/contrib_peak"] = float(pk.mean())
            out[f"{subset}/{m}/contrib_refine"] = float(rf.mean())
    for m in methods:
        for name in ("peak", "refine"):
            c, t = acc[m][name]
            out[f"joint_acc/{m}/{name}"] = c / max(t, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)

    print(f"\n{out['n_images']} scenes ({out['n_multi']} multi-person); matched joints: "
          f"{n_peak} peak, {n_refine} refine-added ({out['refine_fraction']:.1%})")
    print(f"{'':10}{'joint acc peak':>16}{'joint acc refine':>18}")
    for m in methods:
        print(f"{m:10}{out[f'joint_acc/{m}/peak']:>16.4f}{out[f'joint_acc/{m}/refine']:>18.4f}")
    for subset in ("all", "multi"):
        print(f"\n[{subset}] PGA and gap to AE split by joint origin (contributions sum to the PGA gap)")
        for m in ("pg_lb", "pg_eff"):
            g_pk = out[f"{subset}/ae/contrib_peak"] - out[f"{subset}/{m}/contrib_peak"]
            g_rf = out[f"{subset}/ae/contrib_refine"] - out[f"{subset}/{m}/contrib_refine"]
            print(f"  {m:7} PGA {out[f'{subset}/{m}/pga']:.4f} vs AE {out[f'{subset}/ae/pga']:.4f}: "
                  f"gap {g_pk + g_rf:+.4f} = peak {g_pk:+.4f} + refine {g_rf:+.4f}")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
