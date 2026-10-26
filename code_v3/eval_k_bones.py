"""
K from bone validity (PEng): count people as connected pieces of the valid-bone graph.

PG-GAT-H scores every candidate bone (joint pair of skeletally adjacent types)
for "same person" (AUC 0.999 on the E2E scenes). Joints linked by bones with
validity > tau form connected pieces; a person is one piece per skeleton
component it shows. With the 16-edge table the head (0-4) has no bone to the
body, so head and body pieces are counted separately and the larger count is
used. Pieces from a person whose connecting joint was not detected (e.g. a lone
forearm without its shoulder) over-count, so variants anchor the count:

    comp     max(#body pieces, #head pieces)
    comp2    same, pieces with >= 2 joints only
    torso    #pieces containing a torso joint (shoulders / hips), max with #head pieces
             containing the nose or an eye

Each estimate is combined with the hard lower bound, K = max(estimate, K_lb),
and compared with K_lb alone, the MEng K-head + clamp, and the true K of the
graph (eff). Every K is scored with THA on two embeddings, so the K effect is
read with the embedding held fixed: the headline PG-GAT and PG-GAT-H itself.

    python eval_k_bones.py \
        --bones_checkpoint outputs/finetune_pg_gat_hier_coco/c7b02e4a/best.pt \
        --headline outputs/pg_gat_finetune_sweep/3362f752/best.pt \
        --k_head outputs/k_head_meng_headline_coco/a636b1c4/best.pt \
        --out outputs/peng_gap/k_bones_c7b02e4a.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from eval_end_to_end import build_graph_from_detections
from eval_hungarian_grouping import predict_tha
from eval_k_selection import k_lower_bound
from eval_peng_gap import load_models
from evaluator import compute_pga

HEAD = {0, 1, 2, 3, 4}
HEAD_ANCHOR = {0, 1, 2}
TORSO = {5, 6, 11, 12}


def piece_counts(n: int, src: np.ndarray, dst: np.ndarray, valid: np.ndarray, jt: np.ndarray) -> dict:
    s, d = src[valid], dst[valid]
    adj = coo_matrix((np.ones(len(s)), (s, d)), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    head = np.isin(jt, list(HEAD))
    out = {}
    for name, min_size in (("comp", 1), ("comp2", 2)):
        cnt = []
        for part in (~head, head):
            labs, sizes = np.unique(lab[part], return_counts=True)
            cnt.append(int((sizes >= min_size).sum()))
        out[name] = max(cnt)
    torso = len(np.unique(lab[np.isin(jt, list(TORSO))]))
    head_anchor = len(np.unique(lab[np.isin(jt, list(HEAD_ANCHOR))]))
    out["torso"] = max(torso, head_anchor)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bones_checkpoint", type=Path, required=True, help="PG-GAT-H checkpoint (bone scorer)")
    ap.add_argument("--headline", type=Path, required=True, help="flat PG-GAT checkpoint (second embedding)")
    ap.add_argument("--k_head", type=Path, required=True, help="K-head of the headline (MEng baseline K)")
    ap.add_argument("--arrays", type=Path, default=Path("outputs/e2e_per_image/arrays"))
    ap.add_argument("--tau", type=float, nargs="+", default=[0.5, 0.8])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    hier, _, pre_h = load_models(a.bones_checkpoint, None, a.device)
    flat, k_head, pre_f = load_models(a.headline, a.k_head, a.device)
    embs = {"headline": (flat, pre_f), "hier": (hier, pre_h)}

    rows = []
    files = sorted(a.arrays.glob("*.npz"), key=lambda p: int(p.stem))
    with torch.no_grad():
        for i, f in enumerate(files):
            d = np.load(f)
            m_idx, gt = d["matched_idx"], d["gt_person"]
            pos, typ = d["det_pos"][m_idx], d["det_types"][m_idx]
            g_h = build_graph_from_detections(pos, typ, pre_h, a.device)
            if g_h is None:
                continue
            g_h = g_h.to(a.device)
            e_h = hier(g_h)
            aux = hier._aux
            n = len(m_idx)
            jt = g_h.joint_types
            k_lb = k_lower_bound(jt)
            ks = {"lb": k_lb, "eff": len(np.unique(gt))}
            g_f = build_graph_from_detections(pos, typ, pre_f, a.device).to(a.device)
            e_f = flat(g_f)
            ks["clamp"] = max(max(1, k_head.predict(e_f)), k_lb)
            if aux.get("bone_logits") is not None:
                src, dst = aux["bone_src"].cpu().numpy(), aux["bone_dst"].cpu().numpy()
                p = torch.sigmoid(aux["bone_logits"]).cpu().numpy()
                for tau in a.tau:
                    for name, k in piece_counts(n, src, dst, p > tau, jt.cpu().numpy()).items():
                        ks[f"{name}@{tau}"] = max(k, k_lb)
            else:
                for tau in a.tau:
                    for name in ("comp", "comp2", "torso"):
                        ks[f"{name}@{tau}"] = k_lb
            ks = {s: int(min(max(k, 1), n)) for s, k in ks.items()}

            rec = {"image_id": int(f.stem), "K_eff": ks["eff"], "k": ks}
            gt_t = torch.tensor(gt)
            for ename, e in (("headline", e_f), ("hier", e_h)):
                per_k = {k: float(compute_pga(predict_tha(e, k, jt), gt_t)) for k in sorted(set(ks.values()))}
                rec[ename] = {s: per_k[k] for s, k in ks.items()}
            rows.append(rec)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)

    sels = list(rows[0]["k"].keys())
    k_eff = np.array([r["K_eff"] for r in rows])
    multi = k_eff >= 2
    summary = {"n": len(rows), "n_multi": int(multi.sum())}
    print(f"\n{len(rows)} scenes ({int(multi.sum())} with >= 2 people in the graph); THA PGA per K selector")
    print(f"{'selector':<12}{'K mean':>8}{'K=eff':>8}{'over':>7}{'under':>7}"
          f"{'headline':>10}{'hier':>9}{'hier multi':>12}")
    for s in sels:
        k = np.array([r["k"][s] for r in rows])
        res = {
            "k_mean": float(k.mean()), "k_exact": float((k == k_eff).mean()),
            "over": float((k > k_eff).mean()), "under": float((k < k_eff).mean()),
            "tha_headline": float(np.mean([r["headline"][s] for r in rows])),
            "tha_hier": float(np.mean([r["hier"][s] for r in rows])),
            "tha_hier_multi": float(np.mean([r["hier"][s] for r, m in zip(rows, multi) if m])),
            "tha_headline_multi": float(np.mean([r["headline"][s] for r, m in zip(rows, multi) if m])),
        }
        summary[s] = res
        print(f"{s:<12}{res['k_mean']:>8.2f}{res['k_exact']:>8.1%}{res['over']:>7.1%}{res['under']:>7.1%}"
              f"{res['tha_headline']:>10.4f}{res['tha_hier']:>9.4f}{res['tha_hier_multi']:>12.4f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=2)
    with open(a.out.with_suffix(".jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
