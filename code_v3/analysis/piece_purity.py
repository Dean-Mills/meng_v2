"""Purity of valid-bone pieces on cached E2E scenes (2026-10-20; restored from the session scratch script).

Connected components of candidate bones with PG-GAT-H validity > tau: share of joints in pieces
that mix two GT people, singleton share, joints per piece.

    python -m analysis.piece_purity --checkpoint outputs/finetune_pg_gat_hier_coco/c7b02e4a/best.pt --out <json>
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from eval_end_to_end import build_graph_from_detections
from eval_peng_gap import load_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--arrays", type=Path, default=Path("outputs/e2e_per_image/arrays"))
    ap.add_argument("--taus", type=float, nargs="+", default=[0.5, 0.8, 0.95, 0.99])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    hier, _, pre = load_models(a.checkpoint, None, a.device)
    st = {t: dict(pieces=0, impure=0, multi_pieces=0, joints=0, joints_in_impure=0, singletons=0) for t in a.taus}
    with torch.no_grad():
        for f in sorted(a.arrays.glob("*.npz")):
            d = np.load(f); m = d["matched_idx"]; gt = d["gt_person"]
            g = build_graph_from_detections(d["det_pos"][m], d["det_types"][m], pre, a.device)
            if g is None:
                continue
            hier(g.to(a.device)); aux = hier._aux; n = len(m)
            if aux.get("bone_logits") is None:
                continue
            s, dd = aux["bone_src"].cpu().numpy(), aux["bone_dst"].cpu().numpy()
            p = torch.sigmoid(aux["bone_logits"]).cpu().numpy()
            for t in a.taus:
                v = p > t
                _, lab = connected_components(coo_matrix((np.ones(v.sum()), (s[v], dd[v])), shape=(n, n)), directed=False)
                for c in np.unique(lab):
                    idx = lab == c; k = st[t]
                    k["pieces"] += 1; k["joints"] += int(idx.sum()); k["singletons"] += int(idx.sum() == 1)
                    if idx.sum() > 1:
                        k["multi_pieces"] += 1
                    if len(np.unique(gt[idx])) > 1:
                        k["impure"] += 1; k["joints_in_impure"] += int(idx.sum())
    out = {str(t): {**k, "impure_frac_of_multi": k["impure"] / max(k["multi_pieces"], 1),
                    "joints_in_impure_frac": k["joints_in_impure"] / k["joints"],
                    "singleton_frac": k["singletons"] / k["pieces"], "joints_per_piece": k["joints"] / k["pieces"]}
           for t, k in st.items()}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
