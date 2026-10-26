"""Bone-validity diagnostic, detections vs GT keypoints (2026-10-19; restored from the session scratch script).

For a PG-GAT-H checkpoint: AUC / accuracy of the bone validity logits (label = same person), top-1
correctness per (joint, neighbour type) slot when the true neighbour exists, and how often the top
candidate is scored valid when the true neighbour is absent ("orphan" slots). On the cached E2E
scenes and on COCO val GT keypoints.

    python -m analysis.bone_validity --checkpoint outputs/finetune_pg_gat_hier_coco/c7b02e4a/best.pt --out <json>
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from coco_adapter import CocoAdapter
from dataloader import create_dataloader
from dataset import PoseDataset
from eval_end_to_end import build_graph_from_detections
from eval_peng_gap import load_models


def collect(gat, graphs_labels):
    ys, ss, orphan_hi, orphan_n, top1_ok, top1_n = [], [], 0, 0, 0, 0
    for g, pl in graphs_labels:
        with torch.no_grad():
            gat(g)
        a = gat._aux
        if a.get("bone_logits") is None:
            continue
        src, dst, s = a["bone_src"], a["bone_dst"], a["bone_logits"]
        ys.append((pl[src] == pl[dst]).cpu().numpy()); ss.append(s.cpu().numpy())
        jt = g.joint_types
        bt = gat.bone_type[jt[src], jt[dst]]
        for end, other in ((src, dst), (dst, src)):
            key = end * 64 + bt
            for k in torch.unique(key):
                m = key == k
                i = int(end[m][0]); best = int(torch.argmax(s[m]))
                o = other[m]; e_type = int(jt[o[0]])
                true_exists = bool(((jt == e_type) & (pl == pl[i])).any())
                if true_exists:
                    top1_n += 1; top1_ok += bool(pl[o[best]] == pl[i])
                else:
                    orphan_n += 1; orphan_hi += bool(s[m][best] > 0)
    y = np.concatenate(ys); s = np.concatenate(ss)
    return {"bones": int(len(y)), "pos_rate": float(y.mean()), "auc": float(roc_auc_score(y, s)),
            "acc@0": float(((s > 0) == y).mean()), "top1_when_true_present": top1_ok / max(top1_n, 1),
            "orphan_slots": orphan_n, "orphan_frac": orphan_n / max(orphan_n + top1_n, 1),
            "orphan_top_scored_valid": orphan_hi / max(orphan_n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--arrays", type=Path, default=Path("outputs/e2e_per_image/arrays"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    gat, _, pre = load_models(a.checkpoint, None, a.device)
    e2e = []
    for f in sorted(a.arrays.glob("*.npz")):
        d = np.load(f); idx = d["matched_idx"]
        g = build_graph_from_detections(d["det_pos"][idx], d["det_types"][idx], pre, a.device)
        if g is not None:
            e2e.append((g.to(a.device), torch.tensor(d["gt_person"], device=a.device)))
    ad = CocoAdapter(img_dir=Path("data/coco2017/val2017"),
                     ann_file=Path("data/coco2017/annotations/person_keypoints_val2017.json"), use_depth=False)
    gt = []
    for batch in create_dataloader(PoseDataset(ad), batch_size=4, shuffle=False, num_workers=4):
        for g in pre.process_batch(batch):
            g = g.to(a.device); gt.append((g, g.person_labels))
    out = {"e2e_detections": collect(gat, e2e), "coco_gt_keypoints": collect(gat, gt)}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
