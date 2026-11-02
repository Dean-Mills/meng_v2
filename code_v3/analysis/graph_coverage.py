"""Joint-graph and bone-candidate coverage by crowd size (diagnostics pass, 2026-10-26).

PG-GAT's joint graph is kNN (k = 16 for 256-d models) and PG-GAT-H's bones use the m = 3 nearest
detections per adjacent type, whatever the scene density. Per matched detection, using exactly the
preprocessor's kNN rule (letterboxing is a uniform scale + shift, so neighbourhoods are unchanged):
  same-type coverage   share of the other people's same-type detections inside the joint's kNN neighbourhood
                       (in or out edges) - what the repulsion head can see
  same-person coverage share of the joint's own person's other detections inside the neighbourhood
  bone recall (m=3)    true skeletal neighbour among the 3 nearest detections of that type (if detected)
reported by annotated crowd size K_gt.

    python -m analysis.graph_coverage --arrays outputs/e2e_cache_yolo/arrays --out <json>
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from preprocessor import PosePreprocessor
from sa_gat import COCO_SKELETON

BUCKETS = [(1, 1), (2, 3), (4, 5), (6, 8), (9, 99)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=Path, required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pre = PosePreprocessor(device="cpu", k_neighbors=a.k, use_depth=False)
    adj = defaultdict(set)
    for x, y in COCO_SKELETON:
        adj[x].add(y); adj[y].add(x)
    acc = {b: defaultdict(list) for b in BUCKETS}
    for f in sorted(a.arrays.glob("*.npz")):
        d = np.load(f); idx = d["matched_idx"]; n = len(idx)
        if n < 2:
            continue
        pos = d["det_pos"][idx] / pre.image_size; t = d["det_types"][idx]; pid = d["gt_person"]
        kg = int(d["K_gt"]); b = next(bb for bb in BUCKETS if bb[0] <= kg <= bb[1])
        ei = pre._knn_edges(torch.tensor(pos, dtype=torch.float32), k=a.k).numpy()
        nb = [set() for _ in range(n)]
        for s_, d_ in ei.T:
            nb[s_].add(d_); nb[d_].add(s_)
        for i in range(n):
            st = set(np.where((t == t[i]) & (pid != pid[i]))[0])
            sp = set(np.where((pid == pid[i]) & (np.arange(n) != i))[0])
            if st:
                acc[b]["same_type_cov"].append(len(nb[i] & st) / len(st))
                acc[b]["same_type_any"].append(float(len(nb[i] & st) > 0))
            if sp:
                acc[b]["same_person_cov"].append(len(nb[i] & sp) / len(sp))
            acc[b]["degree"].append(len(nb[i]))
            for nt in adj[t[i]]:
                js = np.where(t == nt)[0]
                true = [j for j in js if pid[j] == pid[i]]
                if js.size and true:
                    order = js[np.argsort(np.linalg.norm(pos[js] - pos[i], axis=1))]
                    acc[b]["bone_recall"].append(float(true[0] in order[: a.m]))
        acc[b]["joints_per_scene"].append(n)
    out = {f"{lo}-{hi}": {k: float(np.mean(v)) for k, v in acc[(lo, hi)].items()} | {"n_joints": len(acc[(lo, hi)]["degree"])}
           for lo, hi in BUCKETS}
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"{'K_gt':<7}{'joints/scene':>13}{'deg':>6}{'same-type cov':>15}{'any same-type':>15}{'same-person cov':>17}{'bone recall':>13}")
    for k, r in out.items():
        print(f"{k:<7}{r.get('joints_per_scene', 0):>13.1f}{r.get('degree', 0):>6.1f}{r.get('same_type_cov', float('nan')):>15.3f}"
              f"{r.get('same_type_any', float('nan')):>15.3f}{r.get('same_person_cov', float('nan')):>17.3f}{r.get('bone_recall', float('nan')):>13.4f}")


if __name__ == "__main__":
    main()
