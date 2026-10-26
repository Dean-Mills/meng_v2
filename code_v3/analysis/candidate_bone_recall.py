"""Candidate-bone recall on cached E2E scenes (2026-10-19; restored from the session scratch script).

For every matched detection and every skeletally adjacent joint type: is the same-person
detection of that type (if detected) among the m nearest detections of that type?
Reports recall and candidate bones per scene for m in (1, 2, 3, 4, 6), for the 16-edge
(sa_gat) and 19-edge skeletons, plus the number of triplet types of each skeleton.

    python -m analysis.candidate_bone_recall --arrays outputs/e2e_per_image/arrays --out <json>
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

SK16 = [(0,1),(0,2),(1,3),(2,4),(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
SK19 = SK16 + [(1,2),(3,5),(4,6)]
MS = (1, 2, 3, 4, 6)


def triplet_types(sk):
    adj = defaultdict(set)
    for a, b in sk:
        adj[a].add(b); adj[b].add(a)
    return sum(len(adj[B]) * (len(adj[B]) - 1) // 2 for B in range(17))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arrays", type=Path, default=Path("outputs/e2e_per_image/arrays"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    files = sorted(a.arrays.glob("*.npz"))
    out = {"n_scenes": len(files), "triplet_types": {"coco16": triplet_types(SK16), "coco19": triplet_types(SK19)}}
    for name, sk in (("coco16", SK16), ("coco19", SK19)):
        adj = defaultdict(set)
        for x, y in sk:
            adj[x].add(y); adj[y].add(x)
        hits = {m: 0 for m in MS}; total = 0; ncand = {m: [] for m in MS}; njoint, npeople = [], []
        for f in files:
            d = np.load(f); idx = d["matched_idx"]; pos = d["det_pos"][idx]; t = d["det_types"][idx]; pid = d["gt_person"]
            njoint.append(len(idx)); npeople.append(len(np.unique(pid)))
            cand = {m: set() for m in MS}
            for i in range(len(idx)):
                for b in adj[t[i]]:
                    js = np.where(t == b)[0]
                    if len(js) == 0:
                        continue
                    order = js[np.argsort(np.linalg.norm(pos[js] - pos[i], axis=1))]
                    for m in MS:
                        for j in order[:m]:
                            cand[m].add(tuple(sorted((i, j))))
                    true = [j for j in js if pid[j] == pid[i]]
                    if not true:
                        continue
                    total += 1
                    rank = list(order).index(true[0])
                    for m in MS:
                        hits[m] += rank < m
            for m in MS:
                ncand[m].append(len(cand[m]))
        out[name] = {"slots_with_true_neighbour": total, "mean_joints": float(np.mean(njoint)), "max_joints": int(max(njoint)),
                     **{f"m{m}": {"recall": hits[m] / total, "bones_mean": float(np.mean(ncand[m])),
                                  "bones_p99": float(np.percentile(ncand[m], 99)), "bones_max": int(max(ncand[m]))} for m in MS}}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
