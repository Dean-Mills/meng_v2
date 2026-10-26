"""Paired bootstrap CIs over scenes for eval_peng_gap.py per-image outputs (restores the inline
bootstrap snippets used 2026-10-19..23: 95 % percentile CI, resampling scenes, numpy seed 0).

Row syntax:  SEL:METHOD            e.g. lb:tha, eff:copkm, clamp:tha, oracle:knn
  --mode gap   mean(native ae - A[SEL][METHOD])                       (gap to the native grouping)
  --mode diff  mean(A[SEL][METHOD] - B[SEL_B][METHOD_B])              (row "SEL:METHOD=SEL_B:METHOD_B" or same on both)
Scenes are the intersection of the image ids present in A (and B). --min_k / --max_k filter by K_gt.

    python -m analysis.paired_ci --a outputs/peng_gap/372658c6_khaug/per_image.jsonl --mode gap --rows clamp:tha clamp:copkm
    python -m analysis.paired_ci --a .../b59d602d/per_image.jsonl --b .../3362f752/per_image.jsonl --mode diff --rows lb:tha eff:tha
"""
import argparse, json
import numpy as np


def load(p):
    return {json.loads(l)["image_id"]: json.loads(l) for l in open(p)}


def ci(d, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b")
    ap.add_argument("--mode", choices=["gap", "diff"], required=True)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--min_k", type=int, default=None); ap.add_argument("--max_k", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    A = load(a.a); B = load(a.b) if a.b else None
    ids = sorted(set(A) & set(B)) if B else sorted(A)
    ids = [i for i in ids if (a.min_k is None or A[i]["K_gt"] >= a.min_k) and (a.max_k is None or A[i]["K_gt"] <= a.max_k)]
    res = []
    for row in a.rows:
        left, _, right = row.partition("=")
        s1, m1 = left.split(":"); s2, m2 = (right.split(":") if right else (s1, m1))
        if a.mode == "gap":
            d = np.array([A[i]["ae"] - A[i][s1][m1] for i in ids])
        else:
            d = np.array([A[i][s1][m1] - B[i][s2][m2] for i in ids])
        mean, lo, hi = ci(d)
        r = {"row": row, "mode": a.mode, "n": len(ids), "mean": mean, "ci_lo": lo, "ci_hi": hi,
             "value_a": float(np.mean([A[i][s1][m1] for i in ids]))}
        res.append(r)
        print(f"{a.mode} {row:<24} n={len(ids):5d}  {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  (A value {r['value_a']:.4f})")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
