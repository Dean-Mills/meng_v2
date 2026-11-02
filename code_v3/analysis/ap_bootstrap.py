"""Paired image-bootstrap CIs for COCO keypoint AP on eval_e2e_ap_peng.py results_*.json files (2026-10-26).

COCOeval.evaluate() runs once per results file (every val2017 image, the eval_e2e_ap_peng protocol); each draw then
resamples images (numpy seed 0, 5000 draws, 95 % percentile CI, as analysis/paired_ci.py) and recomputes AP
(stats[0]: OKS 0.50:0.95, area all, maxDets 20) from the per-image matches with the COCOeval.accumulate arithmetic.
Matching is per image, so an image drawn k times = its detections and non-ignored GT weighted by k; nothing is
re-matched. With every weight 1 the point estimate is COCOeval's AP (checked against a fresh COCOeval per subset).
A and B see the same draws, so the CI is on AP(A) - AP(B); use --a native --b ours for the AP gap.

Interval = BCa (bias-corrected and accelerated; acceleration from the leave-one-image-out jackknife), not the plain
percentile: AP is biased under resampling (the max-interpolated precision envelope rises with a noisier PR curve) and
by different amounts per method (HigherHRNet t0: native +0.0011, PG-GAT-H +0.0000 over 1,000 draws), which shifts a
percentile interval. The percentile interval and the bootstrap bias are kept in the JSON.

Subsets by annotated people with keypoints (as eval_e2e_ap_peng): all, 1-5, 2-5, ge6.

    python -m analysis.ap_bootstrap --a outputs/verify_20261026/ap/hrnet_t0/results_native_full.json \
        --b outputs/verify_20261026/ap/hrnet_t0/results_crowd__lb_tha.json --subset all ge6
"""
import argparse, io, json
from contextlib import redirect_stdout
from statistics import NormalDist
import numpy as np

ANN = "data/coco2017/annotations/person_keypoints_val2017.json"
SUBSETS = {"all": lambda n: True, "1-5": lambda n: 1 <= n <= 5, "2-5": lambda n: 2 <= n <= 5, "ge6": lambda n: n >= 6}


def per_image(coco, path):
    """COCOeval matches for area 'all': sorted scores, img index, tp / fp [T, N], non-ignored GT per image."""
    from pycocotools.cocoeval import COCOeval
    with redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(json.load(open(path))), iouType="keypoints")
        ev.params.imgIds = sorted(coco.getImgIds())
        ev.evaluate()
    p = ev.params
    assert len(p.catIds) == 1 and p.areaRngLbl[0] == "all"
    T, md = len(p.iouThrs), p.maxDets[-1]
    sc, im, tp, fp, npig = [], [], [], [], np.zeros(len(p.imgIds))
    for i, e in enumerate(ev.evalImgs[:len(p.imgIds)]):          # catId x areaRng x imgId: first block = area all
        if e is None:
            continue
        npig[i] = np.count_nonzero(e["gtIgnore"] == 0)
        m, ig = e["dtMatches"][:, :md] > 0, e["dtIgnore"][:, :md].astype(bool)
        sc.append(np.asarray(e["dtScores"][:md])); im.append(np.full(m.shape[1], i))
        tp.append(m & ~ig); fp.append(~m & ~ig)
    o = np.argsort(-np.concatenate(sc), kind="mergesort")          # accumulate: images concatenated in imgIds order
    tp, fp = np.concatenate(tp, axis=1)[:, o], np.concatenate(fp, axis=1)[:, o]
    return dict(img_ids=p.imgIds, img=np.concatenate(im)[o], tp=tp.astype(float), fp=fp.astype(float),
                npig=npig, rec=p.recThrs, T=T, ev=ev)


def ap(d, w):
    """AP with image weights w (COCOeval.accumulate for one category / area / maxDet, then the stats[0] mean)."""
    wd = w[d["img"]]
    tps, fps = np.cumsum(d["tp"] * wd, axis=1), np.cumsum(d["fp"] * wd, axis=1)
    npig = (w * d["npig"]).sum()
    if npig == 0:
        return -1.0
    q = np.zeros((d["T"], len(d["rec"])))
    for t in range(d["T"]):
        rc = tps[t] / npig
        pr = np.maximum.accumulate((tps[t] / (fps[t] + tps[t] + np.spacing(1)))[::-1])[::-1]
        idx = np.searchsorted(rc, d["rec"], side="left")
        q[t] = np.where(idx < len(pr), pr[np.minimum(idx, len(pr) - 1)], 0.0)
    return float(q.mean())


def bca(point, boots, jack, alpha=0.05):
    """BCa interval from bootstrap replicates and leave-one-out jackknife values (Efron & Tibshirani 1993, 14.3)."""
    N = NormalDist()
    frac = (np.sum(boots < point) + 0.5 * np.sum(boots == point)) / len(boots)
    z0 = N.inv_cdf(min(max(frac, 1e-6), 1 - 1e-6))
    u = jack.mean() - jack
    acc = float(np.sum(u ** 3) / (6 * np.sum(u ** 2) ** 1.5)) if np.any(u) else 0.0
    q = [N.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))) for z in (N.inv_cdf(alpha / 2), N.inv_cdf(1 - alpha / 2))]
    return (float(np.percentile(boots, 100 * q[0])), float(np.percentile(boots, 100 * q[1]))), z0, acc


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--a", required=True); ap_.add_argument("--b")
    ap_.add_argument("--subset", nargs="+", default=["all", "ge6"], choices=list(SUBSETS))
    ap_.add_argument("--ann", default=ANN)
    ap_.add_argument("--n_boot", type=int, default=5000); ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--out", default=None)
    a = ap_.parse_args()
    from pycocotools.coco import COCO
    with redirect_stdout(io.StringIO()):
        coco = COCO(a.ann)
    cat = coco.getCatIds(catNms=["person"])
    A = per_image(coco, a.a); B = per_image(coco, a.b) if a.b else None
    n_gt = np.array([sum(1 for x in coco.loadAnns(coco.getAnnIds(imgIds=i, catIds=cat, iscrowd=False))
                         if x.get("num_keypoints", 0) > 0) for i in A["img_ids"]])
    res = []
    for s in a.subset:
        sel = np.array([SUBSETS[s](n) for n in n_gt])
        idx = np.flatnonzero(sel)
        w1 = sel.astype(float)
        point = {k: ap(D, w1) for k, D in (("a", A), ("b", B)) if D is not None}
        # check: the reweighted point estimate is COCOeval's AP on that subset
        for k, D in (("a", A), ("b", B)):
            if D is None:
                continue
            ev = D["ev"]; ev.params.imgIds = [D["img_ids"][i] for i in idx]
            with redirect_stdout(io.StringIO()):
                ev.evaluate(); ev.accumulate(); ev.summarize()
            assert abs(ev.stats[0] - point[k]) < 1e-12, (s, k, ev.stats[0], point[k])
        rng = np.random.default_rng(a.seed)
        bs = {k: [] for k in point}
        for _ in range(a.n_boot):
            w = np.bincount(idx[rng.integers(0, len(idx), len(idx))], minlength=len(sel)).astype(float)
            for k, D in (("a", A), ("b", B)):
                if D is not None:
                    bs[k].append(ap(D, w))
        bs = {k: np.array(v) for k, v in bs.items()}
        jk = {k: [] for k in point}
        for i in idx:
            w = w1.copy(); w[i] = 0.0
            for k, D in (("a", A), ("b", B)):
                if D is not None:
                    jk[k].append(ap(D, w))
        jk = {k: np.array(v) for k, v in jk.items()}
        pct = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))

        def stat(name, pt, b, j):
            ci, z0, acc = bca(pt, b, j)
            return {name: pt, f"{name}_ci": ci, f"{name}_ci_pct": pct(b), f"{name}_bias": float(b.mean() - pt),
                    f"{name}_z0": z0, f"{name}_acc": acc}
        r = {"subset": s, "n_images": int(len(idx)), "n_boot": a.n_boot, "seed": a.seed, "ci": "BCa 95 %",
             "a": a.a, "b": a.b, **stat("ap_a", point["a"], bs["a"], jk["a"])}
        line = f"{s:<4} n={len(idx):5d}  A {point['a']:.4f} [{r['ap_a_ci'][0]:.4f}, {r['ap_a_ci'][1]:.4f}]"
        if B is not None:
            r.update(stat("ap_b", point["b"], bs["b"], jk["b"]))
            r.update(stat("diff", point["a"] - point["b"], bs["a"] - bs["b"], jk["a"] - jk["b"]))
            line += (f"  B {point['b']:.4f} [{r['ap_b_ci'][0]:.4f}, {r['ap_b_ci'][1]:.4f}]"
                     f"  A-B {r['diff']:+.4f} [{r['diff_ci'][0]:+.4f}, {r['diff_ci'][1]:+.4f}]"
                     f"  (pct [{r['diff_ci_pct'][0]:+.4f}, {r['diff_ci_pct'][1]:+.4f}], bias {r['diff_bias']:+.4f})")
        res.append(r); print(line, flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
