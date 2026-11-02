"""AP-gap anatomy: where the native grouping earns the AP that PG-GAT-H loses (diagnostics, 2026-10-29).

Compares two eval_e2e_ap_peng.py results files on the same detector cache and threshold (A = native, B = ours) on
every val2017 image, COCOeval keypoint protocol (area all, maxDets 20). Four read-outs:
  1. AP per OKS threshold (0.50 ... 0.95) for A and B: is the gap at loose or tight thresholds?
  2. bin swap: AP of B with A's instances substituted on one crowd-size bin (annotated people with keypoints); the AP
     recovered is that bin's share of the gap (shares need not sum exactly to the gap: AP is not additive).
  3. ranking: (a) both files re-scored with each instance's best OKS to any GT person (oracle ranking: removes score
     effects, leaves instance quality); (b) B re-scored with the score of the A instance that contributes most of its
     joints (B's grouping with A's ranking).
  4. slot anatomy per GT person: each pooled detection -> GT person by the per-keypoint OKS term (ap_ceiling.oks_assign,
     >= --min_kp_oks); a person's home instance = the instance with the highest OKS to it; each labelled keypoint slot
     of the home instance is classed correct (this person's detection) / other (another person's) / junk (no person's)
     / empty. Reported: A x B slot transitions with the per-keypoint OKS term they move, where the person's own
     detection went in B when B's slot is wrong, the persons whose OKS crosses 0.5 / 0.75 / 0.9 between A and B, and
     for slots one method has right and the other wrong, the confidences of the two detections (would detection
     confidence have told them apart?).
Instances are mapped back to pooled detections by (type, x, y, conf), which eval_e2e_ap copies unchanged.

    python -m analysis.ap_gap_anatomy --cache_dir outputs/e2e_ap/hrnet_cache \
        --a outputs/verify_20261026/ap/hrnet_t0/results_native_pooled.json \
        --b outputs/verify_20261026/ap/hrnet_t0/results_crowd__lb_tha.json --out <json>
"""
import argparse, io, json
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
from analysis.ap_ceiling import SIGMAS, oks_assign
from eval_e2e_ap import pool_detections

JOINTS = ("nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri",
          "l_hip", "r_hip", "l_kne", "r_kne", "l_ank", "r_ank")
BINS = (("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-5", 4, 5), ("6-8", 6, 8), ("9-14", 9, 14), ("15+", 15, 10 ** 9))
CLASSES = ("correct", "other", "junk", "empty")
TAUS = (0.5, 0.75, 0.9)


def matches(coco, res):
    """COCOeval matches for area 'all', detections in COCOeval.accumulate order (images in imgIds order)."""
    from pycocotools.cocoeval import COCOeval
    with redirect_stdout(io.StringIO()):
        ev = COCOeval(coco, coco.loadRes(res), iouType="keypoints")
        ev.params.imgIds = sorted(coco.getImgIds())
        ev.evaluate(); ev.accumulate(); ev.summarize()
    p, md = ev.params, ev.params.maxDets[-1]
    sc, im, tp, fp, npig = [], [], [], [], np.zeros(len(p.imgIds))
    for i, e in enumerate(ev.evalImgs[:len(p.imgIds)]):
        if e is None:
            continue
        npig[i] = np.count_nonzero(e["gtIgnore"] == 0)
        m, ig = e["dtMatches"][:, :md] > 0, e["dtIgnore"][:, :md].astype(bool)
        sc.append(np.asarray(e["dtScores"][:md])); im.append(np.full(m.shape[1], i))
        tp.append(m & ~ig); fp.append(~m & ~ig)
    return dict(sc=np.concatenate(sc), img=np.concatenate(im), tp=np.concatenate(tp, 1).astype(float),
                fp=np.concatenate(fp, 1).astype(float), npig=npig, rec=p.recThrs, coco_ap=float(ev.stats[0]))


def ap_per_t(d, rec, npig):
    """COCOeval.accumulate AP per OKS threshold for one category / area / maxDet (d already filtered)."""
    o = np.lexsort((d["img"], -d["sc"]))
    tps, fps = np.cumsum(d["tp"][:, o], 1), np.cumsum(d["fp"][:, o], 1)
    q = np.zeros((len(tps), len(rec)))
    for t in range(len(tps)):
        rc = tps[t] / npig
        pr = np.maximum.accumulate((tps[t] / (fps[t] + tps[t] + np.spacing(1)))[::-1])[::-1]
        idx = np.searchsorted(rc, rec, side="left")
        q[t] = np.where(idx < len(pr), pr[np.minimum(idx, len(pr) - 1)], 0.0)
    return q.mean(1)


def hybrid(A, B, take_a):
    """A's detections on images where take_a, B's elsewhere."""
    ka, kb = take_a[A["img"]], ~take_a[B["img"]]
    return {k: (np.concatenate([A[k][..., ka], B[k][..., kb]], -1)) for k in ("sc", "img", "tp", "fp")}


def oks_matrix(inst, gt, areas):
    """COCOeval.computeOks for instances [I, 17, 3] x GT persons [P, 17, 3] with >= 1 labelled keypoint."""
    if len(inst) == 0 or len(gt) == 0:
        return np.zeros((len(inst), len(gt))), np.zeros((len(inst), len(gt), 17))
    d2 = ((inst[:, None, :, :2] - gt[None, :, :, :2]) ** 2).sum(-1)
    term = np.exp(-d2 / (2 * SIGMAS) ** 2 / (areas[None, :, None] + np.spacing(1)) / 2)
    vis = (gt[:, :, 2] > 0)[None].astype(float)
    return (term * vis).sum(-1) / vis.sum(-1), term


def by_image(res):
    out = defaultdict(list)
    for r in res:
        out[r["image_id"]].append(r)
    return out


def slots_of(insts, key2det):
    """Instance keypoints [I, 17, 3] and the pooled detection behind each slot [I, 17] (-1 = empty)."""
    pool = {k: list(v) for k, v in key2det.items()}
    kps = np.array([np.array(r["keypoints"], np.float64).reshape(17, 3) for r in insts]).reshape(-1, 17, 3)
    src, miss = np.full(kps.shape[:2], -1), 0
    for i, j in zip(*np.nonzero(kps[:, :, 2] > 0)):
        q = pool.get((int(j), float(kps[i, j, 0]), float(kps[i, j, 1]), float(kps[i, j, 2])))
        if q:
            src[i, j] = q.pop(0)
        else:
            miss += 1
    return kps, src, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=Path, required=True)
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--conf_threshold", type=float, default=0.0)
    ap.add_argument("--min_kp_oks", type=float, default=0.5)
    ap.add_argument("--coco_ann_file", default="data/coco2017/annotations/person_keypoints_val2017.json")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from pycocotools.coco import COCO
    with redirect_stdout(io.StringIO()):
        coco = COCO(a.coco_ann_file)
    cat = coco.getCatIds(catNms=["person"])
    img_ids = sorted(coco.getImgIds())
    resA, resB = json.load(open(a.a)), json.load(open(a.b))
    if a.max_images:
        keep = set(img_ids[:a.max_images])
        resA = [r for r in resA if r["image_id"] in keep]; resB = [r for r in resB if r["image_id"] in keep]
    insA, insB = by_image(resA), by_image(resB)

    # --- 4. slot anatomy (also yields the oracle and A-ranked scores for 3) ---
    trans = Counter(); dterm = Counter(); went = Counter(); went_type = Counter(); lost_type = Counter()
    trans_bin = defaultdict(Counter); persons = []; n_miss = 0; differs = Counter(); n_img_bin = Counter()
    oracle_A, oracle_B, ranked_B = [], [], []
    swap_conf = {"lost": [], "won": []}                      # (conf of A's slot detection, conf of B's) per changed slot
    n_gt_of = {}
    for img_id in img_ids:
        anns = [x for x in coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=cat, iscrowd=False))
                if x.get("num_keypoints", 0) > 0]
        n_gt_of[img_id] = len(anns)
        if a.max_images and img_id not in keep:
            continue
        b = next(n for n, lo, hi in BINS if lo <= len(anns) <= hi)
        n_img_bin[b] += 1
        joints = np.load(a.cache_dir / f"{img_id}.npy")
        pos, typ, conf, _ = pool_detections(joints, a.conf_threshold)
        key2det = defaultdict(list)
        for d in range(len(pos)):
            key2det[(int(typ[d]), float(pos[d, 0]), float(pos[d, 1]), float(conf[d]))].append(d)
        kA, sA, mA = slots_of(insA[img_id], key2det); kB, sB, mB = slots_of(insB[img_id], key2det)
        n_miss += mA + mB
        setsA = {frozenset(s[s >= 0]) for s in sA}; setsB = {frozenset(s[s >= 0]) for s in sB}
        differs[b] += setsA != setsB
        gt = np.array([np.array(x["keypoints"], np.float64).reshape(17, 3) for x in anns]).reshape(-1, 17, 3)
        areas = np.array([x["area"] for x in anns], np.float64)
        oA, tA = oks_matrix(kA, gt, areas); oB, tB = oks_matrix(kB, gt, areas)
        oracle_A += [dict(r, score=float(o)) for r, o in zip(insA[img_id], oA.max(1) if len(anns) else np.zeros(len(kA)))]
        oracle_B += [dict(r, score=float(o)) for r, o in zip(insB[img_id], oB.max(1) if len(anns) else np.zeros(len(kB)))]
        det2a = np.full(len(pos), -1)
        for i, s in enumerate(sA):
            det2a[s[s >= 0]] = i
        for r, s in zip(insB[img_id], sB):
            own = Counter(det2a[s[s >= 0]].tolist())
            best = max(own, key=lambda i: (own[i], insA[img_id][i]["score"] if i >= 0 else -1)) if own else -1
            ranked_B.append(dict(r, score=insA[img_id][best]["score"] if best >= 0 else r["score"]))
        if not len(anns) or not len(pos) or not len(kA) or not len(kB):
            continue
        lab = oks_assign(pos, typ, list(gt), areas, a.min_kp_oks)
        homeA, homeB = oA.argmax(0), oB.argmax(0)
        home_of_B = {int(h): g for g, h in enumerate(homeB)}
        for g in range(len(anns)):
            row = dict(image_id=img_id, n_gt=len(anns), area=float(areas[g]),
                       oks_a=float(oA[homeA[g], g]), oks_b=float(oB[homeB[g], g]), wrong_a=0, wrong_b=0, diff=[])
            for t in np.flatnonzero(gt[g, :, 2] > 0):
                cls = []
                for s, h in ((sA, homeA[g]), (sB, homeB[g])):
                    d = s[h, t]
                    cls.append("empty" if d < 0 else "correct" if lab[d] == g else "other" if lab[d] >= 0 else "junk")
                own = np.flatnonzero((lab == g) & (typ == t))
                avail = len(own) > 0
                key = (cls[0], cls[1], avail)
                trans[key] += 1; trans_bin[b][key] += 1
                dterm[key] += float(tA[homeA[g], g, t] - tB[homeB[g], g, t])
                row["wrong_a"] += cls[0] != "correct"; row["wrong_b"] += cls[1] != "correct"
                if cls[0] != cls[1]:
                    row["diff"].append((JOINTS[t], cls[0], cls[1]))
                if cls[0] == "correct" and cls[1] != "correct":
                    lost_type[JOINTS[t]] += 1
                if cls[0] != cls[1] and "empty" not in cls and "correct" in cls:
                    dA, dB = sA[homeA[g], t], sB[homeB[g], t]
                    swap_conf["lost" if cls[0] == "correct" else "won"].append((float(conf[dA]), float(conf[dB])))
                if avail and cls[1] != "correct":
                    d = own[np.argmax(conf[own])]
                    hb = np.flatnonzero(sB == d)
                    if not len(hb):
                        w = "not in any instance"
                    else:
                        i = int(np.nonzero(sB == d)[0][0])
                        w = ("own home, outranked" if i == homeB[g] else
                             "another person's home" if i in home_of_B else "no-person instance")
                    went[(cls[0] == "correct", w)] += 1
                    if cls[0] == "correct":
                        went_type[(JOINTS[t], w)] += 1
            persons.append(row)

    # --- 1-3. AP read-outs ---
    A, B = matches(coco, resA), matches(coco, resB)
    rec, npig = A["rec"], A["npig"].sum()
    apA, apB = ap_per_t(A, rec, npig), ap_per_t(B, rec, npig)
    assert abs(apA.mean() - A["coco_ap"]) < 1e-9 and abs(apB.mean() - B["coco_ap"]) < 1e-9, (apA.mean(), A["coco_ap"])
    swap = {}
    for name, lo, hi in BINS:
        take = np.array([lo <= n_gt_of[i] <= hi for i in sorted(coco.getImgIds())])
        swap[name] = float(ap_per_t(hybrid(A, B, take), rec, npig).mean() - apB.mean())
    rank = {}
    for name, rs in (("oracle_a", oracle_A), ("oracle_b", oracle_B), ("b_with_a_scores", ranked_B)):
        rank[name] = matches(coco, rs)["coco_ap"]

    # --- report ---
    P = persons
    cross = {}
    for tau in TAUS:
        lost = [p for p in P if p["oks_a"] >= tau > p["oks_b"]]; won = [p for p in P if p["oks_b"] >= tau > p["oks_a"]]
        nd = Counter(min(sum(1 for _, ca, cb in p["diff"] if ca == "correct"), 3) for p in lost)
        cross[str(tau)] = dict(lost=len(lost), won=len(won), lost_ge6=sum(p["n_gt"] >= 6 for p in lost),
                               won_ge6=sum(p["n_gt"] >= 6 for p in won),
                               lost_by_slots_wrong_in_b={str(k): v for k, v in sorted(nd.items())})
    out = dict(a=a.a, b=a.b, cache=str(a.cache_dir), conf_threshold=a.conf_threshold, min_kp_oks=a.min_kp_oks,
               unmapped_slots=n_miss, n_persons=len(P),
               ap=dict(a=float(apA.mean()), b=float(apB.mean()), per_t=dict(thr=[float(x) for x in np.linspace(.5, .95, 10)],
                       a=apA.tolist(), b=apB.tolist())),
               bin_swap=dict(gap=float(apA.mean() - apB.mean()), recovered=swap,
                             images={n: n_img_bin[n] for n, _, _ in BINS}, images_differ={n: differs[n] for n, _, _ in BINS}),
               ranking=dict(a=float(apA.mean()), b=float(apB.mean()), **rank),
               slots={f"{ca}->{cb}{'' if av else ' (no own det)'}": dict(n=n, dterm=dterm[(ca, cb, av)])
                      for (ca, cb, av), n in sorted(trans.items(), key=lambda x: -x[1])},
               slots_by_bin={bn: {f"{ca}->{cb}{'' if av else ' (no own det)'}": n for (ca, cb, av), n in c.items()
                                  if ca != cb} for bn, c in trans_bin.items()},
               own_det_when_b_wrong={f"{'a correct' if ac else 'a wrong'}: {w}": n for (ac, w), n in went.items()},
               lost_slots_by_type=dict(lost_type.most_common()),
               lost_slot_destination_by_type={f"{j}: {w}": n for (j, w), n in went_type.most_common()},
               crossings=cross,
               swap_conf={k: dict(n=len(v), median_a=float(np.median([x for x, _ in v])) if v else None,
                                  median_b=float(np.median([y for _, y in v])) if v else None,
                                  frac_a_higher=float(np.mean([x > y for x, y in v])) if v else None,
                                  frac_ratio_ge2=float(np.mean([max(x, y) >= 2 * min(x, y) for x, y in v])) if v else None)
                          for k, v in swap_conf.items()},
               oks_mean=dict(a=float(np.mean([p["oks_a"] for p in P])), b=float(np.mean([p["oks_b"] for p in P])),
                             a_ge6=float(np.mean([p["oks_a"] for p in P if p["n_gt"] >= 6])),
                             b_ge6=float(np.mean([p["oks_b"] for p in P if p["n_gt"] >= 6]))))
    json.dump(out, open(a.out, "w"), indent=1)
    json.dump(P, open(a.out.with_name(a.out.stem + "_persons.json"), "w"))
    print(f"AP A {apA.mean():.4f}  B {apB.mean():.4f}  gap {apA.mean() - apB.mean():+.4f}   unmapped slots {n_miss}")
    print("per OKS thr gap: " + " ".join(f"{t:.2f} {x:+.4f}" for t, x in zip(np.linspace(.5, .95, 10), apA - apB)))
    print("bin swap recovered: " + " ".join(f"{k} {v:+.4f} ({differs[k]}/{n_img_bin[k]} img differ)" for k, v in swap.items())
          + f"  sum {sum(swap.values()):+.4f}")
    print("ranking: " + " ".join(f"{k} {v:.4f}" for k, v in out["ranking"].items()))
    print("slot transitions (A->B, n, OKS term A-B):")
    for k, v in list(out["slots"].items())[:14]:
        print(f"   {k:<34} {v['n']:7d}  {v['dterm']:+9.2f}")
    print("own detection when B wrong:", out["own_det_when_b_wrong"])
    print("lost slots by type:", out["lost_slots_by_type"])
    print("crossings:", json.dumps(cross))
    print("changed-slot confidences (lost = A correct, won = B correct):", json.dumps(out["swap_conf"]))
    print("mean home OKS:", {k: round(v, 4) for k, v in out["oks_mean"].items()})


if __name__ == "__main__":
    main()
