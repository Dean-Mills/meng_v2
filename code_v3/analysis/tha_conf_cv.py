"""2-fold cross-fitted lam for confidence-aware THA (eval_tha_conf_ap.py outputs; 2026-10-29).

For each row (lb, clamp): lam*_f = the lam with the highest AP on fold f (ties -> the smaller lam); the cross-fitted
results take fold-1 images from lam*_0 and fold-0 images from lam*_1, so no image is scored with a lam chosen on it.
  per-detector  lam chosen on that detector's folds                         -> results_<row>_cv.json
  shared        lam chosen on the mean fold AP over the given detectors     -> results_<row>_cvshared.json
Written into each run dir; analysis/ap_bootstrap.py then gives paired CIs against lam 0 and the native grouping.

    python -m analysis.tha_conf_cv --dirs outputs/tha_conf/crowd_{hrnet,yolo,rtmo}_t0 --out outputs/tha_conf/cv_crowd.json
"""
import argparse, json
from pathlib import Path
import numpy as np


def pick(ap_by_lam, lams, fold):
    best = max(ap_by_lam[str(l)][fold] for l in lams)
    return min(l for l in lams if ap_by_lam[str(l)][fold] == best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--ann", default="data/coco2017/annotations/person_keypoints_val2017.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import io
    from contextlib import redirect_stdout
    from pycocotools.coco import COCO
    with redirect_stdout(io.StringIO()):
        img_ids = sorted(COCO(a.ann).getImgIds())
    fold_of = {i: k % 2 for k, i in enumerate(img_ids)}            # eval_tha_conf_ap: fold = index % 2
    S = {d: json.load(open(d / "summary.json")) for d in a.dirs}
    lams = S[a.dirs[0]]["lams"]
    out = {}
    for row in S[a.dirs[0]]["ap"]:
        shared = {f: None for f in ("fold0", "fold1")}
        for f in shared:
            mean = {str(l): {f: float(np.mean([S[d]["ap"][row][str(l)][f] for d in a.dirs]))} for l in lams}
            shared[f] = pick(mean, lams, f)
        for d in a.dirs:
            A = S[d]["ap"][row]
            for tag, (l0, l1) in (("cv", (pick(A, lams, "fold0"), pick(A, lams, "fold1"))),
                                  ("cvshared", (shared["fold0"], shared["fold1"]))):
                r0 = json.load(open(d / f"results_{row}_lam{l0:g}.json"))
                r1 = json.load(open(d / f"results_{row}_lam{l1:g}.json"))
                res = [r for r in r0 if fold_of[r["image_id"]] == 1] + [r for r in r1 if fold_of[r["image_id"]] == 0]
                json.dump(res, open(d / f"results_{row}_{tag}.json", "w"))
                out[f"{d.name}/{row}/{tag}"] = dict(lam_from_fold0=l0, lam_from_fold1=l1,
                                                    ap_lam0=A["0.0"], ap_best_fold0=A[str(l0)]["fold0"],
                                                    ap_best_fold1=A[str(l1)]["fold1"])
                print(f"{d.name:<18} {row:<5} {tag:<8} lam* fold0 {l0:<7g} fold1 {l1:<7g}  "
                      f"lam0 AP f0 {A['0.0']['fold0']:.4f} f1 {A['0.0']['fold1']:.4f}  "
                      f"held-out: f1 at lam*0 {A[str(l0)]['fold1']:.4f}, f0 at lam*1 {A[str(l1)]['fold0']:.4f}")
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
