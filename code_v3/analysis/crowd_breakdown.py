"""Crowd analysis (2026-10-22/23; restores inline snippets): synthetic person-count distribution,
COCO train crowd share and over-sampling fractions, and PGA by crowd size from eval_peng_gap.py
per-image outputs (native - PEng at true K and autonomous).

    python -m analysis.crowd_breakdown --run hrnet=outputs/peng_gap/372658c6_khaug \
        --run yolo=outputs/peng_gap/yolo_372658c6_khaug --run rtmo=outputs/peng_gap/rtmo_372658c6_khaug --out <json>
"""
import argparse, collections, glob, json
from pathlib import Path
import numpy as np

BUCKETS = [(1, 1), (2, 3), (4, 5), (6, 8), (9, 99)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="NAME=peng_gap output dir")
    ap.add_argument("--auto_sel", default="clamp", help="autonomous selector to report (clamp or lb)")
    ap.add_argument("--coco_train", action="store_true", help="also count COCO train crowds (loads annotations)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = {"synthetic": {}}
    for split in ("train", "val", "test"):
        c = collections.Counter()
        for f in glob.glob(f"data/virtual/{split}/*.json"):
            d = json.load(open(f))
            c[len(d.get("people", d.get("persons", d.get("annotations", []))))] += 1
        out["synthetic"][split] = {str(k): v for k, v in sorted(c.items())}
    if a.coco_train:
        from coco_adapter import CocoAdapter
        ad = CocoAdapter(img_dir=Path("data/coco2017/train2017"),
                         ann_file=Path("data/coco2017/annotations/person_keypoints_train2017.json"), use_depth=False)
        k = np.array([len(ad._get_person_anns(i)) for i in ad.img_ids]); n = len(k); c6 = int((k >= 6).sum())
        out["coco_train"] = {"n": n, "k1": float(np.mean(k == 1)), "k2_5": float(np.mean((k >= 2) & (k <= 5))),
                             "k6plus": c6 / n, "k9plus": float(np.mean(k >= 9)),
                             "oversample_frac": {w: c6 * w / (c6 * w + n - c6) for w in (2, 3, 4, 5)}}
    for spec in a.run:
        name, d = spec.split("=", 1)
        D = {json.loads(l)["image_id"]: json.loads(l) for l in open(Path(d) / "per_image.jsonl")}
        out[name] = {}
        for lo, hi in BUCKETS:
            s = [r for r in D.values() if lo <= r["K_gt"] <= hi]
            nat = np.mean([r["ae"] for r in s]); e = np.mean([r["eff"]["tha"] for r in s])
            au = np.mean([r[a.auto_sel]["tha"] for r in s])
            out[name][f"{lo}-{hi}"] = {"n": len(s), "native": float(nat), "true_k_tha": float(e), "gap_true_k": float(nat - e),
                                       "auto_tha": float(au), "gap_auto": float(nat - au)}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
