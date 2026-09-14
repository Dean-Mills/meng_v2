"""Dissertation variant (4.3 x 5.0 in, 9 pt, for 0.7\linewidth of the 441 pt text width).

Figure: performance against crowd size - PGA and AP per annotated person count.

(a) end-to-end PGA on HigherHRNet detections per annotated person count K (1..7, 8+): HigherHRNet
    native AE, PG-GAT with oracle K, PG-GAT with the K-head (from outputs/e2e_per_image, the dump
    behind the dissertation's Ch4 numbers; 2,165 scored scenes).
(b) COCO AP@OKS per annotated person count on the same detector (from outputs/e2e_ap, the journal
    paper's AP run over all 5,000 val2017 images): HigherHRNet native on the pooled joints and the
    autonomous PG-GAT pipeline, scored with pycocotools on the images of each bin.
The K <= 5 range of the synthetic training data is marked; the number of images per bin is printed
under the axis. Every number is written to the JSON sidecar.

Run from code_v3/:
    python ../dissertation/figures/src/fig_crowd_size.py
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = Path(__file__).resolve().parent.parent
E2E = Path("outputs/e2e_per_image/per_image.jsonl")
AP_DIR = Path("outputs/e2e_ap")
ANN = Path("data/coco2017/annotations/person_keypoints_val2017.json")
KMAX = 8   # last bin pools K >= 8

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def binned(k):
    return int(min(k, KMAX))


def main():
    # ── (a) PGA per K from the per-image dump ────────────────────────────
    recs = [json.loads(l) for l in open(E2E)]
    recs = [r for r in recs if r["pga_oracle"] is not None and r["pga_pred"] is not None]
    bins = list(range(1, KMAX + 1))
    pga = {m: [] for m in ("ae", "oracle", "pred")}
    n_pga = []
    for b in bins:
        sel = [r for r in recs if binned(r["K_gt"]) == b]
        n_pga.append(len(sel))
        for m, key in (("ae", "pga_ae"), ("oracle", "pga_oracle"), ("pred", "pga_pred")):
            pga[m].append(float(np.mean([r[key] for r in sel])) if sel else np.nan)

    # ── (b) AP per K from the AP run ─────────────────────────────────────
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO(str(ANN))
    per = {r["image_id"]: r for r in map(json.loads, open(AP_DIR / "per_image.jsonl"))}
    methods = {"native_pooled": "HigherHRNet native (pooled joints)", "pggat_pred": "PG-GAT + K-head (autonomous)"}
    res = {m: json.load(open(AP_DIR / f"results_{m}.json")) for m in methods}
    ap = {m: [] for m in methods}
    n_ap = []
    for b in bins:
        ids = [i for i, r in per.items() if r["n_gt"] >= 1 and binned(r["n_gt"]) == b]
        n_ap.append(len(ids))
        idset = set(ids)
        for m in methods:
            rr = [x for x in res[m] if x["image_id"] in idset]
            if not rr:
                ap[m].append(np.nan); continue
            dt = coco.loadRes(rr); ev = COCOeval(coco, dt, iouType="keypoints"); ev.params.imgIds = ids
            with contextlib.redirect_stdout(io.StringIO()):
                ev.evaluate(); ev.accumulate(); ev.summarize()
            ap[m].append(float(ev.stats[0]))

    stem = FIGDIR / "crowd_size"
    json.dump({"bins": bins, "last_bin_pools_ge": KMAX, "pga": pga, "n_scenes_pga": n_pga,
               "ap": ap, "n_images_ap": n_ap, "sources": {"pga": str(E2E), "ap": str(AP_DIR)}},
              open(stem.with_suffix(".json"), "w"), indent=1)
    for b, n1, n2 in zip(bins, n_pga, n_ap):
        print(f"K={b:<2} n_pga={n1:<5} AE {pga['ae'][b-1]:.3f} oracle {pga['oracle'][b-1]:.3f} pred {pga['pred'][b-1]:.3f} | "
              f"n_ap={n2:<5} native {ap['native_pooled'][b-1]:.3f} pggat {ap['pggat_pred'][b-1]:.3f}")

    # ── figure (single column, two stacked panels) ──────────────────────
    labels = [str(b) for b in bins[:-1]] + [f"{KMAX}+"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.3, 5.0), sharex=True,
                                   gridspec_kw={"left": 0.16, "right": 0.98, "top": 0.95, "bottom": 0.16, "hspace": 0.25})
    x = np.arange(len(bins))
    for ax in (ax1, ax2):
        ax.axvspan(4.5, len(bins) - 0.5, color="0.93", zorder=0)
    ax1.text(4.6, 0.62, "beyond the synthetic\nrange ($K \\leq 5$)", fontsize=9, color="0.35", va="bottom")
    ax1.plot(x, pga["ae"], "-s", color="#2b5fa3", ms=3.5, label="HigherHRNet AE (native)")
    ax1.plot(x, pga["oracle"], "-o", color="0.45", ms=3.5, label="PG-GAT, oracle $K$")
    ax1.plot(x, pga["pred"], "-o", color="#c23b3b", ms=3.5, label="PG-GAT, K-head $\\hat{K}$")
    ax1.set_ylabel("end-to-end PGA", fontsize=9)
    ax1.set_ylim(0.6, 1.01)
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    ax1.set_title("(a) grouping accuracy on matched detections", fontsize=9)

    ax2.plot(x, ap["native_pooled"], "-s", color="#2b5fa3", ms=3.5, label="HigherHRNet native")
    ax2.plot(x, ap["pggat_pred"], "-o", color="#c23b3b", ms=3.5, label="PG-GAT + K-head")
    ax2.set_ylabel("AP@OKS", fontsize=9)
    ax2.set_ylim(0.0, 0.85)
    ax2.legend(frameon=False, fontsize=9, loc="lower left")
    ax2.set_title("(b) system-level AP on identical joints", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{l}\n$n$={n}" for l, n in zip(labels, n_ap)], fontsize=9)
    ax2.set_xlabel("annotated people in the image", fontsize=9)

    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {stem}.pdf/.png/.json")


if __name__ == "__main__":
    main()
