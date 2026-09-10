"""Insets for the concept figure (concept_modular.tex): one COCO scene from the end-to-end dump,
(1) the pooled HigherHRNet detections with identities discarded (grey), (2) the same detections grouped
by the frozen PG-GAT + K-head + k-means, coloured by the ground-truth person each cluster maps to.
Only detections matched to ground truth (the scored set) are drawn, as in the dissertation's Fig. 4.4.

Run from code_v3/:
    python ../paper_cviu/figures/src/fig_concept_insets.py [--img_id 281759]
Writes paper_cviu/figures/concept_inset_ungrouped.pdf and concept_inset_grouped.pdf (+ .png).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

FIGDIR = Path(__file__).resolve().parent.parent
E2E = Path("outputs/e2e_per_image")
COCO = Path("data/coco2017/val2017")
PERSON_COLORS = ["#c23b3b", "#2b5fa3", "#3c8a4e", "#b58a2a", "#7a4fa3", "#3a9fb0", "#d0699a"]
COCO_SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
                 (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def cluster_to_person(labels, gt):
    k, p = labels.max() + 1, gt.max() + 1
    conf = np.zeros((k, p), int)
    for l, g in zip(labels, gt):
        conf[l, g] += 1
    r, c = linear_sum_assignment(-conf)
    m = {int(x): int(y) for x, y in zip(r, c)}
    return np.array([m.get(int(l), p + int(l)) for l in labels])


def draw(ax, img, pos, types, labels, colours, crop, lw=1.3, ms=3.0):
    ax.imshow(img)
    for l in np.unique(labels):
        col = colours[int(l)]
        m = labels == l
        lookup = {int(t): p for t, p in zip(types[m], pos[m])}
        for u, v in COCO_SKELETON:
            if u in lookup and v in lookup:
                ax.plot([lookup[u][0], lookup[v][0]], [lookup[u][1], lookup[v][1]], "-", color=col, lw=lw, alpha=0.9, zorder=2)
        ax.plot(pos[m, 0], pos[m, 1], "o", color=col, ms=ms, mec="white", mew=0.5, zorder=3)
    x0, y0, x1, y1 = crop
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_id", type=int, default=281759)
    a = ap.parse_args()
    z = np.load(E2E / "arrays" / f"{a.img_id}.npz", allow_pickle=True)
    img = Image.open(COCO / str(z["file_name"])).convert("RGB")
    m = z["matched_idx"]
    pos, types, gt = z["det_pos"][m], z["det_types"][m], z["gt_person"]
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    w, h = hi - lo
    crop = (max(0, lo[0] - 0.08 * w - 8), max(0, lo[1] - 0.08 * h - 8), min(img.width, hi[0] + 0.08 * w + 8), min(img.height, hi[1] + 0.08 * h + 8))
    aspect = (crop[3] - crop[1]) / (crop[2] - crop[0])
    W = 1.6
    # (1) ungrouped: every matched detection in one grey, no skeleton lines
    fig, ax = plt.subplots(figsize=(W, W * aspect))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.imshow(img); ax.plot(pos[:, 0], pos[:, 1], "o", color="0.15", ms=3.0, mec="white", mew=0.5, zorder=3)
    ax.set_xlim(crop[0], crop[2]); ax.set_ylim(crop[3], crop[1]); ax.axis("off")
    fig.savefig(FIGDIR / "concept_inset_ungrouped.pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(FIGDIR / "concept_inset_ungrouped.png", dpi=250, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    # (2) grouped by PG-GAT + K-head
    labels = np.asarray(z["labels_pred"])
    mapped = cluster_to_person(labels, gt)
    colours = {int(l): PERSON_COLORS[int(mapped[labels == l][0]) % len(PERSON_COLORS)] for l in np.unique(labels)}
    fig, ax = plt.subplots(figsize=(W, W * aspect))
    fig.subplots_adjust(0, 0, 1, 1)
    draw(ax, img, pos, types, labels, colours, crop)
    fig.savefig(FIGDIR / "concept_inset_grouped.pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(FIGDIR / "concept_inset_grouped.png", dpi=250, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"image {a.img_id}: K_gt {int(z['K_gt'])}, K_pred {int(z['K_pred'])}, {len(m)} matched detections, "
          f"misassigned {(mapped != gt).sum()}; wrote concept_inset_ungrouped/grouped .pdf/.png")


if __name__ == "__main__":
    main()
