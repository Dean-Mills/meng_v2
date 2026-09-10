"""Figure: the four PG-GAT edge categories and the skeleton hop distance, on the COCO skeleton.

Two people side by side. From one anchor joint (the left shoulder of person A) every possible edge is
drawn and coloured by the category PG-GAT assigns it: skeletal neighbour (directly connected joint
types), same limb (same anatomical chain, not adjacent), cross-body (everything else), and same type
(the other person's left shoulder). The number at each joint of person A is the skeleton hop distance
from the anchor, the second feature of Modification 2. Categories and hops are computed from the
model's own tables (sa_gat.py), not hand-labelled.

Run from code_v3/:
    python ../paper_cviu/figures/src/fig_edge_categories.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path.cwd()))
import sa_gat  # noqa: E402  (tables only)

FIGDIR = Path(__file__).resolve().parent.parent
CAT_COLORS = {"same type": "#c23b3b", "skeletal neighbour": "#3c8a4e", "same limb": "#2b5fa3", "cross-body": "0.6"}
ANCHOR = 5   # left shoulder

# a standing pose, image coordinates (y down), 17 COCO joints
POSE = np.array([
    [0.00, 0.00],   # nose
    [-0.06, -0.05], [0.06, -0.05],     # eyes
    [-0.13, -0.02], [0.13, -0.02],     # ears
    [-0.30, 0.25], [0.30, 0.25],       # shoulders
    [-0.42, 0.62], [0.42, 0.62],       # elbows
    [-0.48, 0.98], [0.48, 0.98],       # wrists
    [-0.18, 0.95], [0.18, 0.95],       # hips
    [-0.20, 1.45], [0.20, 1.45],       # knees
    [-0.22, 1.95], [0.22, 1.95],       # ankles
])
NAMES = ["nose", "L eye", "R eye", "L ear", "R ear", "L shoulder", "R shoulder", "L elbow", "R elbow",
         "L wrist", "R wrist", "L hip", "R hip", "L knee", "R knee", "L ankle", "R ankle"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8, "figure.dpi": 150,
})


def category(a, b, skel, limb):
    if a == b:
        return "same type"
    if skel[a, b]:
        return "skeletal neighbour"
    if limb[a, b]:
        return "same limb"
    return "cross-body"


def main():
    skel = sa_gat._build_skeleton_neighbors()
    limb = sa_gat._build_same_limb()
    hops = sa_gat._build_hop_distances().numpy()
    # the helper sets may be sets of pairs or matrices; normalise to matrices
    def to_mat(x):
        if isinstance(x, set):
            m = np.zeros((17, 17), bool)
            for i, j in x:
                m[i, j] = m[j, i] = True
            return m
        return np.asarray(x, bool)
    skel, limb = to_mat(skel), to_mat(limb)

    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    A = POSE.copy(); B = POSE.copy(); B[:, 0] += 1.35
    for P, col in ((A, "0.3"), (B, "0.55")):
        for i, j in sa_gat.COCO_SKELETON:
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], "-", color="0.8", lw=1.2, zorder=1)
        ax.plot(P[:, 0], P[:, 1], "o", color=col, ms=3.2, zorder=3)
    # edges from the anchor of A to every joint of A, coloured by category
    for j in range(17):
        if j == ANCHOR:
            continue
        c = category(ANCHOR, j, skel, limb)
        ax.plot([A[ANCHOR, 0], A[j, 0]], [A[ANCHOR, 1], A[j, 1]], "-", color=CAT_COLORS[c],
                lw=1.6 if c != "cross-body" else 0.9, alpha=0.95 if c != "cross-body" else 0.7, zorder=2)
    # same-type edge to the other person's left shoulder
    ax.plot([A[ANCHOR, 0], B[ANCHOR, 0]], [A[ANCHOR, 1], B[ANCHOR, 1]], "-", color=CAT_COLORS["same type"], lw=1.8, zorder=2)
    ax.plot(A[ANCHOR, 0], A[ANCHOR, 1], "o", color="black", ms=5, zorder=4)
    ax.annotate("anchor:\nL shoulder", A[ANCHOR], textcoords="offset points", xytext=(-9, -2), fontsize=8, ha="right", va="center")
    # hop distances on person A (body joints); the head group is not linked to the torso in the
    # skeleton table the model uses (no ear-shoulder edges), so its hop value is the sentinel maximum
    maxhop = int(hops.max())
    for j in range(5, 17):
        if j == ANCHOR:
            continue
        dx = -7 if A[j, 0] < 0 else 7
        ax.annotate(f"{int(hops[ANCHOR, j])}", A[j], textcoords="offset points", xytext=(dx, 0), fontsize=8,
                    ha="right" if dx < 0 else "left", va="center", color="0.25")
    ax.text(A[0, 0], A[0, 1] - 0.30, f"head joints: no skeleton path to the anchor, hop = {maxhop} (maximum)",
            ha="center", va="bottom", fontsize=8, color="0.25")
    ax.text(A[0, 0], A[0, 1] - 0.16, "person A", ha="center", fontsize=8)
    ax.text(B[0, 0], B[0, 1] - 0.16, "person B", ha="center", fontsize=8)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    handles = [Line2D([], [], color=c, lw=2, label=k) for k, c in CAT_COLORS.items()]
    handles.append(Line2D([], [], marker="$3$", color="0.25", lw=0, label="hop distance from the anchor"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
              handlelength=1.4, columnspacing=1.0)
    stem = FIGDIR / "edge_categories"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    counts = {k: 0 for k in CAT_COLORS}
    for j in range(17):
        if j != ANCHOR:
            counts[category(ANCHOR, j, skel, limb)] += 1
    print("edges from the anchor within person A:", counts, "| plus 1 same-type edge to person B")
    print(f"wrote {stem}.pdf/.png")


if __name__ == "__main__":
    main()
