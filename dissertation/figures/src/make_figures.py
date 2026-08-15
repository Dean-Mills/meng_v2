"""Generate the data-driven dissertation figures.

Produces (into dissertation/figures/):
  1. predk_recall.pdf     — predicted-K advantage vs detection recall (4 points)
  2. skeleton_graph.pdf   — COCO skeleton topology + kNN graph on a real 2-person scene
  3. synth_examples.pdf   — 2x2 grid of synthetic scenes, K=2..5, keypoints overlaid

House style: serif (Times via mathptmx-compatible STIX/Times), >=8pt fonts,
no frames, vector output. Run from code_v3/ so the repo imports resolve:
    python ../dissertation/figures/src/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = Path(__file__).resolve().parent.parent  # dissertation/figures/

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
JOINT_NAMES = [
    "nose", "L eye", "R eye", "L ear", "R ear",
    "L shoulder", "R shoulder", "L elbow", "R elbow",
    "L wrist", "R wrist", "L hip", "R hip",
    "L knee", "R knee", "L ankle", "R ankle",
]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: predicted-K advantage vs detection recall
# ─────────────────────────────────────────────────────────────────────────────

def fig_predk_recall():
    # (label, recall, predK_advantage) — from eval_e2e_* runs on b5noyg7t
    # and the GT-keypoint control (eval_k_estimation, recall = 1.0).
    points = [
        ("RTMO-L",          0.774, +0.0304),
        ("HigherHRNet-W32", 0.793, +0.0053),
        ("YOLO11x-pose",    0.816, +0.0006),
        ("GT keypoints",    1.000, -0.0066),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--", zorder=1)
    ax.plot(xs, ys, "-", color="0.3", lw=1.0, zorder=2)
    ax.plot(xs, ys, "o", color="black", ms=5, zorder=3)
    offsets = {"RTMO-L": (8, 0), "HigherHRNet-W32": (8, 4),
               "YOLO11x-pose": (8, 6), "GT keypoints": (-8, 8)}
    ha = {"GT keypoints": "right"}
    for label, x, y in points:
        dx, dy = offsets[label]
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9, ha=ha.get(label, "left"))
    ax.set_xlabel("Keypoint detection recall")
    ax.set_ylabel("Predicted-$K$ PGA advantage\nover oracle $K$")
    ax.set_xlim(0.75, 1.03)
    fig.savefig(FIGDIR / "predk_recall.pdf")
    plt.close(fig)
    print("wrote predk_recall.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: skeleton topology + kNN graph on a 2-person scene
# ─────────────────────────────────────────────────────────────────────────────

def fig_skeleton_graph():
    sys.path.insert(0, str(Path.cwd()))
    from virtual_adapter import VirtualAdapter  # noqa: E402

    # Left panel: canonical skeleton drawn from a hand-placed standing pose.
    pose = np.array([
        [0.50, 0.115], [0.475, 0.085], [0.525, 0.085], [0.445, 0.10], [0.555, 0.10],
        [0.40, 0.22], [0.60, 0.22], [0.35, 0.38], [0.65, 0.38],
        [0.32, 0.52], [0.68, 0.52], [0.44, 0.52], [0.56, 0.52],
        [0.42, 0.72], [0.58, 0.72], [0.41, 0.92], [0.59, 0.92],
    ])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6))

    ax = axes[0]
    for a, b in COCO_SKELETON:
        ax.plot(*zip(pose[a], pose[b]), "-", color="0.45", lw=1.4, zorder=1)
    ax.plot(pose[:, 0], pose[:, 1], "o", color="black", ms=5, zorder=2)
    head_offsets = {0: (0, -11, "center"), 1: (-5, 8, "right"), 2: (5, 8, "left"),
                    3: (-9, -2, "right"), 4: (9, -2, "left")}
    for j, (x, y) in enumerate(pose):
        if j in head_offsets:
            dx, dy, ha = head_offsets[j]
        else:
            side = -1 if x < 0.5 else 1
            dx, dy, ha = 7 * side, 3, ("left" if side > 0 else "right")
        ax.annotate(JOINT_NAMES[j], (x, y), textcoords="offset points",
                    xytext=(dx, dy), fontsize=8, ha=ha)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) COCO 17-joint skeleton", fontsize=10)

    # Right panel: kNN graph (k=8) on a real 2-person synthetic scene.
    adapter = VirtualAdapter(data_dir=Path("data/virtual/test"))

    def scene_arrays(smp):
        pts, person, types = [], [], []
        for pid, kp in enumerate(smp["keypoints"]):
            kp = kp.numpy()
            for j in range(17):
                if kp[j, 3] > 0:
                    pts.append([kp[j, 0], kp[j, 1]])
                    person.append(pid)
                    types.append(j)
        return np.array(pts), np.array(person), np.array(types)

    def knn_pairs(pts, k=8):
        d = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
        np.fill_diagonal(d, np.inf)
        pairs = set()
        for i in range(len(pts)):
            for j in np.argsort(d[i])[:k]:
                pairs.add((min(i, int(j)), max(i, int(j))))
        return pairs

    best, best_cross = None, -1
    for i in range(len(adapter)):
        smp = adapter[i]
        if smp["num_people"] != 2:
            continue
        vis = [(kp[:, 3] > 0).sum().item() for kp in smp["keypoints"]]
        if min(vis) < 12:
            continue
        pts, person, types = scene_arrays(smp)
        cross = sum(1 for a, b in knn_pairs(pts) if person[a] != person[b])
        if cross > best_cross:
            best, best_cross = smp, cross
    sample = best
    assert sample is not None, "no suitable 2-person scene found"

    pts, person, types = scene_arrays(sample)

    ax = axes[1]
    pairs = knn_pairs(pts)
    same_type_cross = None
    for a, b in sorted(pairs):
        cross = person[a] != person[b]
        col = "#c23b3b" if cross else "0.75"
        lw = 1.0 if cross else 0.5
        ax.plot(*zip(pts[a], pts[b]), "-", color=col, lw=lw,
                zorder=2 if cross else 1, alpha=0.9 if cross else 1.0)
        if cross and types[a] == types[b] and same_type_cross is None:
            same_type_cross = (a, b)
    colors = ["#2b5fa3", "#3c8a4e"]
    for pid in (0, 1):
        m = person == pid
        ax.plot(pts[m, 0], pts[m, 1], "o", color=colors[pid], ms=5, zorder=3,
                label=f"person {pid + 1}")
    if same_type_cross is not None:
        a, b = same_type_cross
        ax.plot(*zip(pts[a], pts[b]), "-", color="#c23b3b", lw=2.0, zorder=4)
        mid = pts[[a, b]].mean(axis=0)
        ax.annotate("same-type cross-person edge", mid,
                    textcoords="offset points", xytext=(10, -10),
                    fontsize=8, color="#c23b3b")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("(b) $k$NN graph ($k=8$) on a two-person scene", fontsize=10)

    fig.savefig(FIGDIR / "skeleton_graph.pdf")
    plt.close(fig)
    print("wrote skeleton_graph.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: 2x2 grid of synthetic scenes, K = 2..5
# ─────────────────────────────────────────────────────────────────────────────

def fig_synth_examples():
    sys.path.insert(0, str(Path.cwd()))
    from virtual_adapter import VirtualAdapter  # noqa: E402
    import cv2

    adapter = VirtualAdapter(data_dir=Path("data/virtual/test"))
    wanted = {2: None, 3: None, 4: None, 5: None}
    for i in range(len(adapter)):
        s = adapter[i]
        kcount = s["num_people"]
        if kcount in wanted and wanted[kcount] is None:
            wanted[kcount] = s
        if all(v is not None for v in wanted.values()):
            break

    colors = ["#c23b3b", "#2b5fa3", "#3c8a4e", "#b58a2a", "#7a4fa3"]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    for ax, kcount in zip(axes.flat, sorted(wanted)):
        s = wanted[kcount]
        img = s["image"]
        if hasattr(img, "numpy"):
            img = img.numpy()
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        if img.max() <= 1.5:
            img = (img * 255).astype(np.uint8)
        img = img.astype(np.uint8)
        ax.imshow(img)
        for pid, kp in enumerate(s["keypoints"]):
            kp = kp.numpy()
            vis = kp[:, 3] > 0
            for a, b in COCO_SKELETON:
                if vis[a] and vis[b]:
                    ax.plot([kp[a, 0], kp[b, 0]], [kp[a, 1], kp[b, 1]],
                            "-", color=colors[pid % 5], lw=1.0)
            ax.plot(kp[vis, 0], kp[vis, 1], "o", color=colors[pid % 5], ms=2.5)
        allkp = np.concatenate([kp.numpy()[kp.numpy()[:, 3] > 0][:, :2]
                                for kp in s["keypoints"]])
        x0, y0 = allkp.min(axis=0); x1, y1 = allkp.max(axis=0)
        mx, my = 0.25 * (x1 - x0) + 10, 0.25 * (y1 - y0) + 10
        H, W = img.shape[:2]
        ax.set_xlim(max(0, x0 - mx), min(W, x1 + mx))
        ax.set_ylim(min(H, y1 + my), max(0, y0 - my))
        ax.set_title(f"$K = {kcount}$", fontsize=10)
        ax.axis("off")
    fig.savefig(FIGDIR / "synth_examples.pdf")
    plt.close(fig)
    print("wrote synth_examples.pdf")





# ─────────────────────────────────────────────────────────────────────────────
# Figures 4 & 5: sweep landscapes fetched from W&B (cached locally)
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_CACHE = Path(__file__).resolve().parent / "sweep_data.json"
ENTITY_PROJECT = "deanmills/multiperson-pose-grouping"
ARCH_SWEEP, FT_SWEEP = "wg95w0k0", "i9vpg4t1"
ARCH_WINNER, FT_WINNER = "08d0ffde", "3362f752"  # run names (= output GUIDs)


def _fetch_sweep_data():
    """Fetch per-run config + best_val_pga for both sweeps; cache to JSON."""
    if SWEEP_CACHE.exists():
        return json.loads(SWEEP_CACHE.read_text())
    import wandb
    api = wandb.Api()
    data = {}
    for sweep_id in (ARCH_SWEEP, FT_SWEEP):
        rows = []
        for r in api.sweep(f"{ENTITY_PROJECT}/{sweep_id}").runs:
            pga = r.summary.get("best_val_pga")
            if pga is None:
                continue
            rows.append({"name": r.name, "state": r.state, "pga": float(pga),
                         "config": {k: v for k, v in r.config.items()
                                    if k.startswith(("sa_gat__", "training__"))}})
        data[sweep_id] = rows
    SWEEP_CACHE.write_text(json.dumps(data, indent=1))
    return data


def fig_arch_sweep():
    data = _fetch_sweep_data()[ARCH_SWEEP]
    axes_spec = [
        ("sa_gat__num_layers", "layers"),
        ("sa_gat__num_heads", "heads"),
        ("sa_gat__hidden_dim", "hidden dim"),
        ("sa_gat__output_dim", "output dim"),
        ("sa_gat__joint_embedding_dim", "joint emb.\\ dim"),
        ("sa_gat__dropout", "dropout"),
    ]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.2), sharey=True)
    for ax, (key, label) in zip(axes.flat, axes_spec):
        vals = sorted({row["config"][key] for row in data})
        pos = {v: i for i, v in enumerate(vals)}
        for row in data:
            x = pos[row["config"][key]] + rng.uniform(-0.13, 0.13)
            win = row["name"] == ARCH_WINNER
            ax.plot(x, row["pga"], "o",
                    color="#c23b3b" if win else "0.45",
                    ms=7 if win else 4, zorder=3 if win else 2,
                    alpha=1.0 if win else 0.75)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"{v:g}" for v in vals], fontsize=9)
        ax.set_xlabel(label.replace("\\ ", " "), fontsize=10)
        ax.tick_params(labelsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Synth val PGA")
    fig.tight_layout()
    fig.savefig(FIGDIR / "arch_sweep.pdf")
    plt.close(fig)
    print("wrote arch_sweep.pdf")


def fig_ft_sweep():
    data = _fetch_sweep_data()[FT_SWEEP]
    epoch_colors = {10: "#8fb4d9", 15: "#2b5fa3", 20: "#b58a2a", 30: "#c23b3b"}
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    seen = set()
    for row in data:
        lr = row["config"]["training__lr"]
        ep = int(row["config"]["training__epochs"])
        win = row["name"] == FT_WINNER
        lbl = f"{ep} epochs" if ep not in seen else None
        seen.add(ep)
        ax.plot(lr, row["pga"], "o", color=epoch_colors.get(ep, "0.4"),
                ms=9 if win else 5.5, zorder=3 if win else 2, label=lbl,
                markeredgecolor="black" if win else "none", markeredgewidth=1.2)
        if win:
            ax.annotate("winner\n(\\texttt{meng-headline})" if False else "winner",
                        (lr, row["pga"]), textcoords="offset points",
                        xytext=(8, -3), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Fine-tune learning rate")
    ax.set_ylabel("Synth val PGA after fine-tune")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(FIGDIR / "ft_sweep.pdf")
    plt.close(fig)
    print("wrote ft_sweep.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Chapter 1 motivating example — ungrouped detections vs grouped
# ─────────────────────────────────────────────────────────────────────────────

COCO_DIR = Path("data/coco2017")
CONTEXT_IMG_ID = 482800  # val2017, 3 people, two of them near-coincident


def fig_context_example():
    """Real COCO scene shown twice: as the detector emits it, and after grouping.

    Image 482800 was chosen from val2017 by scanning for scenes with two to
    three well-annotated people whose bounding boxes overlap without heavy
    occlusion and whose same-type keypoints fall close together. The two
    right-hand figures satisfy this; the left-hand figure is the easy case.
    """
    from PIL import Image  # noqa: E402

    ann_file = COCO_DIR / "annotations/person_keypoints_val2017.json"
    d = json.load(open(ann_file))
    im = next(i for i in d["images"] if i["id"] == CONTEXT_IMG_ID)
    anns = [a for a in d["annotations"]
            if a["image_id"] == CONTEXT_IMG_ID and not a["iscrowd"]
            and a["num_keypoints"] >= 12]
    kps = [np.array(a["keypoints"]).reshape(17, 3).astype(float) for a in anns]
    pil = Image.open(COCO_DIR / "val2017" / im["file_name"]).convert("RGB")

    # Crop to the people with a margin, so keypoints stay legible in print.
    pts = np.vstack([k[k[:, 2] > 0][:, :2] for k in kps])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    # Padding is asymmetric on purpose: two landscape panels side by side would
    # render below the 8 cm minimum figure height at \linewidth, so each panel is
    # kept close to portrait.
    padx, pady = 0.03 * (x1 - x0), 0.18 * (y1 - y0)
    x0, x1 = max(0, x0 - padx), min(im["width"], x1 + padx)
    y0, y1 = max(0, y0 - pady), min(im["height"], y1 + pady)

    # Closest same-type cross-person pair — the ambiguity to call out in (a).
    close = None
    best = np.inf
    for i in range(len(kps)):
        for j in range(i + 1, len(kps)):
            for t in range(17):
                if kps[i][t, 2] > 0 and kps[j][t, 2] > 0:
                    dist = np.linalg.norm(kps[i][t, :2] - kps[j][t, :2])
                    if dist < best:
                        best, close = dist, (i, j, t)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    for ax in axes:
        ax.imshow(pil)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.axis("off")

    # (a) what the detector emits: typed keypoints with no person identity
    ax = axes[0]
    ax.plot(pts[:, 0], pts[:, 1], "o", color="black", ms=5.5,
            mec="white", mew=1.0, linestyle="none", zorder=3)
    if close is not None:
        i, j, t = close
        cx = float(np.mean([kps[i][t, 0], kps[j][t, 0]]))
        cy = float(np.mean([kps[i][t, 1], kps[j][t, 1]]))
        r = min(48.0, max(20.0, 0.9 * best))
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, color="#d68910",
                                lw=1.6, zorder=4))
        name = JOINT_NAMES[t].replace("L ", "left ").replace("R ", "right ")
        ax.text(cx, cy - r - 8, f"two {name}s", fontsize=10, color="#8a5606",
                ha="center", va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
    ax.set_title("(a) keypoints without person identity", fontsize=11)

    # (b) what grouping must recover
    ax = axes[1]
    # Persons 0 and 2 are the near-coincident pair, so they take the two most
    # contrasting colours; person 1 stands clear of both.
    colors = ["#2b5fa3", "#3c8a4e", "#c23b3b"]
    for pid, kp in enumerate(kps):
        c = colors[pid % len(colors)]
        for u, v in COCO_SKELETON:
            if kp[u, 2] > 0 and kp[v, 2] > 0:
                ax.plot([kp[u, 0], kp[v, 0]], [kp[u, 1], kp[v, 1]], "-",
                        color=c, lw=1.6, zorder=2)
        m = kp[:, 2] > 0
        ax.plot(kp[m, 0], kp[m, 1], "o", color=c, ms=5.5, mec="white",
                mew=1.0, zorder=3)
    ax.set_title("(b) keypoints assigned to person identities", fontsize=11)

    fig.savefig(FIGDIR / "context_example.pdf")
    plt.close(fig)
    print("wrote context_example.pdf")


if __name__ == "__main__":
    fig_predk_recall()
    fig_skeleton_graph()
    fig_synth_examples()
    fig_arch_sweep()
    fig_ft_sweep()
    fig_context_example()
