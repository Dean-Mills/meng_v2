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




# ─────────────────────────────────────────────────────────────────────────────
# Figure: synthetic-to-real transfer scatter (Chapter 5, Section 5.2)
# ─────────────────────────────────────────────────────────────────────────────

def fig_transfer_scatter():
    """Synthetic-validation PGA vs COCO-validation PGA for every configuration
    reported in Chapter 4 that has both numbers (27 runs). Values are the
    Chapter 4 table cells (docs/results.md canonical); W&B run ids in comments.
    COCO = k-means on ground-truth keypoints, 2,307 scenes (TriGAT: 2,180).
    """
    # family, label, synth val, COCO val, fine-tuned?, annotate?
    pts = [
        ("Standard GAT", "depth on",            0.9990, 0.8366, False, "depth on"),          # su8rmvti
        ("Standard GAT", "depth off (baseline)", 0.8783, 0.8841, False, "baseline"),         # jkd1ln4a
        ("Learned head", "DEC",                 0.8608, 0.8798, False, "DEC"),               # 5q0p9it9
        ("Learned head", "slot attention",      0.9025, 0.8254, False, "slot attention"),    # 2kq3i8yv
        ("Learned head", "graph partitioning",  0.9714, 0.8366, False, "graph partitioning"),# s3qr04xm
        ("PG-GAT (synthetic only)", "type-pair only", 0.8825, 0.8819, False, None),          # 8peqkcyh
        ("PG-GAT (synthetic only)", "pos-enc only",   0.8825, 0.8884, False, None),          # ffl2tde3
        ("PG-GAT (synthetic only)", "repulsion only", 0.8834, 0.8785, False, None),          # gfqkbqnw
        ("PG-GAT (synthetic only)", "all three, 2-layer", 0.8896, 0.8868, False, "PG-GAT, 2-layer"),  # r5ekg0zl
        ("PG-GAT (synthetic only)", "architecture-sweep winner", 0.9634, 0.9010, False, "sweep winner"),  # 08d0ffde
        ("Hyperbolic PG-GAT", "H1, d=128",      0.8773, 0.8919, False, None),                # aox7h3di
        ("Hyperbolic PG-GAT", "H2, d=32",       0.8856, 0.8969, False, "hyperbolic d=32"),   # g45tdxi5
        ("Hyperbolic PG-GAT", "sweep winner",   0.8795, 0.8876, False, None),                # ua4wdb3p
        ("TriGAT", "triplets",                  0.8740, 0.8564, False, "TriGAT"),            # y0tw6vc9
        ("COCO fine-tuned", "legacy 20-epoch",  0.9178, 0.9565, True,  "20-epoch fine-tune"),# ugzbgepn
        ("COCO fine-tuned", "headline (fine-tune sweep)", 0.9634, 0.9715, True, "headline"), # b5noyg7t
        ("COCO fine-tuned", "hyperbolic Option B", 0.8900, 0.9399, True, "hyperbolic, fine-tuned"),  # lragtfzu
    ]
    sadmon = [  # vanilla, v1, v1 clamped, v1 lambda=1, v1 fixed sigma, v1 no-spectral, v2a-d
        (0.7879, 0.8657), (0.7872, 0.8658), (0.7918, 0.8649), (0.7892, 0.8628), (0.7769, 0.8642),
        (0.7889, 0.8684), (0.7849, 0.8622), (0.7925, 0.8677), (0.7934, 0.8660), (0.7910, 0.8652),
    ]
    style = {
        "Standard GAT":            dict(color="0.25", marker="s"),
        "Learned head":            dict(color="#d68910", marker="^"),
        "SA-DMoN (10 variants)":   dict(color="#5b9bd5", marker="o"),
        "PG-GAT (synthetic only)": dict(color="#c23b3b", marker="o"),
        "Hyperbolic PG-GAT":       dict(color="#7d3c98", marker="D"),
        "TriGAT":                  dict(color="#3c8a4e", marker="v"),
        "COCO fine-tuned":         dict(color="#c23b3b", marker="*"),
    }

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    lo, hi = 0.76, 1.005
    ax.plot([lo, hi], [lo, hi], "--", color="0.6", lw=1.0, zorder=1)
    ax.text(0.836, 0.8385, "COCO = synthetic", fontsize=8.5, color="0.45",
            rotation=40, ha="left", va="bottom", rotation_mode="anchor")

    sx, sy = zip(*sadmon)
    ax.scatter(sx, sy, s=22, zorder=3, label="SA-DMoN (10 variants)", **style["SA-DMoN (10 variants)"])
    ax.annotate("SA-DMoN ×10", (float(np.mean(sx)), float(np.mean(sy))), textcoords="offset points",
                xytext=(-6, 14), fontsize=8.5, ha="center")

    seen = set()
    for fam, name, x, y, ft, ann in pts:
        st = style[fam]
        kw = dict(color=st["color"], marker=st["marker"], zorder=4,
                  s=110 if ft else 40, edgecolors="white" if ft else "none", linewidths=0.6)
        ax.scatter([x], [y], label=fam if fam not in seen else None, **kw)
        seen.add(fam)
        if ann:
            off = {
                "depth on": (-6, -12), "baseline": (8, -11), "DEC": (-8, -11),
                "slot attention": (0, -12), "graph partitioning": (-8, 5),
                "PG-GAT, 2-layer": (8, 4), "sweep winner": (8, -4),
                "hyperbolic d=32": (8, 4), "TriGAT": (8, -4),
                "20-epoch fine-tune": (8, 4), "headline": (8, 2), "hyperbolic, fine-tuned": (8, -3),
            }.get(ann, (7, 3))
            ha = "right" if off[0] < 0 else ("center" if off[0] == 0 else "left")
            ax.annotate(ann, (x, y), textcoords="offset points", xytext=off, fontsize=8.5, ha=ha)

    # fine-tuning arrows: synthetic-only parent -> fine-tuned child
    for (x0, y0), (x1, y1) in [((0.8896, 0.8868), (0.9178, 0.9565)),
                               ((0.9634, 0.9010), (0.9634, 0.9715)),
                               ((0.8856, 0.8969), (0.8900, 0.9399))]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="0.5", lw=0.9,
                                    shrinkA=5, shrinkB=7, mutation_scale=9))
    ax.text(0.9675, 0.934, "COCO\nfine-tuning", fontsize=8.5, color="0.4", ha="left", va="center")

    ax.set_xlim(lo, hi)
    ax.set_ylim(0.815, 0.985)
    ax.set_xlabel("synthetic validation PGA")
    ax.set_ylabel("COCO validation PGA (ground-truth keypoints, $k$-means)")
    ax.legend(loc="upper left", fontsize=8.5, frameon=False, handletextpad=0.4)
    ax.grid(True, color="0.92", lw=0.6)
    fig.savefig(FIGDIR / "transfer_scatter.pdf")
    plt.close(fig)
    print("wrote transfer_scatter.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figures from the per-image end-to-end dump (code_v3/eval_e2e_per_image.py)
# ─────────────────────────────────────────────────────────────────────────────

import os
E2E_DIR = Path(os.environ.get("E2E_DIR", "outputs/e2e_per_image"))
PERSON_COLOURS = ["#2b5fa3", "#c23b3b", "#3c8a4e", "#d68910", "#7d3c98", "#17a2b8",
                  "#8b4513", "#e83e8c", "#6c757d", "#20c997"]


def _load_e2e_records():
    return [json.loads(l) for l in open(E2E_DIR / "per_image.jsonl")]


def _panel_labels_to_colours(labels, gt_person):
    """Map cluster labels to ground-truth person indices by Hungarian matching so the
    same person keeps the same colour across panels; surplus clusters get new colours.
    Returns (label -> colour index, PGA of this labelling)."""
    from scipy.optimize import linear_sum_assignment
    L, P = np.unique(labels), np.unique(gt_person)
    C = np.array([[np.sum((labels == l) & (gt_person == p)) for p in P] for l in L], dtype=float)
    r, c = linear_sum_assignment(-C)
    lab2col = {int(L[i]): int(c_i) for i, c_i in zip(r, c)}
    nxt = len(P)
    for l in L:
        if int(l) not in lab2col:
            lab2col[int(l)] = nxt
            nxt += 1
    return lab2col, float(C[r, c].sum() / len(labels))


def _draw_grouping(ax, img, pos, types, labels, gt_person, title, crop):
    lab2col, pga = _panel_labels_to_colours(labels, gt_person)
    ax.imshow(img)
    x0, y0, x1, y1 = crop
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.axis("off")
    for l in np.unique(labels):
        col = PERSON_COLOURS[lab2col[int(l)] % len(PERSON_COLOURS)]
        m = labels == l
        lookup = {int(t): p for t, p in zip(types[m], pos[m])}
        for u, v in COCO_SKELETON:
            if u in lookup and v in lookup:
                ax.plot([lookup[u][0], lookup[v][0]], [lookup[u][1], lookup[v][1]], "-",
                        color=col, lw=1.4, alpha=0.9, zorder=2)
        ax.plot(pos[m, 0], pos[m, 1], "o", color=col, ms=3.6, mec="white", mew=0.6,
                linestyle="none", zorder=3)
    ax.set_title(f"{title}, PGA {pga:.2f}", fontsize=8.5, pad=2)
    return pga


def fig_e2e_examples(image_ids=None, out_name="e2e_examples.pdf"):
    """Rows = COCO val scenes, columns = (a) HigherHRNet native AE grouping,
    (b) PG-GAT + k-means with oracle K, (c) PG-GAT + k-means with the K-head's K.
    Only the detections matched to ground truth (the scored set) are drawn, coloured by
    the ground-truth person their cluster was assigned to; a joint in another person's
    colour is a grouping error. Reads outputs/e2e_per_image/arrays/<id>.npz."""
    from PIL import Image  # noqa: E402
    recs = {r["image_id"]: r for r in _load_e2e_records()}
    if image_ids is None:
        image_ids = E2E_EXAMPLE_IDS
    scenes = []
    for img_id in image_ids:
        a = np.load(E2E_DIR / "arrays" / f"{img_id}.npz", allow_pickle=True)
        img = Image.open(COCO_DIR / "val2017" / str(a["file_name"])).convert("RGB")
        m = a["matched_idx"]
        pos = a["det_pos"][m]
        lo, hi = pos.min(axis=0), pos.max(axis=0)
        w, h = hi - lo
        padx, pady = 0.10 * w + 12, 0.10 * h + 12
        crop = (max(0, lo[0] - padx), max(0, lo[1] - pady),
                min(img.width, hi[0] + padx), min(img.height, hi[1] + pady))
        scenes.append((img_id, a, img, crop))
    # panel width is fixed by the text width; each row's height follows its crop's aspect
    panel_w = (6.1 - 0.25) / 3
    heights = [panel_w * (c[3] - c[1]) / (c[2] - c[0]) for _, _, _, c in scenes]
    title_h = 0.22
    fig_h = sum(heights) + title_h * len(scenes) + 0.05
    fig = plt.figure(figsize=(6.1, fig_h))
    gs = fig.add_gridspec(len(scenes), 3, height_ratios=[h + title_h for h in heights],
                          left=0.035, right=0.995, top=1 - 0.02 / fig_h, bottom=0.02 / fig_h,
                          wspace=0.03, hspace=0.06)
    for r, (img_id, a, img, crop) in enumerate(scenes):
        rec = recs[img_id]
        m = a["matched_idx"]
        pos, types, gt_person = a["det_pos"][m], a["det_types"][m], a["gt_person"]
        panels = [
            (a["ae_ids"][m], "(a) HigherHRNet AE"),
            (a["labels_oracle"], f"(b) PG-GAT, oracle $K={rec['K_gt']}$"),
            (a["labels_pred"], f"(c) PG-GAT, $\\hat{{K}}={rec['K_pred']}$"),
        ]
        for c, (labels, title) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            _draw_grouping(ax, img, pos, types, np.asarray(labels), gt_person, title, crop)
            if c == 0:
                ax.text(-0.02, 0.5, f"COCO {img_id}", transform=ax.transAxes, rotation=90,
                        fontsize=8.5, ha="right", va="center", color="0.35")
    fig.savefig(FIGDIR / out_name)
    plt.close(fig)
    print(f"wrote {out_name} for images {list(image_ids)}")


# Scenes chosen 2026-09-03 from the per-image dump (2-5 annotated people, >=20 scored detections):
#   281759  five people, all three groupings perfect (crowded success)
#   480021  K-head correct (K=4), PG-GAT misassigns joints at oracle K (0.89) while AE is perfect
#           (a grouping error, not a count error). Replaced 576031 (same criterion, near-square crop)
#           on 2026-09-03 because the four-row figure plus caption overflowed the page by 58 pt.
#   305309  third annotated person barely detected: oracle K=3 splits a person (0.71),
#           the K-head's K=2 is perfect (under-count that helps)
#   547886  K-head under-counts (K=3 -> 2), merging two people (1.00 -> 0.65; under-count that hurts)
# select_e2e_examples() reproduces the shortlist these were taken from.
E2E_EXAMPLE_IDS = [281759, 480021, 305309, 547886]


def select_e2e_examples(recs):
    """Pick three scored scenes with 2-5 people: a crowded success (all three perfect),
    a modularity-cost case (AE perfect, PG-GAT oracle-K clearly worse), and a case where the
    K-head's under-count beats oracle K (the effective-person-count mechanism)."""
    ok = [r for r in recs.values() if r["pga_pred"] is not None and 2 <= r["K_gt"] <= 5 and r["n_matched"] >= 20]
    success = sorted([r for r in ok if r["pga_ae"] == 1 and r["pga_oracle"] == 1 and r["pga_pred"] == 1 and r["K_gt"] >= 3],
                     key=lambda r: -r["n_matched"])
    cost = sorted([r for r in ok if r["pga_ae"] >= 0.98 and r["pga_oracle"] <= 0.85],
                  key=lambda r: r["pga_oracle"] - r["pga_ae"])
    predk = sorted([r for r in ok if r["K_pred"] < r["K_gt"] and r["pga_pred"] - r["pga_oracle"] >= 0.10],
                   key=lambda r: -(r["pga_pred"] - r["pga_oracle"]))
    picks = []
    for cand in (success, cost, predk):
        for r in cand:
            if r["image_id"] not in picks:
                picks.append(r["image_id"])
                break
    return picks


def fig_k_confusion(out_name="k_confusion.pdf", kmax=8):
    """Confusion matrix of the K-head's predicted person count against the annotated count
    on HigherHRNet detections (the scored oracle-K set). Counts >= kmax are pooled."""
    recs = [r for r in _load_e2e_records() if r["pga_oracle"] is not None and r["K_pred"] is not None]
    kg_raw = np.array([r["K_gt"] for r in recs])
    kp_raw = np.array([r["K_pred"] for r in recs])
    exact = float(np.mean(kg_raw == kp_raw))            # on the raw counts (matches Ch4)
    off1 = float(np.mean(np.abs(kg_raw - kp_raw) <= 1))
    kg, kp = np.minimum(kg_raw, kmax), np.minimum(kp_raw, kmax)  # pooled only for display
    M = np.zeros((kmax, kmax), dtype=int)
    for g, p in zip(kg, kp):
        M[g - 1, p - 1] += 1
    fig, ax = plt.subplots(figsize=(3.9, 3.5))
    ax.imshow(M, cmap="Blues", vmin=0, vmax=max(M.max(), 1))
    for i in range(kmax):
        for j in range(kmax):
            if M[i, j]:
                ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.55 * M.max() else "black")
    ticks = [str(k) for k in range(1, kmax)] + [f"{kmax}+"]
    ax.set_xticks(range(kmax)); ax.set_xticklabels(ticks)
    ax.set_yticks(range(kmax)); ax.set_yticklabels(ticks)
    ax.set_xlabel("predicted person count $\\hat{K}$")
    ax.set_ylabel("annotated person count $K$")
    ax.set_title(f"exact {exact:.1%}, within one {off1:.1%} ($n={len(recs)}$)", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    ax.tick_params(length=0)
    fig.savefig(FIGDIR / out_name)
    plt.close(fig)
    print(f"wrote {out_name}: exact {exact:.4f} off1 {off1:.4f} n {len(recs)} mean pred {kp_raw.mean():.3f} gt {kg_raw.mean():.3f}")


FIGURES = {
    "predk_recall": fig_predk_recall,
    "skeleton_graph": fig_skeleton_graph,
    "synth_examples": fig_synth_examples,
    "arch_sweep": fig_arch_sweep,
    "ft_sweep": fig_ft_sweep,
    "context_example": fig_context_example,
    "transfer_scatter": fig_transfer_scatter,
    "e2e_examples": fig_e2e_examples,
    "k_confusion": fig_k_confusion,
}

if __name__ == "__main__":
    # No arguments: regenerate everything. Otherwise only the named figures.
    for name in (sys.argv[1:] or list(FIGURES)):
        FIGURES[name]()
