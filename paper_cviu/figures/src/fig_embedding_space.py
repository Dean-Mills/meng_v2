"""Figure: "the embedding is the lever" - one COCO scene, three checkpoints, the embedding space.

Columns: (0) the scene with ground-truth person colours; (1-3) standard GAT baseline -> PG-GAT
synthetic-only -> PG-GAT COCO fine-tuned. Top row: 2-D projection of the per-keypoint embeddings,
coloured by ground-truth person; keypoints the k-means read-out assigns to the wrong person are
ringed. Bottom row: cosine-similarity matrix of the same embeddings, nodes ordered by person, so
the block-diagonal structure sharpening left to right is the same fact without a projection.

Scene selection (stated, reproducible): COCO val2017 scenes with 3-5 people, every person with
>= 10 labelled keypoints; among them the scene whose per-scene PGA triple is closest (L1) to the
dataset-level COCO val PGA of the three checkpoints (0.8841 / 0.9010 / 0.9715), so the panel is
representative rather than the most dramatic. Override with --img_id.

Run from code_v3/ (repo imports):
    python ../paper_cviu/figures/src/fig_embedding_space.py --proj pca
    python ../paper_cviu/figures/src/fig_embedding_space.py --proj tsne
Writes paper_cviu/figures/embedding_space_<proj>.pdf and .png (preview) and a JSON sidecar.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path.cwd()))
from config import ExperimentConfig            # noqa: E402
from preprocessor import PosePreprocessor       # noqa: E402
from evaluator import compute_pga, predict_knn  # noqa: E402
from dataset import PoseDataset                 # noqa: E402
from coco_adapter import CocoAdapter            # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent          # paper_cviu/figures/
CHECKPOINTS = [   # (label, path, dataset-level COCO val PGA, W&B run)
    ("Standard GAT",            "outputs/train_knn_no_depth/4ca258c4/best.pt",   0.8841, "jkd1ln4a"),
    ("PG-GAT, synthetic only",  "outputs/pg_gat_arch_sweep/08d0ffde/best.pt",    0.9010, "08d0ffde"),
    ("PG-GAT, COCO fine-tuned", "outputs/pg_gat_finetune_sweep/3362f752/best.pt", 0.9715, "b5noyg7t"),
]
PERSON_COLORS = ["#c23b3b", "#2b5fa3", "#3c8a4e", "#b58a2a", "#7a4fa3"]
COCO_SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
                 (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def load_gat(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ExperimentConfig(**ckpt["config"])
    assert cfg.sa_gat_hyperbolic is None and cfg.sa_gat_v2 is None
    if cfg.sa_gat is not None:
        from sa_gat import SAGATEmbedding
        gat, dim, use_depth = SAGATEmbedding(cfg.sa_gat), cfg.sa_gat.output_dim, cfg.sa_gat.use_depth
    else:
        from gat import GATEmbedding
        gat, dim, use_depth = GATEmbedding(cfg.gat), cfg.gat.output_dim, cfg.gat.use_depth
    assert not use_depth, "figure assumes the no-depth node features"
    gat = gat.to(device); gat.load_state_dict(ckpt["gat_state"]); gat.eval()
    pre = PosePreprocessor(device=device, k_neighbors=16 if dim >= 256 else 8, use_depth=False)
    return gat, pre, dim


def cluster_to_person(labels: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Hungarian mapping cluster -> person (the PGA convention); returns the mapped person per node."""
    k, p = labels.max() + 1, gt.max() + 1
    conf = np.zeros((k, p), int)
    for l, g in zip(labels, gt):
        conf[l, g] += 1
    r, c = linear_sum_assignment(-conf)
    m = {int(a): int(b) for a, b in zip(r, c)}
    return np.array([m.get(int(l), -1) for l in labels])


def project(emb: np.ndarray, how: str) -> np.ndarray:
    if how == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=0).fit_transform(emb)
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=min(30, max(5, len(emb) // 4)), init="pca",
                random_state=0, metric="cosine").fit_transform(emb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_img_dir", type=Path, default=Path("data/coco2017/val2017"))
    ap.add_argument("--coco_ann_file", type=Path, default=Path("data/coco2017/annotations/person_keypoints_val2017.json"))
    ap.add_argument("--img_id", type=str, default=None, help="COCO image id; default = representative scene (see docstring)")
    ap.add_argument("--proj", choices=["pca", "tsne", "both"], default="pca")
    ap.add_argument("--min_kps", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    models = [(lab, *load_gat(p, a.device), mean, run) for lab, p, mean, run in CHECKPOINTS]
    adapter = CocoAdapter(img_dir=a.coco_img_dir, ann_file=a.coco_ann_file, min_people=3, max_people=5,
                          device="cpu", use_depth=False)
    ds = PoseDataset(adapter)

    def scene(idx):
        s = ds[idx]                                   # transformed to 512, keypoints scaled to match
        kps = s["keypoints"]
        if any((k[:, 3] > 0).sum() < a.min_kps for k in kps):
            return None
        g = models[-1][2].create_graph(kps)           # headline preprocessor (k = 16)
        if g is None:
            return None
        gt = g.person_labels.cpu().numpy()
        out = {"idx": idx, "img_id": s["img_id"], "image": s["image"], "kps": kps, "gt": gt,
               "types": g.joint_types.cpu().numpy(), "K": int(s["num_people"]), "emb": [], "labels": [], "pga": []}
        with torch.no_grad():
            for lab, gat, pre, dim, mean, run in models:
                gg = pre.create_graph(kps).to(a.device)
                e = gat(gg)
                l = predict_knn(e, out["K"])
                out["pga"].append(float(compute_pga(l, gg.person_labels)))
                out["emb"].append(e.cpu().numpy()); out["labels"].append(l.cpu().numpy())
        return out

    if a.img_id is not None:
        idx = [i for i, iid in enumerate(adapter.img_ids) if str(iid) == a.img_id]
        assert idx, f"image {a.img_id} not in the 3-5-person set"
        chosen = scene(idx[0]); assert chosen is not None, "scene fails the >= min_kps rule"
    else:
        means = np.array([m[4] for m in models])
        best, best_d, n_cand = None, 1e9, 0
        for i in range(len(ds)):
            sc = scene(i)
            if sc is None:
                continue
            n_cand += 1
            d = float(np.abs(np.array(sc["pga"]) - means).sum())
            if d < best_d:
                best, best_d = sc, d
        chosen = best
        print(f"candidates {n_cand}; chosen image {chosen['img_id']} (L1 distance to dataset means {best_d:.4f})")
    print(f"image {chosen['img_id']}: K = {chosen['K']}, nodes = {len(chosen['gt'])}, "
          f"per-scene PGA = {['%.4f' % p for p in chosen['pga']]}")

    # ── figure ───────────────────────────────────────────────────────────
    gt, K = chosen["gt"], chosen["K"]
    order = np.argsort(gt, kind="stable")
    bounds = np.cumsum([np.sum(gt == p) for p in range(K)])
    projs = ["pca", "tsne"] if a.proj == "both" else [a.proj]
    nrows = len(projs) + 1
    fig = plt.figure(figsize=(7.2, 1.9 * nrows + 0.2))
    gs = fig.add_gridspec(nrows, 4, width_ratios=[1.45, 1, 1, 1], height_ratios=[1] * nrows,
                          left=0.01, right=0.93, top=0.92, bottom=0.03, wspace=0.22, hspace=0.38)

    ax = fig.add_subplot(gs[:, 0])
    img = chosen["image"].permute(1, 2, 0).numpy()
    ax.imshow(img)
    for pid, kp in enumerate(chosen["kps"]):
        kp = kp.numpy(); col = PERSON_COLORS[pid % 5]
        for i, j in COCO_SKELETON:
            if kp[i, 3] > 0 and kp[j, 3] > 0:
                ax.plot([kp[i, 0], kp[j, 0]], [kp[i, 1], kp[j, 1]], "-", color=col, lw=1.0, alpha=0.9)
        v = kp[:, 3] > 0
        ax.plot(kp[v, 0], kp[v, 1], "o", color=col, ms=2.6, mec="white", mew=0.4)
    allk = np.concatenate([k.numpy()[k.numpy()[:, 3] > 0, :2] for k in chosen["kps"]])
    pad = 0.12 * (allk.max(axis=0) - allk.min(axis=0)).max()
    x0, y0 = np.maximum(allk.min(axis=0) - pad, 0); x1, y1 = allk.max(axis=0) + pad
    ax.set_xlim(x0, min(x1, img.shape[1])); ax.set_ylim(min(y1, img.shape[0]), y0); ax.axis("off")
    ax.set_title(f"COCO val2017 {chosen['img_id']}\n$K={K}$, {len(gt)} labelled keypoints", fontsize=8)

    for c, (lab, gat, pre, dim, mean, run) in enumerate(models, start=1):
        emb, labels, pga = chosen["emb"][c - 1], chosen["labels"][c - 1], chosen["pga"][c - 1]
        mapped = cluster_to_person(labels, gt)
        wrong = mapped != gt
        for r, pr in enumerate(projs):
            xy = project(emb, pr)
            axt = fig.add_subplot(gs[r, c])
            for p in range(K):
                m = gt == p
                axt.scatter(xy[m, 0], xy[m, 1], s=11, color=PERSON_COLORS[p % 5], alpha=0.9, linewidths=0, zorder=3)
            if wrong.any():
                axt.scatter(xy[wrong, 0], xy[wrong, 1], s=40, facecolors="none", edgecolors="black", linewidths=0.7, zorder=4)
            axt.set_xticks([]); axt.set_yticks([]); axt.set_aspect("equal", adjustable="datalim")
            axt.margins(0.15)
            for sp in axt.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color("0.6")
            if r == 0:
                axt.set_title(f"{lab}\nval PGA {mean:.4f} $\\cdot$ scene {pga:.3f}", fontsize=8)
            axt.text(0.5, -0.04, f"{int(wrong.sum())} of {len(gt)} misassigned", transform=axt.transAxes,
                     fontsize=8, color="0.25", ha="center", va="top")
            if c == 1:
                axt.set_ylabel(("PCA" if pr == "pca" else "t-SNE") + " of the\nkeypoint embeddings", fontsize=8, labelpad=3)

        axb = fig.add_subplot(gs[nrows - 1, c])
        sim = (emb @ emb.T)[np.ix_(order, order)]
        im = axb.imshow(sim, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
        for b in bounds[:-1]:
            axb.axhline(b - 0.5, color="white", lw=0.6); axb.axvline(b - 0.5, color="white", lw=0.6)
        axb.set_xticks([]); axb.set_yticks([])
        starts = np.concatenate([[0], bounds[:-1]])
        for p, (s0, s1) in enumerate(zip(starts, bounds)):
            axb.plot([-3.5, -3.5], [s0 - 0.5, s1 - 0.5], color=PERSON_COLORS[p % 5], lw=3, clip_on=False, solid_capstyle="butt")
        if c == 1:
            axb.set_ylabel("cosine similarity,\nnodes ordered by person", fontsize=8, labelpad=3)
        if c == 3:
            cax = fig.add_axes([0.94, axb.get_position().y0, 0.012, axb.get_position().height])
            cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=8, length=2)

    stem = FIGDIR / f"embedding_space_{a.proj}"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    json.dump({"img_id": chosen["img_id"], "K": K, "n_nodes": int(len(gt)), "proj": a.proj, "projections": projs,
               "checkpoints": [{"label": m[0], "path": p[1], "coco_val_pga": m[4], "run": m[5], "scene_pga": chosen["pga"][i],
                                "misassigned": int((cluster_to_person(chosen["labels"][i], gt) != gt).sum())}
                               for i, (m, p) in enumerate(zip(models, CHECKPOINTS))]},
              open(stem.with_suffix(".json"), "w"), indent=2)
    print(f"wrote {stem}.pdf/.png/.json")


if __name__ == "__main__":
    main()
