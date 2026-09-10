"""Figure: where the attention goes - per edge category and per person relation.

Captures GATv2 attention coefficients from every layer of (i) the standard GAT baseline and (ii) the
headline PG-GAT by wrapping each GATv2Conv's forward at runtime (return_attention_weights=True);
sa_gat.py / gat.py are not modified. Every edge of every scene is categorised exactly as PG-GAT does
(same-type / skeletal neighbour / same limb / cross-body, using the model's own skeleton tables) plus
the self-loop that GATv2Conv appends, and by person relation from the ground truth (same person /
other person). Attention is averaged over the heads of a group (standard heads, repulsion head) so the
mass per layer sums to the number of target nodes; shares are then compared with the uniform share
(the fraction of edges in that category), i.e. what uniform attention would give.

Panels: (a) attention share by edge category, baseline vs PG-GAT standard heads, layer-averaged, with
uniform shares marked; (b) share of non-self attention on same-person edges per layer; (c) the scene of
the embedding figure with each keypoint's two most-attended incoming edges (last PG-GAT layer, standard
heads), coloured by category, width proportional to attention.

Run from code_v3/:
    python ../paper_cviu/figures/src/fig_attention_edges.py [--img_id 554328] [--max_scenes N]
Writes paper_cviu/figures/attention_edges.pdf/.png and a JSON sidecar with every number plotted.
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

sys.path.insert(0, str(Path.cwd()))
from config import ExperimentConfig            # noqa: E402
from preprocessor import PosePreprocessor       # noqa: E402
from dataset import PoseDataset                 # noqa: E402
from coco_adapter import CocoAdapter            # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent
MODELS = [   # (label, path, W&B run)
    ("Standard GAT", "outputs/train_knn_no_depth/4ca258c4/best.pt", "jkd1ln4a"),
    ("PG-GAT (headline)", "outputs/pg_gat_finetune_sweep/3362f752/best.pt", "b5noyg7t"),
]
CATS = ["same type", "skeletal\nneighbour", "same limb", "cross-body", "self-loop"]
CAT_COLORS = ["#c23b3b", "#3c8a4e", "#2b5fa3", "0.55", "0.8"]
PERSON_COLORS = ["#c23b3b", "#2b5fa3", "#3c8a4e", "#b58a2a", "#7a4fa3"]
COCO_SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
                 (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def load_gat(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ExperimentConfig(**ckpt["config"])
    if cfg.sa_gat is not None:
        from sa_gat import SAGATEmbedding
        gat, dim, use_depth = SAGATEmbedding(cfg.sa_gat), cfg.sa_gat.output_dim, cfg.sa_gat.use_depth
    else:
        from gat import GATEmbedding
        gat, dim, use_depth = GATEmbedding(cfg.gat), cfg.gat.output_dim, cfg.gat.use_depth
    assert not use_depth
    gat = gat.to(device); gat.load_state_dict(ckpt["gat_state"]); gat.eval()
    return gat, PosePreprocessor(device=device, k_neighbors=16 if dim >= 256 else 8, use_depth=False)


def hook_convs(gat, store):
    """Wrap every GATv2Conv so its attention lands in `store` as (layer, group, edge_index, alpha)."""
    from torch_geometric.nn import GATv2Conv

    def wrap(conv, layer, group):
        orig = conv.forward

        def fwd(x, edge_index, edge_attr=None, **kw):
            out, (ei, alpha) = orig(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            store.append((layer, group, ei.detach(), alpha.detach()))
            return out
        conv.forward = fwd

    for li, layer in enumerate(gat.gat_layers):
        if isinstance(layer, GATv2Conv):                     # baseline GATEmbedding
            wrap(layer, li, "standard")
        else:                                                # SAGATLayer
            wrap(layer.gat_standard, li, "standard")
            if getattr(layer, "gat_repulsion", None) is not None:
                wrap(layer.gat_repulsion, li, "repulsion")


def categorise(ei, joint_types, person, skel_mat, limb_mat):
    src, dst = ei
    ts, td = joint_types[src], joint_types[dst]
    cat = torch.full((ei.size(1),), 3, dtype=torch.long, device=ei.device)   # cross-body
    cat[limb_mat[ts, td]] = 2
    cat[skel_mat[ts, td]] = 1
    cat[ts == td] = 0
    cat[src == dst] = 4                                                        # self-loop
    same_person = person[src] == person[dst]
    return cat, same_person


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_img_dir", type=Path, default=Path("data/coco2017/val2017"))
    ap.add_argument("--coco_ann_file", type=Path, default=Path("data/coco2017/annotations/person_keypoints_val2017.json"))
    ap.add_argument("--img_id", type=str, default="554328", help="scene for panel (c)")
    ap.add_argument("--max_scenes", type=int, default=None)
    ap.add_argument("--redraw", action="store_true", help="skip the computation; redraw from the JSON + npz sidecars")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    stem = FIGDIR / "attention_edges"
    if a.redraw:
        out = json.load(open(stem.with_suffix(".json")))
        z = np.load(str(stem) + "_scene.npz")
        scene_c = {k: z[k] for k in z.files}
        scene_c["image"] = torch.from_numpy(scene_c["image"]); scene_c["kps"] = None
        loaded = [(lab, None, None, run) for lab, _, run in MODELS]
        n_scenes = out["n_scenes"]
    else:
        out, scene_c, loaded, n_scenes = compute(a)
    draw(a, out, scene_c, loaded, stem)


def compute(a):
    loaded = [(lab, *load_gat(p, a.device), run) for lab, p, run in MODELS]
    pg = loaded[1][1]
    skel_mat, limb_mat = pg.skel_neighbor_mat.to(a.device), pg.same_limb_mat.to(a.device)
    stores = {lab: [] for lab, *_ in loaded}
    for lab, gat, pre, run in loaded:
        hook_convs(gat, stores[lab])

    adapter = CocoAdapter(img_dir=a.coco_img_dir, ann_file=a.coco_ann_file, min_people=2, device="cpu", use_depth=False)
    ds = PoseDataset(adapter)
    n_scenes = len(ds) if a.max_scenes is None else min(a.max_scenes, len(ds))

    # accumulators: [model][layer][group] -> {"mass": [5], "count": [5], "mass_same": float, "mass_other": float, "n_same", "n_other"}
    acc = {lab: {} for lab, *_ in loaded}
    scene_c = None
    with torch.no_grad():
        for i in range(n_scenes):
            s = ds[i]
            kps = s["keypoints"]
            for lab, gat, pre, run in loaded:
                g = pre.create_graph(kps)
                if g is None:
                    continue
                g = g.to(a.device)
                stores[lab].clear()
                gat(g)
                for layer, group, ei, alpha in stores[lab]:
                    cat, same = categorise(ei, g.joint_types, g.person_labels, skel_mat, limb_mat)
                    am = alpha.mean(dim=1)                                    # head-averaged, sums to N per layer
                    d = acc[lab].setdefault(layer, {}).setdefault(group, {
                        "mass": np.zeros(5), "count": np.zeros(5), "mass_same": 0.0, "mass_other": 0.0,
                        "n_same": 0, "n_other": 0})
                    for c in range(5):
                        m = cat == c
                        d["mass"][c] += float(am[m].sum()); d["count"][c] += int(m.sum())
                    ns = cat != 4
                    d["mass_same"] += float(am[ns & same].sum()); d["mass_other"] += float(am[ns & ~same].sum())
                    d["n_same"] += int((ns & same).sum()); d["n_other"] += int((ns & ~same).sum())
                    if lab == loaded[1][0] and str(s["img_id"]) == a.img_id and group == "standard" \
                            and layer == len(pg.gat_layers) - 1:
                        scene_c = {"image": s["image"], "kps": kps, "ei": ei.cpu().numpy(), "alpha": am.cpu().numpy(),
                                   "cat": cat.cpu().numpy(), "person": g.person_labels.cpu().numpy(),
                                   "pos": (g.x[:, :2] * pre.image_size).cpu().numpy()}
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{n_scenes}", flush=True)

    # ── numbers ──────────────────────────────────────────────────────────
    out = {"n_scenes": n_scenes, "models": {}}
    for lab, gat, pre, run in loaded:
        out["models"][lab] = {"run": run, "k_neighbors": pre.k_neighbors, "layers": {}}
        for layer in sorted(acc[lab]):
            out["models"][lab]["layers"][layer] = {}
            for group, d in acc[lab][layer].items():
                tot = d["mass"].sum(); ntot = d["count"].sum()
                out["models"][lab]["layers"][layer][group] = {
                    "attention_share": (d["mass"] / tot).tolist(), "edge_share": (d["count"] / ntot).tolist(),
                    "same_person_attention_share": d["mass_same"] / (d["mass_same"] + d["mass_other"]),
                    "same_person_edge_share": d["n_same"] / (d["n_same"] + d["n_other"]),
                }
    stem = FIGDIR / "attention_edges"
    json.dump(out, open(stem.with_suffix(".json"), "w"), indent=1)
    if scene_c is not None:
        np.savez_compressed(str(stem) + "_scene.npz", image=scene_c["image"].numpy(), ei=scene_c["ei"], alpha=scene_c["alpha"],
                            cat=scene_c["cat"], person=scene_c["person"], pos=scene_c["pos"])
    return out, scene_c, loaded, n_scenes


def draw(a, out, scene_c, loaded, stem):
    for lab in out["models"]:
        out["models"][lab]["layers"] = {int(k): v for k, v in out["models"][lab]["layers"].items()}

    def layer_avg(lab, group, key):
        L = [v[group][key] for v in out["models"][lab]["layers"].values() if group in v]
        return np.mean(np.array(L), axis=0)

    for lab in out["models"]:
        for group in ("standard", "repulsion"):
            if any(group in v for v in out["models"][lab]["layers"].values()):
                sh, es = layer_avg(lab, group, "attention_share"), layer_avg(lab, group, "edge_share")
                print(f"{lab:<18s} {group:<10s} attention share " + " ".join(f"{x:.3f}" for x in sh)
                      + " | edge share " + " ".join(f"{x:.3f}" for x in es)
                      + f" | same-person {layer_avg(lab, group, 'same_person_attention_share'):.3f} vs uniform {layer_avg(lab, group, 'same_person_edge_share'):.3f}")

    # ── figure ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.8, 1.35], left=0.06, right=0.99, top=0.88, bottom=0.2, wspace=0.38)
    base, pgl = loaded[0][0], loaded[1][0]

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(5); w = 0.36
    sb, sp = layer_avg(base, "standard", "attention_share"), layer_avg(pgl, "standard", "attention_share")
    ub, up = layer_avg(base, "standard", "edge_share"), layer_avg(pgl, "standard", "edge_share")
    ax.bar(x - w / 2, sb, w, color="0.75", label=f"{base} ($k$=8)")
    ax.bar(x + w / 2, sp, w, color="#c23b3b", label=f"PG-GAT standard heads ($k$=16)")
    ax.plot(x - w / 2, ub, "_", color="black", ms=9, mew=1.2, label="uniform attention (edge share)")
    ax.plot(x + w / 2, up, "_", color="black", ms=9, mew=1.2)
    ax.set_ylim(0, 0.86)
    ax.set_xticks(x); ax.set_xticklabels(["same\ntype", "skeletal\nneighbour", "same\nlimb", "cross-\nbody", "self-\nloop"], fontsize=8)
    ax.set_ylabel("share of attention mass", fontsize=8)
    ax.set_title("(a) attention by edge category (layer average)", fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.2, borderaxespad=0.2)

    ax = fig.add_subplot(gs[0, 1])
    for lab, col, mk, short, kk in ((base, "0.45", "s", "Standard GAT", 8), (pgl, "#c23b3b", "o", "PG-GAT", 16)):
        L = out["models"][lab]["layers"]
        ys = [L[l]["standard"]["same_person_attention_share"] for l in sorted(L)]
        us = [L[l]["standard"]["same_person_edge_share"] for l in sorted(L)]
        xs = np.arange(1, len(ys) + 1)
        ax.plot(xs, ys, "-" + mk, color=col, ms=4)
        ax.plot(xs, us, "--", color=col, lw=0.8)
        ax.text(xs[-1] + 0.1, ys[-1], short, fontsize=8, color=col, va="center")
        ax.text(xs[-1] + 0.1, us[-1], f"uniform, $k$={kk}", fontsize=8, color=col, va="center")
    ax.set_xlabel("layer", fontsize=8); ax.set_xticks([1, 2, 3]); ax.set_xlim(0.8, 4.3)
    ax.set_ylabel("share of non-self attention\non same-person edges", fontsize=8)
    ax.set_title("(b) attention within the person", fontsize=8)
    ax.set_ylim(0.5, 0.97)

    ax = fig.add_subplot(gs[0, 2])
    if scene_c is not None:
        img = scene_c["image"].permute(1, 2, 0).numpy(); ax.imshow(img)
        pos, ei, am, cat, person = scene_c["pos"], scene_c["ei"], scene_c["alpha"], scene_c["cat"], scene_c["person"]
        N = len(pos)
        for n in range(N):
            inc = np.where((ei[1] == n) & (cat != 4))[0]
            for e in inc[np.argsort(-am[inc])[:2]]:
                if am[e] < 0.03:
                    continue
                sidx = ei[0, e]
                ax.plot([pos[sidx, 0], pos[n, 0]], [pos[sidx, 1], pos[n, 1]], "-", color=CAT_COLORS[cat[e]],
                        lw=0.4 + 4.0 * am[e], alpha=0.85, solid_capstyle="round")
        for p in range(person.max() + 1):
            m = person == p
            ax.plot(pos[m, 0], pos[m, 1], "o", color=PERSON_COLORS[p % 5], ms=2.8, mec="white", mew=0.5, zorder=5)
        pad = 0.12 * (pos.max(axis=0) - pos.min(axis=0)).max()
        x0, y0 = np.maximum(pos.min(axis=0) - pad, 0); x1, y1 = pos.max(axis=0) + pad
        ax.set_xlim(x0, min(x1, img.shape[1])); ax.set_ylim(min(y1, img.shape[0]), y0)
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=CAT_COLORS[c], lw=2, label=CATS[c].replace("\n", " ")) for c in range(4)],
                  frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2,
                  handlelength=1.2, columnspacing=1.2, handletextpad=0.5)
        ax.set_title("(c) top-2 attended edges per keypoint, last layer", fontsize=8)
    ax.axis("off")

    fig.savefig(stem.with_suffix(".pdf")); fig.savefig(stem.with_suffix(".png"), dpi=200)
    print(f"wrote {stem}.pdf/.png/.json ({n_scenes_of(out)} scenes)")


def n_scenes_of(out):
    return out["n_scenes"]


if __name__ == "__main__":
    main()
