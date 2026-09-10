"""Figure: the detector swap, qualitatively - one scene, three detectors, one frozen grouping module.

Rows: HigherHRNet-W32-512 (cached output of the AP run), YOLO11x-pose (ultralytics), RTMO-L (rtmlib,
body7 weights) - the same three detectors and the same 0.1 pooling threshold as the dissertation's
detector-swap table. Columns: (a) the pooled keypoints with the detector's identities discarded (what
crosses the interface), (b) the detector's native grouping, (c) the frozen PG-GAT + K-head + k-means
grouping of the pooled keypoints. As in the dissertation's Fig. 4.4 only the detections matched to
ground truth (10 px per-joint-type Hungarian) are drawn, coloured by the ground-truth person their
cluster maps to, so a joint in another person's colour is a grouping error; each panel states its PGA.

Run from code_v3/:
    python ../paper_cviu/figures/src/fig_detector_swap.py [--img_id 281759]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path.cwd()))
from config import ExperimentConfig                                   # noqa: E402
from preprocessor import PosePreprocessor                              # noqa: E402
from evaluator import compute_pga, predict_knn                         # noqa: E402
from eval_end_to_end import match_detections_to_gt, build_graph_from_detections  # noqa: E402
from eval_e2e_generic import YoloPoseDetector, RtmoDetector, _pool_person_keypoints  # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent
COCO_IMG = Path("data/coco2017/val2017")
COCO_ANN = Path("data/coco2017/annotations/person_keypoints_val2017.json")
HRNET_CACHE = Path("outputs/e2e_ap/hrnet_cache")
CKPT = "outputs/pg_gat_finetune_sweep/3362f752/best.pt"
KHEAD = "outputs/k_head_meng_headline_coco/a636b1c4/best.pt"
CONF = 0.1
PERSON_COLORS = ["#c23b3b", "#2b5fa3", "#3c8a4e", "#b58a2a", "#7a4fa3", "#3a9fb0", "#d0699a", "#7d7d3a"]
COCO_SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
                 (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix", "font.size": 8, "figure.dpi": 150,
})


def load_models(device):
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ExperimentConfig(**ckpt["config"])
    from sa_gat import SAGATEmbedding
    gat = SAGATEmbedding(cfg.sa_gat).to(device); gat.load_state_dict(ckpt["gat_state"]); gat.eval()
    dim = cfg.sa_gat.output_dim
    pre = PosePreprocessor(device=device, k_neighbors=16 if dim >= 256 else 8, use_depth=False)
    from k_head import KEstimationHead
    kh = torch.load(KHEAD, map_location=device, weights_only=False)
    k_head = KEstimationHead(embedding_dim=dim).to(device); k_head.load_state_dict(kh["k_head_state"]); k_head.eval()
    return gat, pre, k_head


def hrnet_pool(img_id):
    joints = np.load(HRNET_CACHE / f"{img_id}.npy")           # [n, 17, 3] (y, x, conf)
    if len(joints) == 0:
        return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, int)
    xy = joints[:, :, [1, 0]]; conf = joints[:, :, 2]
    return _pool_person_keypoints(xy, conf, CONF)


def cluster_to_person(labels, gt):
    k, p = labels.max() + 1, gt.max() + 1
    conf = np.zeros((k, p), int)
    for l, g in zip(labels, gt):
        conf[l, g] += 1
    r, c = linear_sum_assignment(-conf)
    m = {int(x): int(y) for x, y in zip(r, c)}
    return np.array([m.get(int(l), p + int(l)) for l in labels])


def draw(ax, img, pos, types, labels, gt, crop, title, grey=False):
    ax.imshow(img)
    if grey:
        ax.plot(pos[:, 0], pos[:, 1], "o", color="0.15", ms=2.8, mec="white", mew=0.5, zorder=3)
    else:
        mapped = cluster_to_person(labels, gt)
        for l in np.unique(labels):
            m = labels == l
            col = PERSON_COLORS[int(mapped[m][0]) % len(PERSON_COLORS)]
            lookup = {int(t): p for t, p in zip(types[m], pos[m])}
            for u, v in COCO_SKELETON:
                if u in lookup and v in lookup:
                    ax.plot([lookup[u][0], lookup[v][0]], [lookup[u][1], lookup[v][1]], "-", color=col, lw=1.2, alpha=0.9, zorder=2)
            ax.plot(pos[m, 0], pos[m, 1], "o", color=col, ms=2.8, mec="white", mew=0.5, zorder=3)
    x0, y0, x1, y1 = crop
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    ax.set_title(title, fontsize=8, pad=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_id", type=int, default=281759)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    from pycocotools.coco import COCO
    coco = COCO(str(COCO_ANN))
    info = coco.loadImgs(a.img_id)[0]
    image = cv2.imread(str(COCO_IMG / info["file_name"]))
    img_rgb = image[:, :, ::-1]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=a.img_id, catIds=coco.getCatIds(catNms=["person"]), iscrowd=False))
    gt_list = []
    for ann in anns:
        k = np.array(ann["keypoints"], np.float32).reshape(17, 3)
        if (k[:, 2] > 0).sum() >= 3:
            k4 = np.zeros((17, 4), np.float32); k4[:, 0], k4[:, 1], k4[:, 3] = k[:, 0], k[:, 1], k[:, 2]
            gt_list.append(k4)
    K_gt = len(gt_list)

    gat, pre, k_head = load_models(a.device)
    detectors = [
        ("HigherHRNet-W32-512\n(heatmaps + AE)", lambda: hrnet_pool(a.img_id)),
        ("YOLO11x-pose\n(single-stage regression)", lambda: YoloPoseDetector(None, a.device)(image, CONF)),
        ("RTMO-L\n(one-stage bottom-up)", lambda: RtmoDetector(None, "cuda" if a.device == "cuda" else "cpu")(image, CONF)),
    ]
    rows = []
    with torch.no_grad():
        for name, fn in detectors:
            pos, typ, native = fn()
            m_idx, m_person, _ = match_detections_to_gt(pos, typ, gt_list)
            gt = m_person
            g = build_graph_from_detections(pos, typ, pre, a.device).to(a.device)   # every pooled detection, as deployed
            emb = gat(g)
            K_pred = max(1, min(int(k_head.predict(emb)), len(pos)))
            labels_all = predict_knn(emb, K_pred).cpu().numpy()
            labels = labels_all[m_idx]
            pga_native = float(compute_pga(torch.tensor(native[m_idx]), torch.tensor(gt)))
            pga_pg = float(compute_pga(torch.tensor(labels), torch.tensor(gt)))
            rows.append({"name": name, "pos": pos[m_idx], "types": typ[m_idx], "native": native[m_idx], "labels": labels,
                         "gt": gt, "n_det": int(len(pos)), "n_matched": int(len(m_idx)), "n_native": int(native.max() + 1) if len(native) else 0,
                         "K_pred": K_pred, "pga_native": pga_native, "pga_pg": pga_pg})
            print(f"{name.splitlines()[0]:<22s} pooled {len(pos):>4d} matched {len(m_idx):>4d} native instances {rows[-1]['n_native']:>2d} "
                  f"K_pred {K_pred} | PGA native {pga_native:.3f} PG-GAT {pga_pg:.3f}")

    allpos = np.concatenate([r["pos"] for r in rows])
    lo, hi = allpos.min(axis=0), allpos.max(axis=0); w, h = hi - lo
    crop = (max(0, lo[0] - 0.08 * w - 8), max(0, lo[1] - 0.08 * h - 8), min(info["width"], hi[0] + 0.08 * w + 8), min(info["height"], hi[1] + 0.08 * h + 8))
    aspect = (crop[3] - crop[1]) / (crop[2] - crop[0])
    panel_w = 2.25
    fig = plt.figure(figsize=(7.2, 3 * (panel_w * aspect + 0.28) + 0.1))
    gs = fig.add_gridspec(3, 3, left=0.05, right=0.995, top=0.97, bottom=0.01, wspace=0.03, hspace=0.18)
    for r, row in enumerate(rows):
        panels = [("(a) pooled keypoints, identities discarded" if r == 0 else "pooled keypoints", True, row["native"]),
                  (f"(b) native grouping, PGA {row['pga_native']:.2f}" if r == 0 else f"native grouping, PGA {row['pga_native']:.2f}", False, row["native"]),
                  (f"(c) PG-GAT, $\\hat{{K}}={row['K_pred']}$, PGA {row['pga_pg']:.2f}" if r == 0 else f"PG-GAT, $\\hat{{K}}={row['K_pred']}$, PGA {row['pga_pg']:.2f}", False, row["labels"])]
        for c, (title, grey, labels) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            draw(ax, img_rgb, row["pos"], row["types"], labels, row["gt"], crop, title, grey=grey)
            if c == 0:
                ax.text(-0.02, 0.5, row["name"], transform=ax.transAxes, rotation=90, fontsize=8, ha="right", va="center")
    stem = FIGDIR / "detector_swap"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    json.dump({"img_id": a.img_id, "K_gt": K_gt, "conf_threshold": CONF, "checkpoint": CKPT, "k_head": KHEAD,
               "rows": [{k: v for k, v in r.items() if k in ("name", "n_det", "n_matched", "n_native", "K_pred", "pga_native", "pga_pg")} for r in rows]},
              open(stem.with_suffix(".json"), "w"), indent=1)
    print(f"wrote {stem}.pdf/.png/.json (K_gt = {K_gt})")


if __name__ == "__main__":
    main()
