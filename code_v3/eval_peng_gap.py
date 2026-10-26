"""
PEng end-to-end gap table on one fixed scene set (no detector re-run).

Re-scores the HigherHRNet scenes cached by eval_e2e_per_image.py
(outputs/e2e_per_image/arrays/<image_id>.npz: detections, AE ids, 10 px
matches, GT) so that every row of the table -- HigherHRNet AE and every
(K selector x grouping method) cell -- is computed on the identical scene
set. Closes the 2026-08-07 open item ("AE not recomputed on the clamp run's
matched subset") and doubles as the frozen E2E benchmark for new PEng
backbones: pass a different --checkpoint and the scenes stay fixed.

K selectors:
    oracle  K = annotated people (>= 3 labelled kps)      (MEng protocol)
    eff     K = GT people with >= 1 matched detection     (diagnostic: true K of the graph)
    pred    K = K-head                                    (MEng autonomous)
    lb      K = K_lb = max per-type detection count       (bound only, no K-head)
    clamp   K = max(pred, K_lb)                           (PEng feasibility clamp)
    kdet    K = AE instances among matched detections     (diagnostic: detector's count)
Grouping methods: kNN (k-means), COP-Kmeans, THA. Every K is clipped to [1, n].

Usage (from code_v3/):
    python eval_peng_gap.py \
        --checkpoint outputs/pg_gat_finetune_sweep/3362f752/best.pt \
        --k_head outputs/k_head_meng_headline_coco/a636b1c4/best.pt \
        --arrays outputs/e2e_per_image/arrays \
        --out_dir outputs/peng_gap/3362f752

--letterbox maps detections into the frame PG-GAT is trained in: dataset.py
resizes every training image to fit 512 x 512 and pads it (keypoints scaled
and offset to match), while build_graph_from_detections divides raw pixel
coordinates by 512. Off by default so earlier numbers reproduce.

--k_head is optional: without it the pred and clamp rows are skipped and the
autonomous number is the K_lb row (it ties the clamp on meng-headline).

Pure addition: imports the existing graph builder and grouping functions
unchanged; no W&B.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_end_to_end import build_graph_from_detections
from config import ExperimentConfig
from eval_k_selection import k_lower_bound
from preprocessor import PosePreprocessor
from evaluator import compute_pga, predict_knn
from eval_cop_kmeans import predict_cop_kmeans
from eval_hungarian_grouping import predict_tha

SELECTORS = ["oracle", "eff", "pred", "lb", "clamp", "kdet"]
METHODS = {
    "knn": lambda emb, k, jt: predict_knn(emb, k),
    "copkm": lambda emb, k, jt: predict_cop_kmeans(emb, k, jt),
    "tha": lambda emb, k, jt: predict_tha(emb, k, jt),
}


def load_models(checkpoint: Path, k_head_path, device: str):
    """Euclidean backbones only (PG-GAT, PG-GAT-H); K-head optional."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ExperimentConfig(**ckpt["config"])
    if cfg.pg_gat_hier is not None:
        from pg_gat_hier import PGGATHierEmbedding
        gat, mcfg = PGGATHierEmbedding(cfg.pg_gat_hier), cfg.pg_gat_hier
    else:
        assert cfg.sa_gat is not None and cfg.sa_gat_hyperbolic is None, "Euclidean PG-GAT checkpoints only"
        from sa_gat import SAGATEmbedding
        gat, mcfg = SAGATEmbedding(cfg.sa_gat), cfg.sa_gat
    gat.load_state_dict(ckpt["gat_state"])
    gat = gat.to(device).eval()
    pre = PosePreprocessor(device=device, k_neighbors=16 if mcfg.output_dim >= 256 else 8,
                           use_depth=mcfg.use_depth)
    k_head = None
    if k_head_path is not None:
        from k_head import KEstimationHead
        k_head = KEstimationHead(embedding_dim=mcfg.output_dim).to(device)
        k_head.load_state_dict(torch.load(k_head_path, map_location=device, weights_only=False)["k_head_state"])
        k_head.eval()
    print(f"Loaded {type(gat).__name__} (epoch {ckpt.get('epoch', '?')})"
          + (" + K-head" if k_head is not None else ", no K-head"))
    return gat, k_head, pre


def letterbox(pos: np.ndarray, width: int, height: int, target: int) -> np.ndarray:
    """dataset.PoseDataset._transform applied to (x, y) pixel coordinates."""
    scale = target / max(height, width)
    pad_left = (target - int(width * scale)) // 2
    pad_top = (target - int(height * scale)) // 2
    return pos * scale + np.array([pad_left, pad_top], dtype=pos.dtype)


def score_scene(emb, jt, gt_labels, ks: dict) -> dict:
    """Group once per distinct K, then read each selector's row."""
    n = emb.size(0)
    ks = {s: int(min(max(k, 1), n)) for s, k in ks.items()}
    per_k = {}
    for k in sorted(set(ks.values())):
        per_k[k] = {m: float(compute_pga(fn(emb, k, jt), gt_labels)) for m, fn in METHODS.items()}
    return {s: {"k": k, **per_k[k]} for s, k in ks.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--k_head", type=Path, default=None)
    ap.add_argument("--arrays", type=Path, default=Path("outputs/e2e_per_image/arrays"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed numpy / torch so COP-Kmeans restarts are repeatable (verification runs)")
    ap.add_argument("--letterbox", action="store_true",
                    help="map detections into the 512 letterbox frame used in training")
    ap.add_argument("--coco_img_dir", type=Path, default=Path("data/coco2017/val2017"),
                    help="images for backbones with image evidence (implies --letterbox)")
    a = ap.parse_args()
    if a.seed is not None:
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    gat, k_head, pre = load_models(a.checkpoint, a.k_head, a.device)
    needs_image = getattr(gat, "needs_image", False)
    if needs_image:
        from torchvision.io import read_image
        from image_evidence import letterbox_image
        a.letterbox = True
    selectors = [s for s in SELECTORS if k_head is not None or s not in ("pred", "clamp")]

    files = sorted(a.arrays.glob("*.npz"), key=lambda p: int(p.stem))
    if a.max_images:
        files = files[:a.max_images]
    print(f"{len(files)} cached scenes in {a.arrays}")
    sizes = {}
    if a.letterbox:
        with open(a.arrays.parent / "per_image.jsonl") as fm:
            for line in fm:
                r = json.loads(line)
                sizes[r["image_id"]] = (r["width"], r["height"])
        print("coordinates: letterboxed to the training frame")

    rows = []
    fh = open(a.out_dir / "per_image.jsonl", "w")
    with torch.no_grad():
        for i, f in enumerate(files):
            d = np.load(f)
            m_idx, m_person = d["matched_idx"], d["gt_person"]
            det_pos, det_types, ae_ids = d["det_pos"], d["det_types"], d["ae_ids"]
            pos = det_pos[m_idx]
            if a.letterbox:
                pos = letterbox(pos, *sizes[int(f.stem)], pre.image_size)
            graph = build_graph_from_detections(pos, det_types[m_idx], pre, a.device)
            if graph is None:
                continue
            if needs_image:
                img = read_image(str(a.coco_img_dir / str(d["file_name"])))
                img = img.repeat(3, 1, 1) if img.shape[0] == 1 else img[:3]
                graph.image = letterbox_image(img, pre.image_size)
            graph = graph.to(a.device)
            emb = gat(graph)
            jt = graph.joint_types
            gt_labels = torch.tensor(m_person, dtype=torch.long)

            k_lb = k_lower_bound(jt)
            ks = {
                "oracle": int(d["K_gt"]),
                "eff": len(np.unique(m_person)),
                "lb": k_lb,
                "kdet": len(np.unique(ae_ids[m_idx])),
            }
            if k_head is not None:
                k_pred = max(1, k_head.predict(emb))
                ks.update(pred=k_pred, clamp=max(k_pred, k_lb))
            rec = {
                "image_id": int(f.stem), "n": int(len(m_idx)), "K_gt": int(d["K_gt"]),
                "ae": float(compute_pga(torch.tensor(ae_ids[m_idx], dtype=torch.long), gt_labels)),
                **score_scene(emb, jt, gt_labels, ks),
            }
            rows.append(rec)
            fh.write(json.dumps(rec) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(files)}", flush=True)
    fh.close()

    n = len(rows)
    k_gt = np.array([r["K_gt"] for r in rows])
    k_eff = np.array([r["eff"]["k"] for r in rows])
    summary = {"n": n, "checkpoint": str(a.checkpoint), "k_head": str(a.k_head),
               "ae": float(np.mean([r["ae"] for r in rows])),
               "k_gt_mean": float(k_gt.mean()), "k_eff_mean": float(k_eff.mean())}
    for s in selectors:
        k = np.array([r[s]["k"] for r in rows])
        summary[s] = {
            "k_mean": float(k.mean()),
            "k_exact_vs_eff": float((k == k_eff).mean()),
            **{m: float(np.mean([r[s][m] for r in rows])) for m in METHODS},
        }
    json.dump(summary, open(a.out_dir / "summary.json", "w"), indent=2)

    print(f"\nn = {n}   native grouping (ae_ids; HigherHRNet AE on the default cache) = {summary['ae']:.4f}")
    print(f"{'selector':<8} {'K mean':>7} {'K=eff':>7} " + " ".join(f"{m:>8}" for m in METHODS)
          + "   gap(best)")
    for s in selectors:
        r = summary[s]
        best = max(r[m] for m in METHODS)
        print(f"{s:<8} {r['k_mean']:>7.2f} {r['k_exact_vs_eff']:>7.1%} "
              + " ".join(f"{r[m]:>8.4f}" for m in METHODS) + f"   {summary['ae'] - best:+.4f}")
    print(f"\nSaved {a.out_dir / 'summary.json'} and per_image.jsonl")


if __name__ == "__main__":
    main()
