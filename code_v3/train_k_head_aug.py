"""
K-head on frozen embeddings, trained on detection-realistic graphs (PEng).

After PG-GAT-H + augmentation (2026-10-20) the embedding share of the E2E gap is
~0.002 and K estimation is the largest share (K_lb THA 0.9776 vs true-K 0.9829).
The MEng K-head (train_k_head_frozen.py) was trained on complete COCO GT graphs and
selected on synthetic val; on detections it ties the bare bound K_lb. This variant:

  * trains on COCO train2017 graphs passed through augment.detection_augment
    (joint dropout p ~ U(0, drop_max), v = 1), with target = number of people
    still present in the augmented graph — what K means on detections;
  * selects on a fixed held-out slice of COCO train2017 (augmented with a fixed
    seed), never on val2017, whose images are the E2E test scenes;
  * keeps the MEng KEstimationHead architecture, so the comparison is like for like.

Works for any backbone eval_peng_gap.load_models can load (PG-GAT, PG-GAT-H incl.
image evidence). Output checkpoint has the train_k_head_frozen.py layout
(gat_state, k_head_state, config), so every K-aware eval script accepts it.

    python train_k_head_aug.py \
        --checkpoint outputs/finetune_pg_gat_hier_aug_coco/b59d602d/best.pt \
        --name k_head_aug_b59d602d
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from augment import detection_augment
from coco_adapter import CocoAdapter
from dataloader import create_dataloader
from dataset import PoseDataset
from eval_peng_gap import load_models
from k_head import KEstimationHead
from k_head_hist import KHistHead, k_head_output, predict_k


def graphs_of(batch, pre, gat, drop_max, generator=None):
    graphs = pre.process_batch(batch)
    if getattr(gat, "needs_image", False):
        from image_evidence import attach_images
        graphs = attach_images(graphs, batch)
    graphs = [detection_augment(g, pre, drop_max, True, generator) for g in graphs]
    return [g for g in graphs if g is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True, help="backbone to freeze")
    ap.add_argument("--name", type=str, required=True)
    ap.add_argument("--coco_train_dir", type=Path, default=Path("data/coco2017/train2017"))
    ap.add_argument("--coco_train_ann", type=Path,
                    default=Path("data/coco2017/annotations/person_keypoints_train2017.json"))
    ap.add_argument("--joint_drop_max", type=float, default=0.4)
    ap.add_argument("--n_val", type=int, default=2000, help="held-out COCO train images for selection")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--head", choices=["mlp", "hist"], default="mlp",
                    help="mlp = KEstimationHead (MEng); hist = KHistHead (+ per-type counts and K_lb)")
    ap.add_argument("--max_images", type=int, default=None, help="smoke test: cap train / held-out images")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    run_id = uuid.uuid4().hex[:8]
    save_dir = Path("outputs") / a.name / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    latest = Path("outputs") / a.name / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_id)

    gat, _, pre = load_models(a.checkpoint, None, a.device)
    for p in gat.parameters():
        p.requires_grad_(False)
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    emb_dim = (ckpt["config"].get("pg_gat_hier") or ckpt["config"]["sa_gat"])["output_dim"]

    full = PoseDataset(CocoAdapter(img_dir=a.coco_train_dir, ann_file=a.coco_train_ann, use_depth=False))
    perm = torch.randperm(len(full), generator=torch.Generator().manual_seed(0)).tolist()
    val_idx, train_idx = perm[:a.n_val], perm[a.n_val:]
    if a.max_images:
        val_idx, train_idx = val_idx[:a.max_images], train_idx[:a.max_images]
    train_loader = create_dataloader(Subset(full, train_idx), batch_size=4, shuffle=True, num_workers=4)
    val_loader = create_dataloader(Subset(full, val_idx), batch_size=4, shuffle=False, num_workers=4)

    k_head = (KHistHead if a.head == "hist" else KEstimationHead)(embedding_dim=emb_dim).to(a.device)
    opt = torch.optim.Adam(k_head.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=1e-5)
    print(f"K-head (aug, {type(k_head).__name__}) on frozen {type(gat).__name__} from {a.checkpoint}")
    print(f"Run ID {run_id}; train {len(train_idx)} / held-out {len(val_idx)} COCO train images; "
          f"joint drop U(0, {a.joint_drop_max}), v = 1; {a.epochs} epochs")

    history, best = [], -1.0
    for epoch in range(1, a.epochs + 1):
        k_head.train()
        tot, n, t0 = 0.0, 0, time.time()
        for batch in train_loader:
            for g in graphs_of(batch, pre, gat, a.joint_drop_max):
                g = g.to(a.device)
                with torch.no_grad():
                    emb = gat(g)
                loss = F.l1_loss(k_head_output(k_head, emb, g.joint_types),
                                 torch.tensor(float(g.num_people), device=a.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                n += 1
        sched.step()

        k_head.eval()
        gen = torch.Generator().manual_seed(1)   # identical augmented held-out set every epoch
        exact = off1 = nv = 0
        with torch.no_grad():
            for batch in val_loader:
                for g in graphs_of(batch, pre, gat, a.joint_drop_max, gen):
                    g = g.to(a.device)
                    kp, kg = predict_k(k_head, gat(g), g.joint_types), int(g.num_people)
                    exact += kp == kg
                    off1 += abs(kp - kg) <= 1
                    nv += 1
        rec = {"epoch": epoch, "loss": tot / max(n, 1), "val_exact": exact / max(nv, 1),
               "val_off1": off1 / max(nv, 1), "n_val_graphs": nv}
        history.append(rec)
        log = (f"Epoch {epoch:2d}/{a.epochs} | loss {rec['loss']:.4f} | held-out exact {rec['val_exact']:.3f} "
               f"off1 {rec['val_off1']:.3f} | {time.time() - t0:.0f}s")
        state = {"epoch": epoch, "val_k_accuracy": rec["val_exact"], "gat_state": ckpt["gat_state"],
                 "k_head_state": k_head.state_dict(), "head_state": ckpt.get("head_state"),
                 "config": ckpt["config"], "backbone_checkpoint": str(a.checkpoint),
                 "joint_drop_max": a.joint_drop_max, "k_head_type": a.head}
        if rec["val_exact"] > best:
            best = rec["val_exact"]
            torch.save(state, save_dir / "best.pt")
            log += "  ← best"
        print(log, flush=True)
    torch.save(state, save_dir / "final.pt")
    json.dump(history, open(save_dir / "history.json", "w"), indent=2)
    print(f"done: {save_dir}  best held-out exact {best:.3f}")


if __name__ == "__main__":
    main()
