"""
Train SCOT head on frozen SA-GAT embeddings.

Loads a trained SA-GAT checkpoint, freezes the GAT weights, and trains
only the SCOT head to assign keypoints to people using the fixed embeddings.
This prevents the SCOT loss from corrupting the contrastive embeddings.

Usage:
    python train_scot_frozen.py \
        --checkpoint outputs/finetune_sa_gat_coco/latest/best.pt \
        --coco_train_dir data/coco2017/train2017 \
        --coco_train_ann data/coco2017/annotations/person_keypoints_train2017.json \
        --name scot_frozen_coco \
        --epochs 20
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from pathlib import Path

import torch
import torch.optim as optim

from config import ExperimentConfig, SCOTConfig
from gat import GATEmbedding
from preprocessor import PosePreprocessor
from dataset import PoseDataset
from dataloader import create_dataloader
from virtual_adapter import VirtualAdapter
from losses import SCOTLoss
from ot_head import SCOTHead
from evaluator import compute_pga, predict_scot, predict_knn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="SA-GAT checkpoint to freeze")
    parser.add_argument("--virtual_dir", type=Path, default=Path("data/virtual"))
    parser.add_argument("--coco_train_dir", type=Path, default=None,
                        help="COCO train images (if set, trains on COCO)")
    parser.add_argument("--coco_train_ann", type=Path, default=None,
                        help="COCO train annotations")
    parser.add_argument("--name", type=str, default="scot_frozen",
                        help="Run name for output directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k_max", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--sinkhorn_iters", type=int, default=10)
    parser.add_argument("--sinkhorn_tau", type=float, default=0.1)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device

    # GUID-based run directory
    run_id = uuid.uuid4().hex[:8]
    save_dir = Path("outputs") / args.name / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # Update latest symlink
    latest_link = Path("outputs") / args.name / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_id)

    # ── Load frozen GAT ───────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ExperimentConfig(**ckpt["config"])

    if cfg.sa_gat is not None:
        from sa_gat import SAGATEmbedding
        gat = SAGATEmbedding(cfg.sa_gat).to(device)
        embedding_dim = cfg.sa_gat.output_dim
        use_depth = cfg.sa_gat.use_depth
    else:
        gat = GATEmbedding(cfg.gat).to(device)
        embedding_dim = cfg.gat.output_dim
        use_depth = cfg.gat.use_depth

    gat.load_state_dict(ckpt["gat_state"])
    gat.eval()

    # Freeze all GAT parameters
    for param in gat.parameters():
        param.requires_grad = False

    print(f"Loaded and froze GAT from {args.checkpoint}")
    print(f"Embedding dim: {embedding_dim}")

    # ── Data ──────────────────────────────────────────────────────────
    k_neighbors = 16 if embedding_dim >= 256 else 8
    preprocessor = PosePreprocessor(
        device=device, k_neighbors=k_neighbors, use_depth=use_depth,
    )

    if args.coco_train_dir is not None and args.coco_train_ann is not None:
        from coco_adapter import CocoAdapter
        train_adapter = CocoAdapter(
            img_dir=args.coco_train_dir,
            ann_file=args.coco_train_ann,
            use_depth=use_depth,
        )
        train_loader = create_dataloader(
            PoseDataset(train_adapter),
            batch_size=4, shuffle=True, num_workers=4,
        )
    else:
        train_loader = create_dataloader(
            PoseDataset(VirtualAdapter(args.virtual_dir / "train")),
            batch_size=4, shuffle=True, num_workers=4,
        )

    val_loader = create_dataloader(
        PoseDataset(VirtualAdapter(args.virtual_dir / "val")),
        batch_size=4, shuffle=False, num_workers=0,
    )

    # ── SCOT head ─────────────────────────────────────────────────────
    scot_cfg = SCOTConfig(
        hidden_dim=args.hidden_dim,
        k_max=args.k_max,
        sinkhorn_iters=args.sinkhorn_iters,
        sinkhorn_tau=args.sinkhorn_tau,
    )
    scot_head = SCOTHead(scot_cfg, embedding_dim=embedding_dim).to(device)
    scot_loss_fn = SCOTLoss(cfg.loss)

    optimizer = optim.Adam(scot_head.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    print(f"\nTraining SCOT head (frozen GAT)")
    print(f"Run ID:   {run_id}")
    print(f"k_max:    {args.k_max}")
    print(f"Epochs:   {args.epochs}")
    print(f"Save dir: {save_dir}\n")

    # ── Training ──────────────────────────────────────────────────────
    best_val_pga = 0.0

    for epoch in range(1, args.epochs + 1):
        scot_head.train()
        epoch_loss = 0.0
        n_graphs = 0
        t0 = time.time()

        for batch in train_loader:
            graphs = preprocessor.process_batch(batch)
            for graph in graphs:
                # Skip scenes that exceed k_max
                if graph.num_people > args.k_max:
                    continue

                graph = graph.to(device)
                optimizer.zero_grad()

                with torch.no_grad():
                    embeddings = gat(graph)

                k = int(graph.num_people)
                logits, T = scot_head(embeddings, k, graph.joint_types)

                loss_out = scot_loss_fn(logits, graph.person_labels)
                loss = loss_out["total_loss"]

                loss.backward()
                torch.nn.utils.clip_grad_norm_(scot_head.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_graphs += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_graphs, 1)
        elapsed = time.time() - t0

        # Validation
        if epoch % 2 == 0 or epoch == args.epochs:
            scot_head.eval()
            all_pga = []

            with torch.no_grad():
                for batch in val_loader:
                    graphs = preprocessor.process_batch(batch)
                    for graph in graphs:
                        graph = graph.to(device)
                        embeddings = gat(graph)
                        k = int(graph.num_people)

                        pred = predict_scot(scot_head, embeddings, k, graph.joint_types)
                        pga = compute_pga(pred, graph.person_labels)
                        all_pga.append(pga)

            val_pga = sum(all_pga) / max(len(all_pga), 1)

            log = (f"Epoch {epoch:3d}/{args.epochs} | "
                   f"loss {avg_loss:.4f} | "
                   f"val_pga {val_pga:.4f} | "
                   f"{elapsed:.1f}s")

            if val_pga > best_val_pga:
                best_val_pga = val_pga
                # Inject SCOT config so evaluator can load the head
                save_cfg = cfg.model_dump()
                save_cfg["scot"] = scot_cfg.model_dump()
                torch.save({
                    "epoch": epoch,
                    "val_pga": val_pga,
                    "gat_state": ckpt["gat_state"],  # original frozen weights
                    "head_state": scot_head.state_dict(),
                    "config": save_cfg,
                }, save_dir / "best.pt")
                log += "  ← best"

            print(log)
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"loss {avg_loss:.4f} | {elapsed:.1f}s")

    # Save final
    save_cfg = cfg.model_dump()
    save_cfg["scot"] = scot_cfg.model_dump()
    torch.save({
        "epoch": args.epochs,
        "val_pga": best_val_pga,
        "gat_state": ckpt["gat_state"],
        "head_state": scot_head.state_dict(),
        "config": save_cfg,
    }, save_dir / "final.pt")

    print(f"\nTraining complete. Best val PGA: {best_val_pga:.4f}")
    print(f"Saved to {save_dir}")


if __name__ == "__main__":
    main()
