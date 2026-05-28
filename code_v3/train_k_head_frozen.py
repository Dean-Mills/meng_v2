"""
Train K estimation head on frozen SA-GAT embeddings.

Loads a trained SA-GAT checkpoint, freezes the GAT weights, and trains
only the K head MLP to predict person count from the embeddings.
This prevents the K estimation loss from corrupting the contrastive
embeddings.

Usage:
    python train_k_head_frozen.py \
        --checkpoint outputs/checkpoints/sa_gat_full/best.pt \
        --save_dir outputs/checkpoints/k_head_frozen \
        --epochs 50
"""
from __future__ import annotations

import argparse
import shutil
import time
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from config import ExperimentConfig
from gat import GATEmbedding
from preprocessor import PosePreprocessor
from dataset import PoseDataset
from dataloader import create_dataloader
from virtual_adapter import VirtualAdapter
from k_head import KEstimationHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="SA-GAT or GAT checkpoint to freeze")
    parser.add_argument("--virtual_dir", type=Path, default=Path("data/virtual"))
    parser.add_argument("--coco_train_dir", type=Path, default=None,
                        help="COCO train images (if set, trains on COCO)")
    parser.add_argument("--coco_train_ann", type=Path, default=None,
                        help="COCO train annotations")
    parser.add_argument("--name", type=str, default="k_head_frozen",
                        help="Run name for output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project. If unset, wandb logging is disabled.")
    parser.add_argument("--wandb-entity", type=str, default="deanmills")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="",
                        help="Comma-separated list of tags.")
    parser.add_argument("--wandb-notes", type=str, default=None)
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

    # Validation always on virtual for consistent benchmarking
    val_loader = create_dataloader(
        PoseDataset(VirtualAdapter(args.virtual_dir / "val")),
        batch_size=4, shuffle=False, num_workers=0,
    )

    # ── K head ────────────────────────────────────────────────────────
    k_head = KEstimationHead(embedding_dim=embedding_dim).to(device)

    optimizer = optim.Adam(k_head.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    print(f"\nTraining K head (frozen GAT)")
    print(f"Run ID:  {run_id}")
    print(f"Epochs:  {args.epochs}")
    print(f"Save dir: {save_dir}\n")

    # ── W&B (optional) ────────────────────────────────────────────────
    wandb_run = None
    if args.wandb_project:
        import wandb
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_id,
            group=args.wandb_group,
            tags=tags,
            notes=args.wandb_notes,
            config={
                "run_id":        run_id,
                "checkpoint":    str(args.checkpoint),
                "epochs":        args.epochs,
                "lr":            args.lr,
                "embedding_dim": embedding_dim,
                "training_data": "coco" if args.coco_train_dir else "virtual",
                "config":        ckpt["config"],
            },
            save_code=True,
        )
        print(f"W&B:     {wandb_run.url}\n")

    # ── Training ──────────────────────────────────────────────────────
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        k_head.train()
        epoch_loss = 0.0
        n_graphs = 0
        t0 = time.time()

        for batch in train_loader:
            graphs = preprocessor.process_batch(batch)
            for graph in graphs:
                graph = graph.to(device)
                optimizer.zero_grad()

                with torch.no_grad():
                    embeddings = gat(graph)

                k_pred = k_head(embeddings)
                k_gt = torch.tensor(float(graph.num_people), device=device)
                loss = F.l1_loss(k_pred, k_gt)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_graphs += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_graphs, 1)
        elapsed = time.time() - t0

        # Validation
        if epoch % 5 == 0 or epoch == args.epochs:
            k_head.eval()
            n_correct = 0
            n_off_by_one = 0
            n_val = 0

            with torch.no_grad():
                for batch in val_loader:
                    graphs = preprocessor.process_batch(batch)
                    for graph in graphs:
                        graph = graph.to(device)
                        embeddings = gat(graph)
                        k_pred = k_head.predict(embeddings)
                        k_gt = int(graph.num_people)

                        n_val += 1
                        if k_pred == k_gt:
                            n_correct += 1
                        if abs(k_pred - k_gt) <= 1:
                            n_off_by_one += 1

            exact_acc = n_correct / max(n_val, 1)
            off1_acc = n_off_by_one / max(n_val, 1)

            log = (f"Epoch {epoch:3d}/{args.epochs} | "
                   f"loss {avg_loss:.4f} | "
                   f"val exact {exact_acc:.3f} off1 {off1_acc:.3f} | "
                   f"{elapsed:.1f}s")

            if exact_acc > best_val_acc:
                best_val_acc = exact_acc
                # Save: GAT state (frozen) + K head state
                torch.save({
                    "epoch": epoch,
                    "val_k_accuracy": exact_acc,
                    "gat_state": ckpt["gat_state"],  # original frozen weights
                    "k_head_state": k_head.state_dict(),
                    "head_state": ckpt.get("head_state"),  # preserve any head
                    "config": ckpt["config"],
                }, save_dir / "best.pt")
                log += "  ← best"

                if wandb_run is not None:
                    import wandb
                    artifact = wandb.Artifact(
                        name=args.name, type="model",
                        metadata={"epoch": epoch, "val_k_accuracy": exact_acc},
                    )
                    artifact.add_file(str(save_dir / "best.pt"))
                    wandb_run.log_artifact(artifact, aliases=["latest"])

            print(log)

            if wandb_run is not None:
                wandb_run.log({
                    "epoch":          epoch,
                    "train/loss":     avg_loss,
                    "val/exact":      exact_acc,
                    "val/off_by_one": off1_acc,
                    "val/best_exact": best_val_acc,
                }, step=epoch)
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"loss {avg_loss:.4f} | {elapsed:.1f}s")
            if wandb_run is not None:
                wandb_run.log({
                    "epoch":      epoch,
                    "train/loss": avg_loss,
                }, step=epoch)

    # Save final
    torch.save({
        "epoch": args.epochs,
        "val_k_accuracy": best_val_acc,
        "gat_state": ckpt["gat_state"],
        "k_head_state": k_head.state_dict(),
        "head_state": ckpt.get("head_state"),
        "config": ckpt["config"],
    }, save_dir / "final.pt")

    print(f"\nTraining complete. Best val K accuracy: {best_val_acc:.3f}")
    print(f"Saved to {save_dir}")

    if wandb_run is not None:
        wandb_run.summary["best_val_k_exact"] = best_val_acc
        wandb_run.summary["total_epochs"]     = args.epochs
        wandb_run.summary["save_dir"]         = str(save_dir)
        wandb_run.finish()


if __name__ == "__main__":
    main()
