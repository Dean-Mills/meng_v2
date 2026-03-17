"""
Joint trainer for GAT + grouping head.

Trains GAT and grouping head end-to-end on virtual data.
The GAT is trained with contrastive loss throughout.
The grouping head (slot attention or graph partitioning) adds its own loss on top.

Usage:
    python trainer.py --config configs/train_slot.yaml
    python trainer.py --config configs/train_partition.yaml

Expects data at:
    {virtual_dir}/train/
    {virtual_dir}/val/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.optim as optim

from config import ExperimentConfig
from gat import GATEmbedding
from losses import GATOnlyLoss, SlotAttentionLoss, GraphPartitioningLoss
from preprocessor import PosePreprocessor
from dataset import PoseDataset
from dataloader import create_dataloader
from virtual_adapter import VirtualAdapter


def _build_head(cfg: ExperimentConfig, embedding_dim: int):
    """Instantiate the grouping head from config."""
    if cfg.slot_attention is not None:
        from slot_attention import SlotAttention
        return SlotAttention(cfg.slot_attention, embedding_dim=embedding_dim)

    if cfg.graph_partitioning is not None:
        from graph_partitioning import EdgeClassifier
        return EdgeClassifier(cfg.graph_partitioning, embedding_dim=embedding_dim)

    return None


def _build_head_loss(cfg: ExperimentConfig):
    """Instantiate the head loss from config."""
    if cfg.slot_attention is not None:
        return SlotAttentionLoss(cfg.loss)
    if cfg.graph_partitioning is not None:
        return GraphPartitioningLoss(cfg.loss)
    return None


def _head_forward(head, embeddings, graph):
    """
    Run the grouping head forward pass.
    Returns (head_out, head_loss_inputs) where head_loss_inputs are the
    args to pass directly to the head loss.
    """
    if head is None:
        return None, None

    from slot_attention import SlotAttention
    from graph_partitioning import EdgeClassifier

    if isinstance(head, SlotAttention):
        k      = int(graph.num_people)
        logits, slots = head(embeddings, k)
        return (logits, slots), (logits, graph.person_labels)

    if isinstance(head, EdgeClassifier):
        logits, pairs = head(embeddings)
        return (logits, pairs), (logits, pairs, graph.person_labels)

    return None, None


def _make_loader(virtual_dir: Path, split: str, batch_size: int, num_workers: int):
    adapter = VirtualAdapter(virtual_dir / split)
    dataset = PoseDataset(adapter)
    return create_dataloader(dataset, batch_size=batch_size,
                             shuffle=(split == "train"),
                             num_workers=num_workers)


# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: ExperimentConfig, device: str):
    assert cfg.training is not None, "training config missing from yaml"
    tc = cfg.training

    save_dir = Path(tc.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    virtual_dir = Path(tc.virtual_dir)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader = _make_loader(virtual_dir, "train", tc.batch_size, tc.num_workers)
    val_loader   = _make_loader(virtual_dir, "val",   tc.batch_size, 0)

    k_neighbors  = 16 if cfg.gat.output_dim >= 256 else 8
    preprocessor = PosePreprocessor(device=device, k_neighbors=k_neighbors)

    # ── Models ────────────────────────────────────────────────────────────────
    gat  = GATEmbedding(cfg.gat).to(device)
    head = _build_head(cfg, embedding_dim=cfg.gat.output_dim)
    if head is not None:
        head = head.to(device)

    head_name = (
        "slot_attention"     if cfg.slot_attention    is not None else
        "graph_partitioning" if cfg.graph_partitioning is not None else
        "knn_only"
    )
    print(f"\nTraining: {cfg.name}")
    print(f"Head:     {head_name}")
    print(f"Device:   {device}")
    print(f"Epochs:   {tc.epochs}")
    print(f"Save dir: {save_dir}\n")

    # ── Losses ────────────────────────────────────────────────────────────────
    gat_loss_fn  = GATOnlyLoss(cfg.loss)
    head_loss_fn = _build_head_loss(cfg)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    params = list(gat.parameters())
    if head is not None:
        params += list(head.parameters())

    optimizer = optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tc.epochs, eta_min=tc.lr_min
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_pga = 0.0
    history      = []

    for epoch in range(1, tc.epochs + 1):
        gat.train()
        if head is not None:
            head.train()

        epoch_loss      = 0.0
        epoch_gat_loss  = 0.0
        epoch_head_loss = 0.0
        n_graphs        = 0
        t0              = time.time()

        for batch in train_loader:
            graphs = preprocessor.process_batch(batch)

            for graph in graphs:
                graph = graph.to(device)
                optimizer.zero_grad()

                # GAT forward
                embeddings = gat(graph)

                # Contrastive loss
                gat_out  = gat_loss_fn(embeddings, graph.person_labels)
                total    = gat_out["total_loss"]

                # Head loss
                head_loss_val = 0.0
                if head is not None and head_loss_fn is not None:
                    _, loss_inputs = _head_forward(head, embeddings, graph)
                    head_out       = head_loss_fn(*loss_inputs)
                    head_loss_val  = head_out["total_loss"]
                    total          = total + head_loss_val

                total.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss      += total.item()
                epoch_gat_loss  += gat_out["total_loss"].item()
                epoch_head_loss += (head_loss_val.item()
                                    if isinstance(head_loss_val, torch.Tensor)
                                    else head_loss_val)
                n_graphs += 1

        scheduler.step()

        avg_loss     = epoch_loss      / max(n_graphs, 1)
        avg_gat      = epoch_gat_loss  / max(n_graphs, 1)
        avg_head     = epoch_head_loss / max(n_graphs, 1)
        elapsed      = time.time() - t0

        log = (f"Epoch {epoch:3d}/{tc.epochs} | "
               f"loss {avg_loss:.4f} "
               f"(gat {avg_gat:.4f} head {avg_head:.4f}) | "
               f"{elapsed:.1f}s")

        # ── Validation ────────────────────────────────────────────────────────
        if epoch % tc.val_every == 0 or epoch == tc.epochs:
            val_pga = _validate(gat, head, head_name, val_loader,
                                preprocessor, gat_loss_fn, head_loss_fn, device, cfg)
            log += f" | val_pga {val_pga:.4f}"

            if tc.save_best and val_pga > best_val_pga:
                best_val_pga = val_pga
                _save_checkpoint(save_dir / "best.pt", gat, head, optimizer,
                                 epoch, val_pga, cfg)
                log += "  ← best"

            history.append({
                "epoch": epoch, "loss": avg_loss,
                "gat_loss": avg_gat, "head_loss": avg_head,
                "val_pga": val_pga,
            })

        print(log)

    # Always save final
    _save_checkpoint(save_dir / "final.pt", gat, head, optimizer,
                     tc.epochs, best_val_pga, cfg)

    # Save history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val PGA: {best_val_pga:.4f}")
    print(f"Checkpoints saved to {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────

def _validate(gat, head, head_name, loader, preprocessor,
              gat_loss_fn, head_loss_fn, device, cfg) -> float:
    """Run validation, return mean PGA."""
    from evaluator import compute_pga, predict_knn, predict_slot, predict_partition

    gat.eval()
    if head is not None:
        head.eval()

    all_pga = []

    with torch.no_grad():
        for batch in loader:
            graphs = preprocessor.process_batch(batch)
            for graph in graphs:
                graph      = graph.to(device)
                embeddings = gat(graph)
                k          = int(graph.num_people)

                if head_name == "slot_attention":
                    pred_labels = predict_slot(head, embeddings, k)
                elif head_name == "graph_partitioning":
                    pred_labels = predict_partition(
                        head, embeddings,
                        cfg.graph_partitioning.threshold
                    )
                else:
                    pred_labels = predict_knn(embeddings, k)

                pga = compute_pga(pred_labels, graph.person_labels)
                all_pga.append(pga)

    gat.train()
    if head is not None:
        head.train()

    return sum(all_pga) / max(len(all_pga), 1)


def _save_checkpoint(path, gat, head, optimizer, epoch, val_pga, cfg):
    torch.save({
        "epoch":     epoch,
        "val_pga":   val_pga,
        "gat_state": gat.state_dict(),
        "head_state": head.state_dict() if head is not None else None,
        "opt_state":  optimizer.state_dict(),
        "config":     cfg.model_dump(),
    }, path)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/train_slot.yaml"))
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    train(cfg, args.device)


if __name__ == "__main__":
    main()