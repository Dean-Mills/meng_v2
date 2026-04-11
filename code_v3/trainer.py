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
import os
import shutil
import time
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from config import ExperimentConfig
from gat import GATEmbedding
from losses import GATOnlyLoss, SlotAttentionLoss, GraphPartitioningLoss, DMoNLoss, SADMoNLoss, SCOTLoss
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

    if cfg.dmon is not None:
        from dmon import DMoNHead
        return DMoNHead(cfg.dmon, embedding_dim=embedding_dim)

    if cfg.sa_dmon is not None:
        from sa_dmon import SADMoNHead
        return SADMoNHead(cfg.sa_dmon, embedding_dim=embedding_dim)

    if cfg.sa_dmon_v2 is not None:
        from sa_dmon_v2 import SADMoNV2Head
        return SADMoNV2Head(cfg.sa_dmon_v2, embedding_dim=embedding_dim)

    if cfg.scot is not None:
        from ot_head import SCOTHead
        return SCOTHead(cfg.scot, embedding_dim=embedding_dim)

    if cfg.residual_scot is not None:
        from ot_head_residual import ResidualSCOTHead
        return ResidualSCOTHead(cfg.residual_scot, embedding_dim=embedding_dim)

    if cfg.adaptive_scot is not None:
        from ot_head_adaptive import AdaptiveSCOTHead
        return AdaptiveSCOTHead(cfg.adaptive_scot, embedding_dim=embedding_dim)

    if cfg.unbalanced_scot is not None:
        from ot_head_unbalanced import UnbalancedSCOTHead
        return UnbalancedSCOTHead(cfg.unbalanced_scot, embedding_dim=embedding_dim)

    if cfg.dustbin_scot is not None:
        from ot_head_dustbin import DustbinSCOTHead
        return DustbinSCOTHead(cfg.dustbin_scot, embedding_dim=embedding_dim)

    return None


def _build_head_loss(cfg: ExperimentConfig):
    """Instantiate the head loss from config."""
    if cfg.slot_attention is not None:
        return SlotAttentionLoss(cfg.loss)
    if cfg.graph_partitioning is not None:
        return GraphPartitioningLoss(cfg.loss)
    if cfg.dmon is not None:
        return DMoNLoss(cfg.loss, cfg.dmon)
    if cfg.sa_dmon is not None:
        return SADMoNLoss(cfg.loss, cfg.sa_dmon)
    if cfg.sa_dmon_v2 is not None:
        # SADMoNV2Config has the same lambda fields as SADMoNConfig
        return SADMoNLoss(cfg.loss, cfg.sa_dmon_v2)
    if cfg.scot is not None:
        return SCOTLoss(cfg.loss)
    if cfg.residual_scot is not None:
        return SCOTLoss(cfg.loss)
    if cfg.adaptive_scot is not None:
        return SCOTLoss(cfg.loss)
    if cfg.unbalanced_scot is not None:
        return SCOTLoss(cfg.loss)
    if cfg.dustbin_scot is not None:
        return SCOTLoss(cfg.loss)
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

    from dmon import DMoNHead
    if isinstance(head, DMoNHead):
        k = int(graph.num_people)
        logits, s, spec, ortho, clust, type_l = head(
            embeddings, graph.edge_index, k, joint_types=graph.joint_types,
        )
        return (logits, s), (logits, graph.person_labels, spec, ortho, clust, type_l)

    from sa_dmon import SADMoNHead
    if isinstance(head, SADMoNHead):
        k = int(graph.num_people)
        positions = graph.x[:, :2]
        logits, s, spec, ortho, clust, type_l = head(
            embeddings, graph.edge_index, k, positions, graph.joint_types,
        )
        return (logits, s), (logits, graph.person_labels, spec, ortho, clust, type_l)

    from sa_dmon_v2 import SADMoNV2Head
    if isinstance(head, SADMoNV2Head):
        k = int(graph.num_people)
        positions = graph.x[:, :2]
        logits, s, spec, ortho, clust, type_l = head(
            embeddings, graph.edge_index, k, positions, graph.joint_types,
        )
        return (logits, s), (logits, graph.person_labels, spec, ortho, clust, type_l)

    from ot_head import SCOTHead
    if isinstance(head, SCOTHead):
        k = int(graph.num_people)
        logits, T = head(embeddings, k, graph.joint_types)
        return (logits, T), (logits, graph.person_labels)

    from ot_head_residual import ResidualSCOTHead
    if isinstance(head, ResidualSCOTHead):
        k = int(graph.num_people)
        positions = graph.x[:, :2]
        logits, T = head(embeddings, k, positions, graph.joint_types)
        return (logits, T), (logits, graph.person_labels)

    from ot_head_adaptive import AdaptiveSCOTHead
    if isinstance(head, AdaptiveSCOTHead):
        k = int(graph.num_people)
        logits, T = head(embeddings, k, graph.joint_types)
        return (logits, T), (logits, graph.person_labels)

    from ot_head_unbalanced import UnbalancedSCOTHead
    if isinstance(head, UnbalancedSCOTHead):
        k = int(graph.num_people)
        logits, T = head(embeddings, graph.joint_types, k=k)
        return (logits, T), (logits, graph.person_labels)

    from ot_head_dustbin import DustbinSCOTHead
    if isinstance(head, DustbinSCOTHead):
        k = int(graph.num_people)
        logits, T = head(embeddings, graph.joint_types, k=k)
        return (logits, T), (logits, graph.person_labels)

    return None, None


def _make_loader(virtual_dir: Path, split: str, batch_size: int, num_workers: int):
    adapter = VirtualAdapter(virtual_dir / split)
    dataset = PoseDataset(adapter)
    return create_dataloader(dataset, batch_size=batch_size,
                             shuffle=(split == "train"),
                             num_workers=num_workers)


# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: ExperimentConfig, device: str, config_path: Path = None,
          name_override: str = None):
    assert cfg.training is not None, "training config missing from yaml"
    tc = cfg.training

    # Derive run name from config filename or --name override
    if name_override:
        run_name = name_override
    elif config_path is not None:
        run_name = config_path.stem
    else:
        run_name = cfg.name

    # Create GUID-based run directory: outputs/{run_name}/{guid}/
    run_id = uuid.uuid4().hex[:8]
    save_dir = Path("outputs") / run_name / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # Copy config into run directory for reproducibility
    if config_path is not None and config_path.exists():
        shutil.copy2(config_path, save_dir / "config.yaml")

    # Update latest symlink
    latest_link = Path("outputs") / run_name / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_id)

    virtual_dir = Path(tc.virtual_dir)

    # ── Models ────────────────────────────────────────────────────────────────
    if cfg.sa_gat is not None:
        from sa_gat import SAGATEmbedding
        gat = SAGATEmbedding(cfg.sa_gat).to(device)
        embedding_dim = cfg.sa_gat.output_dim
        use_depth = cfg.sa_gat.use_depth
    else:
        gat = GATEmbedding(cfg.gat).to(device)
        embedding_dim = cfg.gat.output_dim
        use_depth = cfg.gat.use_depth

    # ── Data ──────────────────────────────────────────────────────────────────
    if tc.coco_train_dir is not None and tc.coco_train_ann is not None:
        from coco_adapter import CocoAdapter
        train_adapter = CocoAdapter(
            img_dir=Path(tc.coco_train_dir),
            ann_file=Path(tc.coco_train_ann),
            use_depth=use_depth,
        )
        train_dataset = PoseDataset(train_adapter)
        train_loader = create_dataloader(train_dataset, batch_size=tc.batch_size,
                                         shuffle=True, num_workers=tc.num_workers)
    else:
        train_loader = _make_loader(virtual_dir, "train", tc.batch_size, tc.num_workers)

    # Validation always on virtual (consistent benchmark)
    val_loader = _make_loader(virtual_dir, "val", tc.batch_size, 0)

    k_neighbors  = 16 if embedding_dim >= 256 else 8
    preprocessor = PosePreprocessor(device=device, k_neighbors=k_neighbors,
                                    use_depth=use_depth)
    head = _build_head(cfg, embedding_dim=embedding_dim)
    if head is not None:
        head = head.to(device)

    # ── Load pretrained weights (for fine-tuning) ─────────────────────────────
    if tc.pretrained is not None:
        pretrained_ckpt = torch.load(tc.pretrained, map_location=device,
                                     weights_only=False)
        gat.load_state_dict(pretrained_ckpt["gat_state"])
        if head is not None and pretrained_ckpt.get("head_state") is not None:
            head.load_state_dict(pretrained_ckpt["head_state"])
        print(f"Loaded pretrained weights from {tc.pretrained}")

    head_name = (
        "slot_attention"     if cfg.slot_attention     is not None else
        "graph_partitioning" if cfg.graph_partitioning is not None else
        "dmon"               if cfg.dmon               is not None else
        "sa_dmon"            if cfg.sa_dmon             is not None else
        "sa_dmon_v2"         if cfg.sa_dmon_v2          is not None else
        "scot"               if cfg.scot               is not None else
        "residual_scot"      if cfg.residual_scot       is not None else
        "adaptive_scot"      if cfg.adaptive_scot       is not None else
        "unbalanced_scot"    if cfg.unbalanced_scot     is not None else
        "dustbin_scot"       if cfg.dustbin_scot        is not None else
        "knn_only"
    )
    # ── K estimation head (optional) ─────────────────────────────────────────
    k_head = None
    if cfg.train_k_head:
        from k_head import KEstimationHead
        k_head = KEstimationHead(embedding_dim=embedding_dim).to(device)

    print(f"\nTraining: {run_name}")
    print(f"Run ID:   {run_id}")
    print(f"Head:     {head_name}")
    print(f"K head:   {'yes' if k_head else 'no'}")
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
    if k_head is not None:
        params += list(k_head.parameters())

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
                # Skip scenes that exceed SCOT's k_max
                if cfg.scot is not None and graph.num_people > cfg.scot.k_max:
                    continue

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

                # K estimation loss
                if k_head is not None:
                    k_pred = k_head(embeddings)
                    k_gt = torch.tensor(float(graph.num_people), device=device)
                    k_loss = F.l1_loss(k_pred, k_gt)
                    total = total + k_loss

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
                                 epoch, val_pga, cfg, k_head=k_head)
                log += "  ← best"

            history.append({
                "epoch": epoch, "loss": avg_loss,
                "gat_loss": avg_gat, "head_loss": avg_head,
                "val_pga": val_pga,
            })

        print(log)

    # Always save final
    _save_checkpoint(save_dir / "final.pt", gat, head, optimizer,
                     tc.epochs, best_val_pga, cfg, k_head=k_head)

    # Save history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val PGA: {best_val_pga:.4f}")
    print(f"Checkpoints saved to {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────

def _validate(gat, head, head_name, loader, preprocessor,
              gat_loss_fn, head_loss_fn, device, cfg) -> float:
    """Run validation, return mean PGA."""
    from evaluator import compute_pga, predict_knn, predict_slot, predict_partition, predict_dmon, predict_sa_dmon, predict_scot, predict_residual_scot, predict_adaptive_scot, predict_unbalanced_scot, predict_dustbin_scot

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
                elif head_name == "dmon":
                    pred_labels = predict_dmon(
                        head, embeddings, graph.edge_index, k,
                        graph.joint_types,
                    )
                elif head_name in ("sa_dmon", "sa_dmon_v2"):
                    pred_labels = predict_sa_dmon(
                        head, embeddings, graph.edge_index, k,
                        graph.x[:, :2], graph.joint_types,
                    )
                elif head_name == "scot":
                    pred_labels = predict_scot(
                        head, embeddings, k, graph.joint_types,
                    )
                elif head_name == "residual_scot":
                    pred_labels = predict_residual_scot(
                        head, embeddings, k, graph.x[:, :2],
                        graph.joint_types,
                    )
                elif head_name == "adaptive_scot":
                    pred_labels = predict_adaptive_scot(
                        head, embeddings, k, graph.joint_types,
                    )
                elif head_name == "unbalanced_scot":
                    pred_labels = predict_unbalanced_scot(
                        head, embeddings, k, graph.joint_types,
                    )
                elif head_name == "dustbin_scot":
                    pred_labels = predict_dustbin_scot(
                        head, embeddings, k, graph.joint_types,
                    )
                else:
                    pred_labels = predict_knn(embeddings, k)

                pga = compute_pga(pred_labels, graph.person_labels)
                all_pga.append(pga)

    gat.train()
    if head is not None:
        head.train()

    return sum(all_pga) / max(len(all_pga), 1)


def _save_checkpoint(path, gat, head, optimizer, epoch, val_pga, cfg, k_head=None):
    torch.save({
        "epoch":     epoch,
        "val_pga":   val_pga,
        "gat_state": gat.state_dict(),
        "head_state": head.state_dict() if head is not None else None,
        "k_head_state": k_head.state_dict() if k_head is not None else None,
        "opt_state":  optimizer.state_dict(),
        "config":     cfg.model_dump(),
    }, path)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=Path("configs/train_slot.yaml"))
    parser.add_argument("--name", type=str, default=None,
                        help="Override run name (default: derived from config filename)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    train(cfg, args.device, config_path=args.config, name_override=args.name)


if __name__ == "__main__":
    main()