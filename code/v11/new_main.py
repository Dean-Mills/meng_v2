from pathlib import Path
import torch

from config import ExperimentConfig
from gat import GATEmbedding, GATConfig as GATConfigOriginal
from detr_decoder import DETRConfig as DETRConfigOriginal, PoseGroupingModel
from virtual_preprocess import VirtualKeypointPreprocessor
from virtual_dataloader import create_virtual_dataloader
from trainer import Trainer, TrainerConfig as TrainerConfigOriginal
from evaluate import Evaluator, EvalConfig as EvalConfigOriginal
    
import matplotlib
matplotlib.use('Agg')

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG_NAME = "baseline"
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
# =============================================================================


def build_model(cfg: ExperimentConfig):
    """Build GAT or GAT+DETR model from config."""
    
    gat_config = GATConfigOriginal(
        num_joint_types=cfg.gat.num_joint_types,
        joint_embedding_dim=cfg.gat.joint_embedding_dim,
        raw_feature_dim=cfg.gat.raw_feature_dim,
        hidden_dim=cfg.gat.hidden_dim,
        output_dim=cfg.gat.output_dim,
        num_layers=cfg.gat.num_layers,
        num_heads=cfg.gat.num_heads,
        dropout=cfg.gat.dropout,
        use_layer_norm=cfg.gat.use_layer_norm,
        l2_normalize=cfg.gat.l2_normalize,
    )
    gat = GATEmbedding(gat_config)
    
    if cfg.trainer.mode == "gat_only":
        return gat
    
    detr_config = DETRConfigOriginal(
        embedding_dim=cfg.detr.embedding_dim,
        max_people=cfg.detr.max_people,
        num_decoder_layers=cfg.detr.num_decoder_layers,
        num_heads=cfg.detr.num_heads,
        ffn_dim=cfg.detr.ffn_dim,
        dropout=cfg.detr.dropout,
        num_joint_types=cfg.detr.num_joint_types,
    )
    
    return PoseGroupingModel(gat, detr_config)


def train(cfg: ExperimentConfig, config_name: str, device: torch.device) -> Path:
    """Train model and return checkpoint path."""
    
    # Dataloader
    from settings import settings
    original_batch_size = settings.batch_size
    settings.batch_size = cfg.trainer.batch_size
    
    dataloader = create_virtual_dataloader(
        split=cfg.data.split,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers
    )
    settings.batch_size = original_batch_size
    
    # Preprocessor
    preprocessor = VirtualKeypointPreprocessor(
        device=device,
        k_neighbors=cfg.data.k_neighbors,
        image_size=cfg.data.image_size
    )
    
    # Model
    model = build_model(cfg)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Trainer config
    trainer_config = TrainerConfigOriginal(
        mode=cfg.trainer.mode,
        learning_rate=cfg.trainer.learning_rate,
        weight_decay=cfg.trainer.weight_decay,
        max_grad_norm=cfg.trainer.max_grad_norm,
        warmup_epochs=cfg.trainer.warmup_epochs,
        min_lr=cfg.trainer.min_lr,
        lambda_existence=cfg.loss.lambda_existence,
        lambda_assignment=cfg.loss.lambda_assignment,
        lambda_contrastive=cfg.loss.lambda_contrastive,
        contrastive_margin=cfg.loss.contrastive_margin,
        epochs=cfg.trainer.epochs,
        print_every_n_epochs=cfg.trainer.print_every_n_epochs,
        visualize_every_n_epochs=cfg.trainer.visualize_every_n_epochs,
        checkpoint_every_n_epochs=cfg.trainer.checkpoint_every_n_epochs,
        compute_clustering_metrics=cfg.trainer.compute_clustering_metrics,
    )
    
    # Output directory includes config name
    from settings import settings
    output_dir = settings.output_dir / config_name
    
    trainer = Trainer(model, preprocessor, config=trainer_config, device=device, output_dir=output_dir)
    
    # Save experiment config alongside checkpoints
    cfg.to_yaml(trainer.output_dir / "experiment_config.yaml")
    
    # Train
    trainer.train(dataloader)
    
    return trainer.checkpoint_dir / "model_final.pt"


def evaluate(cfg: ExperimentConfig, checkpoint_path: Path, device: torch.device):
    """Evaluate trained model."""
    
    eval_config = EvalConfigOriginal(
        existence_threshold=cfg.eval.existence_threshold,
        save_visualizations=cfg.eval.save_visualizations,
        max_visualizations=cfg.eval.max_visualizations,
        compute_per_joint_metrics=cfg.eval.compute_per_joint_metrics,
    )
    
    evaluator = Evaluator(str(checkpoint_path), config=eval_config, device=device)
    summary = evaluator.evaluate_dataset(split=cfg.data.test_split)
    
    if summary:
        evaluator.plot_per_joint_accuracy(summary)
    
    return summary


def main():
    config_path = CONFIGS_DIR / f"{CONFIG_NAME}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    cfg = ExperimentConfig.from_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print(f"EXPERIMENT: {CONFIG_NAME}")
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print("=" * 70)
    
    # Train
    checkpoint_path = train(cfg, CONFIG_NAME, device)
    
    # Evaluate
    if cfg.data.test_split:
        print("\n" + "=" * 70)
        print("EVALUATION")
        print("=" * 70)
        evaluate(cfg, checkpoint_path, device)


if __name__ == "__main__":
    main()
