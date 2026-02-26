# main.py
"""Train full GAT + DETR model"""
import torch
from gat import GATEmbedding, GATConfig
from detr_decoder import DETRConfig, PoseGroupingModel
from virtual_preprocess import VirtualKeypointPreprocessor
from virtual_dataloader import create_virtual_dataloader
from trainer import Trainer, TrainerConfig


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataloader = create_virtual_dataloader(split="mixed", shuffle=True)
    
    preprocessor = VirtualKeypointPreprocessor(
        device=device,
        k_neighbors=8,
        image_size=512
    )
    
    # GAT config
    gat_config = GATConfig(
        num_joint_types=17,
        joint_embedding_dim=32,
        raw_feature_dim=4,
        hidden_dim=64,
        output_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        use_layer_norm=True,
        l2_normalize=True
    )
    gat = GATEmbedding(gat_config)
    
    # DETR config
    detr_config = DETRConfig(
        embedding_dim=128,  # Must match GAT output
        max_people=10,
        num_decoder_layers=3,
        num_heads=8,
        ffn_dim=512,
        dropout=0.1,
        num_joint_types=17
    )
    
    # Full model
    model = PoseGroupingModel(gat, detr_config)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Trainer config
    trainer_config = TrainerConfig(
        mode="full",
        learning_rate=1e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_epochs=5,
        min_lr=1e-6,
        lambda_existence=1.0,
        lambda_assignment=5.0,
        lambda_contrastive=2.0,
        contrastive_margin=0.5,
        epochs=100,
        print_every_n_epochs=5,
        visualize_every_n_epochs=20,
        checkpoint_every_n_epochs=25,
        compute_clustering_metrics=True
    )
    
    trainer = Trainer(model, preprocessor, config=trainer_config, device=device)
    trainer.train(dataloader)


if __name__ == "__main__":
    main()