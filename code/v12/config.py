from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, model_validator
import yaml

class GATConfig(BaseModel):
    num_joint_types: int
    joint_embedding_dim: int
    raw_feature_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    use_layer_norm: bool
    l2_normalize: bool


class DETRConfig(BaseModel):
    embedding_dim: int
    max_people: int
    num_decoder_layers: int
    num_heads: int
    ffn_dim: int
    dropout: float
    num_joint_types: int

class LossConfig(BaseModel):
    lambda_existence: float
    lambda_assignment: float
    lambda_contrastive: float
    lambda_count: float
    lambda_auxiliary: float = 1.0
    contrastive_margin: float
    label_smoothing: float


class TrainerConfig(BaseModel):
    mode: Literal["gat_only", "full"]
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    warmup_epochs: int
    min_lr: float
    query_lr_multiplier: float = 10.0
    epochs: int
    batch_size: int
    print_every_n_epochs: int
    visualize_every_n_epochs: int
    checkpoint_every_n_epochs: int
    compute_clustering_metrics: bool


class DataConfig(BaseModel):
    split: str
    test_split: Optional[str]
    image_size: int = 512
    k_neighbors: int
    min_visibility: int
    num_workers: int
    shuffle: bool

class EvalConfig(BaseModel):
    existence_threshold: float
    use_count_head: bool = True
    save_visualizations: bool
    max_visualizations: int
    compute_per_joint_metrics: bool


class ExperimentConfig(BaseModel):
    name: str = "default"
    description: str = ""
    
    gat: GATConfig
    detr: DETRConfig
    loss: LossConfig
    trainer: TrainerConfig
    data: DataConfig
    eval: EvalConfig
    
    @model_validator(mode="after")
    def sync_dimensions(self):
        """Ensure DETR embedding_dim matches GAT output_dim."""
        if self.detr.embedding_dim != self.gat.output_dim:
            self.detr.embedding_dim = self.gat.output_dim
        if self.detr.num_joint_types != self.gat.num_joint_types:
            self.detr.num_joint_types = self.gat.num_joint_types
        return self
    
    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
