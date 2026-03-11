from pathlib import Path
from typing import Optional
from pydantic import BaseModel, computed_field
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

    @computed_field
    @property
    def input_dim(self) -> int:
        return self.raw_feature_dim + self.joint_embedding_dim


class DECConfig(BaseModel):
    n_clusters:      int   # K — number of people, passed explicitly per scene at inference
    alpha:           float = 1.0    # degrees of freedom for Student's t-distribution
    update_interval: int   = 100    # steps between target distribution refreshes


class LossConfig(BaseModel):
    contrastive_margin: float
    dec_weight:   float = 1.0   # weight for KL divergence loss when DEC is used
    slot_weight:       float = 1.0   # weight for slot attention loss
    partition_weight:  float = 1.0   # weight for graph partitioning loss


class GraphPartitioningConfig(BaseModel):
    hidden_dim:      int   = 256    # MLP hidden dimension
    dropout:         float = 0.1    # dropout in MLP
    threshold:       float = 0.5    # affinity threshold for connected components at inference
    partition_weight: float = 1.0   # loss weight


class SlotAttentionConfig(BaseModel):
    num_iterations: int   = 3      # refinement iterations — 3 is standard
    slot_weight:    float = 1.0    # weight for slot attention loss


class TrainingConfig(BaseModel):
    # Data paths — expect train/ val/ test/ subdirectories of virtual_dir
    virtual_dir:   str = "data/virtual"
    # Loader
    batch_size:    int   = 4
    num_workers:   int   = 4
    # Optimiser
    lr:            float = 1e-3
    lr_min:        float = 1e-5
    weight_decay:  float = 1e-4
    # Schedule
    epochs:        int   = 50
    val_every:     int   = 5       # validate every N epochs
    # Checkpointing
    save_dir:      str   = "outputs/checkpoints"
    save_best:     bool  = True


class ExperimentConfig(BaseModel):
    name: str = "default"
    description: str = ""
    gat: GATConfig
    loss: LossConfig
    dec:            Optional[DECConfig]           = None
    slot_attention:    Optional[SlotAttentionConfig]   = None
    graph_partitioning: Optional[GraphPartitioningConfig] = None
    training:           Optional[TrainingConfig]           = None

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)