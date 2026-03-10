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
    slot_weight:  float = 1.0   # weight for slot attention loss


class SlotAttentionConfig(BaseModel):
    num_iterations: int   = 3      # refinement iterations — 3 is standard
    slot_weight:    float = 1.0    # weight for slot attention loss


class ExperimentConfig(BaseModel):
    name: str = "default"
    description: str = ""
    gat: GATConfig
    loss: LossConfig
    dec:            Optional[DECConfig]           = None
    slot_attention: Optional[SlotAttentionConfig]  = None

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)