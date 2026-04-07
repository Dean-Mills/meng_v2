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
    use_depth: bool = True  # False → drop z from node features and kNN

    @computed_field
    @property
    def input_dim(self) -> int:
        # raw_feature_dim is the base (with depth), subtract 1 when depth disabled
        feat_dim = self.raw_feature_dim if self.use_depth else self.raw_feature_dim - 1
        return feat_dim + self.joint_embedding_dim


class SAGATConfig(BaseModel):
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
    use_depth: bool = True
    # SA-GAT modifications (each independently toggleable)
    use_type_pair_attention: bool = True    # Mod 1: category-specific attention
    use_position_encoding: bool = True      # Mod 2: skeleton-relative pos encoding
    use_repulsion_heads: bool = True        # Mod 3: dedicated same-type heads
    n_repulsion_heads: int = 1              # how many heads reserved for same-type

    @computed_field
    @property
    def input_dim(self) -> int:
        feat_dim = self.raw_feature_dim if self.use_depth else self.raw_feature_dim - 1
        return feat_dim + self.joint_embedding_dim


class DECConfig(BaseModel):
    n_clusters:      int   # K — number of people, passed explicitly per scene at inference
    alpha:           float = 1.0    # degrees of freedom for Student's t-distribution
    update_interval: int   = 100    # steps between target distribution refreshes


class LossConfig(BaseModel):
    contrastive_margin: float
    dec_weight:        float = 1.0   # weight for KL divergence loss when DEC is used
    slot_weight:       float = 1.0   # weight for slot attention loss
    partition_weight:  float = 1.0   # weight for graph partitioning loss
    dmon_weight:       float = 1.0   # weight for DMoN loss
    sa_dmon_weight:    float = 1.0   # weight for SA-DMoN loss
    scot_weight:       float = 1.0   # weight for SCOT loss
    residual_scot_weight: float = 1.0


class GraphPartitioningConfig(BaseModel):
    hidden_dim:      int   = 256    # MLP hidden dimension
    dropout:         float = 0.1    # dropout in MLP
    threshold:       float = 0.5    # affinity threshold for connected components at inference
    partition_weight: float = 1.0   # loss weight


class SlotAttentionConfig(BaseModel):
    num_iterations: int   = 3      # refinement iterations — 3 is standard
    slot_weight:    float = 1.0    # weight for slot attention loss


class DMoNConfig(BaseModel):
    hidden_dim:         int   = 256    # encoder hidden dimension
    k_max:              int   = 10     # maximum number of clusters (pool size)
    dropout:            float = 0.0    # dropout on assignment logits
    # Sub-weights for individual loss components
    lambda_spectral:    float = 1.0    # spectral modularity loss
    lambda_ortho:       float = 1.0    # orthogonality regularization
    lambda_cluster:     float = 1.0    # collapse prevention
    lambda_type:        float = 1.0    # type exclusivity (modification 1)
    lambda_supervised:  float = 1.0    # supervised CE (modification 3)


class SCOTConfig(BaseModel):
    hidden_dim:         int   = 256    # encoder hidden dimension
    k_max:              int   = 10     # maximum number of people (prototype pool size)
    sinkhorn_iters:     int   = 10     # Sinkhorn iterations
    sinkhorn_tau:       float = 0.1    # temperature (lower = harder assignments)


class DustbinSCOTConfig(BaseModel):
    hidden_dim:         int   = 256
    k_max:              int   = 20
    sinkhorn_iters:     int   = 10
    sinkhorn_tau:       float = 0.1
    dustbin_cost_init:  float = 1.0    # initial cost for rejecting a keypoint
    person_threshold:   float = 0.5    # min mass to count as active person


class UnbalancedSCOTConfig(BaseModel):
    hidden_dim:         int   = 256
    k_max:              int   = 20     # generous upper bound on person count
    sinkhorn_iters:     int   = 20     # more iters needed for unbalanced convergence
    sinkhorn_tau:       float = 0.1    # entropy regularisation
    rho:                float = 1.0    # KL penalty on target marginal (lower = more relaxed)
    person_threshold:   float = 0.5    # minimum total mass to count as active person


class AdaptiveSCOTConfig(BaseModel):
    hidden_dim:         int   = 256
    k_max:              int   = 10
    sinkhorn_iters:     int   = 10
    tau_min:            float = 0.01   # minimum temperature (sharp, kNN-like)
    tau_max:            float = 0.5    # maximum temperature (soft, global OT)


class ResidualSCOTConfig(BaseModel):
    hidden_dim:         int   = 256
    k_max:              int   = 10
    sinkhorn_iters:     int   = 10
    sinkhorn_tau:       float = 0.1
    lambda_residual:    float = 0.2    # weight of learned cost relative to spatial


class SADMoNV2Config(BaseModel):
    hidden_dim:              int   = 256
    k_max:                   int   = 10
    dropout:                 float = 0.0
    use_feature_adjacency:   bool  = True   # Fix 1: feature-based adj for modularity
    use_entropy_type_loss:   bool  = True   # Fix 2: entropy instead of ReLU type loss
    feature_knn_k:           int   = 8      # kNN k for feature adjacency
    # Loss weights (same structure as SADMoNConfig)
    lambda_spectral:         float = 1.0
    lambda_ortho:            float = 0.1
    lambda_cluster:          float = 0.1
    lambda_type:             float = 1.0
    lambda_supervised:       float = 1.0


class SADMoNConfig(BaseModel):
    hidden_dim:         int   = 256    # encoder hidden dimension
    k_max:              int   = 10     # maximum number of clusters (pool size)
    dropout:            float = 0.0    # dropout on assignment logits
    # Sub-weights for individual loss components
    lambda_spectral:    float = 1.0    # skeleton-aware spectral modularity loss
    lambda_ortho:       float = 1.0    # orthogonality regularization
    lambda_cluster:     float = 1.0    # collapse prevention
    lambda_type:        float = 1.0    # type exclusivity
    lambda_supervised:  float = 1.0    # supervised CE


class TrainingConfig(BaseModel):
    # Data paths — expect train/ val/ test/ subdirectories of virtual_dir
    virtual_dir:   str = "data/virtual"
    # COCO fine-tuning (optional — if set, trains on COCO instead of virtual)
    coco_train_dir:  Optional[str] = None   # e.g. "data/coco2017/train2017"
    coco_train_ann:  Optional[str] = None   # e.g. "data/coco2017/annotations/person_keypoints_train2017.json"
    # Pre-trained checkpoint to load before training (for fine-tuning)
    pretrained:      Optional[str] = None
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
    sa_gat:             Optional[SAGATConfig]             = None
    loss: LossConfig
    dec:                Optional[DECConfig]               = None
    slot_attention:     Optional[SlotAttentionConfig]     = None
    graph_partitioning: Optional[GraphPartitioningConfig] = None
    dmon:               Optional[DMoNConfig]              = None
    sa_dmon:            Optional[SADMoNConfig]            = None
    sa_dmon_v2:         Optional[SADMoNV2Config]          = None
    scot:               Optional[SCOTConfig]              = None
    residual_scot:      Optional[ResidualSCOTConfig]      = None
    adaptive_scot:      Optional[AdaptiveSCOTConfig]      = None
    unbalanced_scot:    Optional[UnbalancedSCOTConfig]    = None
    dustbin_scot:       Optional[DustbinSCOTConfig]       = None
    train_k_head:       bool                              = False
    training:           Optional[TrainingConfig]          = None

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)