from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
import json
from pydantic import Field

class Settings(BaseSettings):
    """Project settings with paths for data directories."""

    root_dir: Path = Field(
        default=Path(__file__).resolve().parent.parent.parent,
        description="Root directory of the project"
    )

    data_dir: Optional[Path] = None
    coco_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    coco_annotations_dir: Optional[Path] = None
    coco_train_dir: Optional[Path] = None
    coco_val_dir: Optional[Path] = None
    coco_test_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    target_image_size: int = 512
    print_trainer_results_every_n_epochs: int = 10
    visualize_trainer_results_every_n_epochs: int = 25
    
    # hyperparameters 
    num_people_required: int = 2 # -1 means no restriction
    batch_size: int = 4
    learning_rate: float = 3e-4
    epochs: int = 300
    margin: float = 10.0 # Contrastive loss margin
    input_dim: int = 5 # [x, y, depth, conf, joint_type]
    hidden_dim: int = 128 
    output_dim: int = 32
    num_heads: int = 8
    dropout: float = 0.4
    
    def save_hyperparameters(self, path: Path):
        if path.suffix == "" or path.is_dir():
            path = path / "hyperparameters.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        
        hyper_keys = [
            "num_people_required",
            "batch_size",
            "learning_rate",
            "epochs",
            "margin",
            "input_dim",
            "hidden_dim",
            "output_dim",
            "num_heads",
            "dropout"
        ]
        
        hyperparams = {k: getattr(self, k) for k in hyper_keys}
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(hyperparams, f, indent=4)


    class Config:
        env_prefix = "PROJECT_"
        arbitrary_types_allowed = True

    def model_post_init(self, __context):
        """Derive default paths from root_dir if not provided; ensure output dir exists."""
        r = self.root_dir

        self.data_dir = self.data_dir or (r / "data")
        self.coco_dir = self.coco_dir or (self.data_dir / "coco2017")
        self.cache_dir = self.cache_dir or (self.coco_dir / "cache")
        self.coco_annotations_dir = self.coco_annotations_dir or (self.coco_dir / "annotations")
        self.coco_train_dir = self.coco_train_dir or (self.coco_dir / "train2017")
        self.coco_val_dir = self.coco_val_dir or (self.coco_dir / "val2017")
        self.coco_test_dir = self.coco_test_dir or (self.coco_dir / "test2017")
        self.output_dir = self.output_dir or (r / "outputs" / "pipeline")

        self.output_dir.mkdir(parents=True, exist_ok=True)

settings = Settings()