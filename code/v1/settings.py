from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Project settings with paths for data directories."""
    
    root_dir: Path = Field(
        default=Path(__file__).parent.parent.parent,
        description="Root directory of the project"
    )
    
    data_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    coco_dir: Optional[Path] = None
    coco_annotations_dir: Optional[Path] = None
    coco_train_dir: Optional[Path] = None
    coco_val_dir: Optional[Path] = None
    coco_test_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    target_image_size: int = 512
    
    class Config:
        env_prefix = "PROJECT_"
        arbitrary_types_allowed = True
    
    def model_post_init(self, __context):
        """Set default paths based on root_dir."""
        if self.data_dir is None:
            self.data_dir = self.root_dir / "data"

        if self.coco_dir is None:
            self.coco_dir = self.data_dir / "coco2017"
        
        if self.cache_dir is None:
            self.cache_dir = self.coco_dir / "cache"
        
        if self.coco_annotations_dir is None:
            self.coco_annotations_dir = self.coco_dir / "annotations"
        
        if self.coco_train_dir is None:
            self.coco_train_dir = self.coco_dir / "train2017"
        
        if self.coco_val_dir is None:
            self.coco_val_dir = self.coco_dir / "val2017"
        
        if self.coco_test_dir is None:
            self.coco_test_dir = self.coco_dir / "test2017"
        
        if self.output_dir is None:
            self.output_dir = self.root_dir / "outputs" / "pipeline"
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
settings = Settings()