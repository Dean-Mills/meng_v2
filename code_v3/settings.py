from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    root_dir: Path = Field(default=Path(__file__).resolve().parent.parent.parent)
    data_dir: Optional[Path] = None
    virtual_data_dir: Optional[Path] = None
    coco_dir: Optional[Path] = None
    coco_annotations_dir: Optional[Path] = None
    coco_train_dir: Optional[Path] = None
    coco_val_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    target_image_size: int = 512

    class Config:
        env_prefix = "PROJECT_"
        arbitrary_types_allowed = True

    def model_post_init(self, __context):
        r = self.root_dir
        self.data_dir           = self.data_dir           or (r / "data")
        self.virtual_data_dir   = self.virtual_data_dir   or (self.data_dir / "virtual")
        self.coco_dir           = self.coco_dir           or (self.data_dir / "coco2017")
        self.coco_annotations_dir = self.coco_annotations_dir or (self.coco_dir / "annotations")
        self.coco_train_dir     = self.coco_train_dir     or (self.coco_dir / "train2017")
        self.coco_val_dir       = self.coco_val_dir       or (self.coco_dir / "val2017")
        self.output_dir         = self.output_dir         or (r / "outputs" / "pipeline")
        self.output_dir.mkdir(parents=True, exist_ok=True)

settings = Settings()