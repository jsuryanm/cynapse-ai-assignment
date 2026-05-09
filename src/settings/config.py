from __future__ import annotations 

from pathlib import Path 

import yaml 
from pydantic import BaseModel,Field,field_validator 

from src.exceptions.custom_exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT / "config" / "thresholds.yaml"

class RuntimeConfig(BaseModel):
    device: str = "cuda"
    batch_size: int = 16 
    num_workers: int = 2 

    @field_validator("device") # used to validate or modify the field before object is created
    @classmethod 
    def _device_must_be_cuda_or_cpu(cls,v: str) -> str:
        if v not in {"cpu","cuda"}:
            raise ValueError(f"device must be cuda or cpu, got {v}")
        return v 

class PathsConfig(BaseModel):
    """All paths are stored to the project root"""
    data_dir: Path = Path("data/raw")
    artifacts_dir: Path = Path("artifacts")
    labels_dir: Path = Path("data/labels")
    models_cache: Path = Path("artifacts/models")

    def resolve_all(self,root: Path = PROJECT_ROOT) -> PathsConfig:
        """Return a copy all relative paths"""
        return PathsConfig(data_dir=root / self.data_dir,
                           artifacts_dir=root / self.artifacts_dir,
                           labels_dir=root / self.labels_dir,
                           models_cache=root / self.models_cache)

# config
class ModelsConfig(BaseModel):
    yolo_detection: str = "yolo11m.pt"
    yolo_pose: str = "yolo11m-pose.pt"
    face_detector: str = "buffalo_l"
    age_estimator: str = "mivolo_d1"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

class PipelineRunConfig(BaseModel):
    save_rejected: bool = True 
    save_visualizations: bool = True
    decisions_filename: str = "decisions.parquet"
    manifest_filename: str = "manifest.json"

class AppConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    pipeline: PipelineRunConfig = Field(default_factory=PipelineRunConfig)

# thresholds
class QualityThresholds(BaseModel):
    min_aspect_ratio: float = 1.3
    min_width: int = 80
    min_height: int = 150
    min_brightness: float = 15.0
    max_brightness: float = 240.0
    min_blur_variance: float = 10.0

class PersonDetectionThresholds(BaseModel):
    min_confidence: float = 0.5
    min_bbox_area_ratio: float = 0.4
    max_persons: int = 5


class PoseThresholds(BaseModel):
    keypoint_confidence: float = 0.5
    min_head_kpts: int = 1
    min_torso_kpts: int = 4
    min_leg_kpts: int = 4


class FaceThresholds(BaseModel):
    min_detection_confidence: float = 0.6
    min_landmarks_frontal: int = 4
    min_landmarks_profile: int = 2
    min_face_area_ratio: float = 0.005


class AgeThresholds(BaseModel):
    min_age: int = 16
    age_confidence: float = 0.5


class CLIPAdFilterThresholds(BaseModel):
    similarity_margin: float = 0.05
    real_person_prompts: list[str]
    ad_prompts: list[str]


class Thresholds(BaseModel):
    """All tunable numerical knobs across the pipeline."""

    quality: QualityThresholds = Field(default_factory=QualityThresholds)
    person_detection: PersonDetectionThresholds = Field(
        default_factory=PersonDetectionThresholds
    )
    pose: PoseThresholds = Field(default_factory=PoseThresholds)
    face: FaceThresholds = Field(default_factory=FaceThresholds)
    age: AgeThresholds = Field(default_factory=AgeThresholds)
    clip_ad_filter: CLIPAdFilterThresholds

def _load_yaml(path: Path) -> dict:
    """Read YAML file into a dict"""
    if not path.exists():
        raise ConfigurationError(f"Config file not found: {path}",config_path=path)
    
    try:
        with path.open("r",encoding="utf-8") as f:
            data = yaml.safe_load(f)
    
    except yaml.YAMLError as e:
        raise ConfigurationError(f"YAML syntax error in {path}: {e}",config_path=path) from e 
    
    if not isinstance(data,dict):
        raise  ConfigurationError(f"Top-level YAML in {path} must be mapping (got {type(data).__name__})",
                                  config_path=path)
    return data

def load_app_config(path: Path | None = None) -> AppConfig:
    """Load and validate config.yaml"""
    config_path = path or DEFAULT_CONFIG_PATH
    raw = _load_yaml(config_path)
    
    try:
        config = AppConfig(**raw)
    except Exception as e:
        raise ConfigurationError(f"Validation failed for {config_path}: {e}",config_path=path) from e 
    
    config.paths = config.paths.resolve_all(PROJECT_ROOT)
    return config

def load_thresholds(path: Path | None = None) -> Thresholds:
    thresholds_path = path or DEFAULT_THRESHOLDS_PATH
    raw = _load_yaml(thresholds_path)
    
    try:
        return Thresholds(**raw)
    except Exception as e:
        raise ConfigurationError(f"Validation failed for {thresholds_path}: {e}",
                                 config_path=thresholds_path) from e
    
# if __name__ == "__main__":
#     config = load_app_config()
#     print(config)
#     thresholds = load_thresholds()
#     print(thresholds) 