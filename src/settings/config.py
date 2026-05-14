from __future__ import annotations 

from pathlib import Path 

import yaml 
from pydantic import BaseModel,Field,field_validator 

from src.exceptions.custom_exceptions import ConfigurationError
from src.constants import PROJECT_ROOT_DIR,DEFAULT_CONFIG_PATH,DEFAULT_THRESHOLDS_PATH

PROJECT_ROOT = PROJECT_ROOT_DIR


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
    runs_dir: Path = Path("artifacts/runs")
    labels_dir: Path = Path("data/labels")
    models_cache: Path = Path("artifacts/models")

    def resolve_all(self,root: Path = PROJECT_ROOT) -> PathsConfig:
        """Return a copy all relative paths"""
        return PathsConfig(data_dir=root / self.data_dir,
                           artifacts_dir=root / self.artifacts_dir,
                           runs_dir=root / self.runs_dir,
                           labels_dir=root / self.labels_dir,
                           models_cache=root / self.models_cache)

class ModelsConfig(BaseModel):
    yolo_detection: str = "yolo11m.pt"
    yolo_pose: str = "yolo11m-pose.pt"
    face_detector: str = "buffalo_l"
    age_estimator: str = "iitolstykh/mivolo_v2"
    clip_model: str = "openai/clip-vit-base-patch32"

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
    min_shoulder_kpts: int = 2
    min_knee_kpts: int = 2
    min_hip_kpts: int = 2 


class FaceThresholds(BaseModel):
    min_detection_confidence: float = 0.6
    min_face_area_ratio: float = 0.005
    require_all_landmarks_in_bbox: bool = True


class AgeThresholds(BaseModel):
    min_age: int = 16
    min_gender_confidence: float = 0.6


class CLIPAdFilterThresholds(BaseModel):
    similarity_margin: float = 0.0
    real_person_prompts: list[str] = ["a candid snapshot of a real person in a natural setting",
                                      "an unposed photo of a pedestrian on a city street",
                                      "a casual phone photograph of someone walking",
                                      "a real human being photographed in everyday life",
                                      "a documentary photograph of a person outdoors"]
    
    ad_prompts: list[str] = ["a mannequin on display in a clothing store window",
                            "a plastic mannequin wearing clothes for sale",
                            "a fashion advertisement photographed in a studio",
                            "a clothing store storefront with mannequin displays",
                            "a magazine fashion shoot with professional lighting",
                            "a retail product display photograph"]


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
    """Load and validate config.yaml using Pydantic.
    Ensures paths, model settings are correctly structured"""
    config_path = path or DEFAULT_CONFIG_PATH
    raw = _load_yaml(config_path)
    
    try:
        config = AppConfig(**raw)
    except Exception as e:
        raise ConfigurationError(f"Validation failed for {config_path}: {e}",config_path=path) from e 
    
    config.paths = config.paths.resolve_all(PROJECT_ROOT)
    return config

def load_thresholds(path: Path | None = None) -> Thresholds:
    """Loads threshold.yaml into Thresholds pydantic class."""
    thresholds_path = path or DEFAULT_THRESHOLDS_PATH
    raw = _load_yaml(thresholds_path)
    
    try:
        return Thresholds(**raw)
    except Exception as e:
        raise ConfigurationError(f"Validation failed for {thresholds_path}: {e}",
                                 config_path=thresholds_path) from e