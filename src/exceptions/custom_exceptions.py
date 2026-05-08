from __future__ import annotations 
from pathlib import Path 

class PipelineError(Exception):
    """Base class for all curation-pipeline errors"""


class ConfigurationError(PipelineError):
    """Raised when config or threshold YAML file is missing, malformed or fails validation"""

    def __init__(self, message: str, config_path: Path | None) -> None:
        super().__init__(message)
        self.config_path = config_path

class ModelLoadError(PipelineError):
    """Raised when a model checkpoint can't be loaded"""

    def __init__(self, message: str, model_name: str | None=None) -> None:
        super().__init__(message)
        self.model_name = model_name 

class ImageProcessingError(PipelineError):
    """Raised when the image file is unreadable or its bytes are truncated.
    This occurs when Image.verify fails or cv2.imread() fails"""

class ValidationError(PipelineError):
    """Raised when validation labels are missing, malformed or inconsistent."""

class StageExecutionError(PipelineError):
    """Raised when stage's apply() method crashes"""

    def __init__(self, messages: str, stage_name: str) -> None:
        super().__init__(messages)
        self.stage_name = stage_name