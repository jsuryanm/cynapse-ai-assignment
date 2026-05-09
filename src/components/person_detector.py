from __future__ import annotations

import torch
from ultralytics import YOLO

from src.components.base import BaseFilter 
from src.entities.crop import Crop
from src.entities.stage_results import StageResult
from src.exceptions.custom_exceptions import ModelLoadError
from src.logger.custom_logger import logger 
from src.settings.config import PersonDetectionThresholds

