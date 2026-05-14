from __future__ import annotations 

import cv2 

from src.components.base import BaseFilter
from src.entities.crop import Crop 
from src.entities.stage_results import StageResult

from src.settings.config import QualityThresholds
from src.logger.custom_logger import logger 

class QualityFilter(BaseFilter):
    """Stage 0: image quality check filter using CV techniques"""
    name = "quality"

    def __init__(self, thresholds: QualityThresholds) -> None:
        self.thresholds = thresholds
    
    def _apply(self,crop: Crop) -> StageResult:
        width = crop.width
        height = crop.height 
        aspect = height / width 
        # measures how tall an object is 

        gray = cv2.cvtColor(crop.image,cv2.COLOR_RGB2GRAY)
        brightness = float(gray.mean())
        blur_variance = float(cv2.Laplacian(gray,cv2.CV_64F).var())

        metrics: dict[str,float] = {"width":float(width),
                                    "height":float(height),
                                    "aspect":aspect,
                                    "brightness":brightness,
                                    "blur_variance":blur_variance}
        
        t = self.thresholds 
        failures: list[str] = []

        if width < t.min_width:
            failures.append(f"width: {width} < {t.min_width}")
        
        if height < t.min_height:
            failures.append(f"height: {height} < {t.min_height}")
        
        if aspect < t.min_aspect_ratio:
            failures.append(f"aspect: {aspect:.2f} < {t.min_aspect_ratio}")
        
        if brightness < t.min_brightness:
            failures.append(f"brightness: {brightness:.1f} < {t.min_brightness}")

        if brightness > t.max_brightness:
            failures.append(f"brightness: {brightness:.1f} > {t.max_brightness}")
        
        if blur_variance < t.min_blur_variance:
            failures.append(f"blur: {blur_variance:.1f} < {t.min_blur_variance}")
        
        passed = len(failures) == 0 
        reason = "ok" if passed else " | ".join(failures)
        
        return StageResult(stage_name=self.name,
                           crop_id=crop.crop_id,
                           passed=passed,
                           reason=reason,
                           metrics=metrics)

