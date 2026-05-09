from __future__ import annotations 
from pathlib import Path

import cv2 

from src.components.base import BaseFilter
from src.entities.crop import Crop 
from src.entities.stage_results import StageResult
from src.settings.config import QualityThresholds
from src.utils.io import ImageLoader
from src.constants import IMAGES_DIR
from src.logger.custom_logger import logger 

class QualityFilter(BaseFilter):
    """Stage 0: coarse image quality filter using CV techniques"""
    name = "quality"

    def __init__(self, thresholds: QualityThresholds) -> None:
        self.thresholds = thresholds
    
    def _apply(self,crop: Crop) -> StageResult:
        width = crop.width
        height = crop.height 
        aspect = width / height 

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
            failures.append(f"width: {height} < {t.min_height}")
        
        if aspect < t.min_aspect_ratio:
            failures.append(f"width: {aspect:.2f} < {t.min_aspect_ratio}")
        
        if brightness < t.min_brightness:
            failures.append(f"width: {brightness:.1f} < {t.min_brightness}")

        if brightness > t.max_brightness:
            failures.append(f"{brightness:.1f} > {t.max_brightness}")
        
        if blur_variance < t.min_blur_variance:
            failures.append(f"blur: {blur_variance:.1f} < {t.min_blur_variance}")
        
        passed = len(failures) == 0 
        reason = "ok" if passed else " | ".join(failures)
        
        return StageResult(stage_name=self.name,
                           crop_id=crop.crop_id,
                           passed=passed,
                           reason=reason,
                           metrics=metrics)

if __name__ == "__main__":
    qf = QualityFilter()
    loader = ImageLoader()
    n_passed,n_failed = 0,0 

    for path in sorted(IMAGES_DIR.glob("*.png")):
        crop = loader.load(path)
        result = qf.apply(crop)

        if result.passed:
            n_passed += 1
        else:
            n_failed += 1 
    
    print()

    