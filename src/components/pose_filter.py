from __future__ import annotations 
from collections import Counter 

import torch 
from ultralytics import YOLO 

from src.components.base import BaseFilter
from src.components.quality_filter import QualityFilter
from src.components.person_detector import PersonDetector

from src.entities.crop import Crop 
from src.entities.stage_results import StageResult
from src.utils.io import ImageLoader

from src.constants import (HEAD_KPT_INDICES,
                           TORSO_KPT_INDICES,
                           LEG_KPT_INDICES,
                           IMAGES_DIR)
# COCO keypoint group indices 

from src.exceptions.custom_exceptions import ModelLoadError
from src.logger.custom_logger import logger 

from src.settings.config import (PoseThresholds,
                                 PersonDetectionThresholds,
                                 QualityThresholds)


class PoseFilter(BaseFilter):
    """Full Body Verification with YOLO pose Keypoints. 
    Rejects crops where dominant person is not visible enough
    keypoints in head,torso and leg groups."""

    name = "pose"

    def __init__(self,
                 thresholds: PoseThresholds,
                 model_path: str = "yolo11m-pose.pt",
                 device: str | None = None):
        
        self.thresholds = thresholds 
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading YOLO pose model from {model_path} to {self.device}")

        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            raise ModelLoadError(f"Failed to load YOLO model from {model_path}: {e}") from e
        
        logger.info("YOLO pose model ready")
    
    def _apply(self,crop: Crop) -> StageResult:
        # run pose inference 
        results = self.model(crop.image,verbose=False)
        result = results[0]
        boxes = result.boxes
        n_persons = len(boxes)

        if n_persons == 0:
            return self._build_result(crop_id=crop.crop_id,
                                      passed=False,
                                      reason="no_pose_detected",
                                      metrics={"n_persons_detected":0.0,
                                               "n_visible_keypoints":0.0,
                                               "n_head_visible":0.0,
                                               "n_torso_visible":0.0,
                                               "n_leg_visible":0.0,})
        
        xyxyn = boxes.xyxyn.cpu().numpy()
        areas = (xyxyn[:, 2] - xyxyn[:, 0]) * (xyxyn[:, 3] - xyxyn[:, 1])
        # computes bounding box area for every detected person (x1,y1) top left side and (x2,y2) bottom right side  
        # area tells us how big is the detected person
        largest_idx = int(areas.argmax()) # select the largest detected person

        kpts_conf = result.keypoints.conf[largest_idx].cpu().numpy() # usually out shape: (num_people,17)  
        # kpts_conf shape: (17,) we have confidences for one person
        visible_mask = kpts_conf >= self.thresholds.keypoint_confidence

        n_head_visible = int(visible_mask[list(HEAD_KPT_INDICES)].sum())
        n_torso_visible = int(visible_mask[list(TORSO_KPT_INDICES)].sum())
        n_leg_visible = int(visible_mask[list(LEG_KPT_INDICES)].sum())
        n_visible_total = int(visible_mask.sum())

        metrics: dict[str,float] = {"n_persons_detected":float(n_persons),
                                    "n_visible_keypoints":float(n_visible_total),
                                    "n_head_visible":float(n_head_visible),
                                    "n_torso_visible":float(n_torso_visible),
                                    "n_leg_visible":float(n_leg_visible)}
        
        t = self.thresholds 
        failures: list[str] = []

        if n_head_visible < t.min_head_kpts:
            failures.append(f"head={n_head_visible} < {t.min_head_kpts}")
        
        if n_torso_visible < t.min_torso_kpts:
            failures.append(f"torso={n_torso_visible} < {t.min_torso_kpts}")
        
        if n_leg_visible < t.min_leg_kpts:
            failures.append(f"legs={n_leg_visible} < {t.min_leg_kpts}")
        
        passed = len(failures) == 0
        reason = "ok" if passed else " | ".join(failures)

        return self._build_result(crop_id=crop.crop_id,
                                  passed=passed,
                                  reason=reason,
                                  metrics=metrics)

    def _build_result(self,
                      crop_id: str,
                      passed: bool,
                      reason: str,
                      metrics: dict[str,float]) -> StageResult:
        """Returns StageResults object"""
        return  StageResult(stage_name=self.name,
                            crop_id=crop_id,
                            passed=passed,
                            reason=reason,
                            metrics=metrics)
    
if __name__ == "__main__":
    quality_filter = QualityFilter(QualityThresholds())
    detector = PersonDetector(PersonDetectionThresholds())
    pose = PoseFilter(PoseThresholds())
    loader = ImageLoader()

    outcomes: Counter[str] = Counter()

    for path in sorted(IMAGES_DIR.glob("*.png")):
        crop = loader.load(path)

        if not quality_filter.apply(crop).passed: 
            outcomes["rejected_at_quality"] += 1 
            continue

        if not detector.apply(crop).passed:
            outcomes["rejected_at_detection"] += 1
            continue
        
        if not pose.apply(crop).passed:
            outcomes["rejected_at_pose"] += 1 
            continue
        
        outcomes["passed_all"] += 1 

    for outcome,count in outcomes.most_common():
        logger.info(f"{outcome}:{count}") 