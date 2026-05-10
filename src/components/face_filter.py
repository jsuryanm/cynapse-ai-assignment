from __future__ import annotations 
from collections import Counter 

import cv2 
import numpy as np 

from insightface.app import FaceAnalysis

from src.constants import IMAGES_DIR 

from src.components.base import BaseFilter
from src.components.person_detector import PersonDetector
from src.components.quality_filter import QualityFilter
from src.components.pose_filter import PoseFilter

from src.settings.config import (FaceThresholds,
                                 PersonDetectionThresholds,
                                 PoseThresholds,
                                 QualityThresholds)

from src.entities.crop import Crop 
from src.entities.stage_results import StageResult

from src.utils.io import ImageLoader 
from src.exceptions.custom_exceptions import ModelLoadError
from src.logger.custom_logger import logger

class FaceFilter(BaseFilter):
    """face detection using InsightFace, SCRFD detector"""
    name = "face"

    def __init__(self,
                 thresholds: FaceThresholds,
                 model_name: str = "buffalo_l",
                 det_size: tuple[int,int] = (640,640),
                 ctx_id: int = 0) -> None:
        self.thresholds = thresholds

        logger.info(f"Loading {model_name} via InsightFace on device {ctx_id}")
        try:
            self.app = FaceAnalysis(name=model_name,
                                    allowed_modules=["detection"],
                                    providers=["CUDAExecutionProvider","CPUExecutionProvider"])
            self.app.prepare(ctx_id=ctx_id,det_size=det_size)
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {e}") from e 
        
        logger.info("InsightFace ready")
    
    def _apply(self,crop: Crop) -> StageResult:
        img_bgr = cv2.cvtColor(crop.image,cv2.COLOR_RGB2BGR)
        faces = self.app.get(img_bgr)
        n_faces = len(faces)

        crop_area = float(crop.height * crop.width)

        if n_faces == 0:
            return self._build_result(crop_id=crop.crop_id,
                                      passed=False,
                                      reason="no_face_detected",
                                      metrics={"n_faces":0.0,
                                               "top_face_confidence":0.0,
                                               "top_face_area_ratio":0.0,
                                               "all_landmarks_in_bbox":0.0})

        areas = np.array([(f.bbox[2] - f.bbox[0])*(f.bbox[3] - f.bbox[1]) for f in faces])        
        best_idx = int(areas.argmax()) # Finds the largest face
        best = faces[best_idx]

        top_face_confidence = float(best.det_score)
        top_face_area_ratio = float(areas[best_idx]) / crop_area

        x1,y1,x2,y2  = best.bbox 
        kps_x = best.kps[:,0]
        kps_y = best.kps[:,1]
        
        all_inside = bool(np.all((kps_x >= x1) & (kps_x <= x2) & (kps_y >= y1) & (kps_y <= y2)))
        # checks if all facial landmarks are within detected face in bounding box  
        metrics: dict[str,float] = {"n_faces":float(n_faces),
                                    "top_face_confidence":top_face_confidence,
                                    "top_face_area_ratio":top_face_area_ratio,
                                    "all_landmarks_in_bbox":float(all_inside)}            
        t = self.thresholds 
        failures: list[str] = []

        if top_face_confidence < t.min_detection_confidence:
            failures.append(f"face_conf={top_face_confidence:.2f}<{t.min_detection_confidence}")
        
        if top_face_area_ratio < t.min_face_area_ratio:
            failures.append(f"face_area_ratio={top_face_area_ratio:.4f}<{t.min_face_area_ratio}")
        
        if t.require_all_landmarks_in_bbox and not all_inside:
            failures.append("landmarks_outside_bbox")
        
        passed = len(failures) == 0 
        reason = "ok" if passed else " | ".join(failures)

        if passed:
            crop.extras["face_bbox"] = (float(x1)/crop.width,
                                        float(y1)/crop.height,
                                        float(x2)/crop.width,
                                        float(y2)/crop.height)
        
        return self._build_result(crop_id=crop.crop_id,
                                  passed=passed,
                                  reason=reason,
                                  metrics=metrics)
    
    def _build_result(self,
                      crop_id: str,
                      passed: bool,
                      reason: str,
                      metrics: dict[str,float]) -> StageResult:
        return StageResult(stage_name=self.name,
                           crop_id=crop_id,
                           passed=passed,
                           reason=reason,
                           metrics=metrics)

if __name__ == "__main__":
    quality_filter =  QualityFilter(QualityThresholds())
    detector = PersonDetector(PersonDetectionThresholds())
    pose = PoseFilter(PoseThresholds())
    face = FaceFilter(FaceThresholds())

    loader = ImageLoader()

    outcomes: Counter[str] = Counter()

    for path in sorted(IMAGES_DIR.glob("*.png")):
        crop = loader.load(path)

        if not quality_filter.apply(crop).passed:
            outcomes['rejected_at_quality'] += 1 
            continue 

        if not detector.apply(crop).passed:
            outcomes['rejected_at_detector'] += 1 
            continue

        if not pose.apply(crop).passed:
            outcomes['rejected_at_pose'] += 1 
            continue 

        if not face.apply(crop).passed:
            outcomes['rejected_at_face'] += 1 
            continue

        outcomes['passed_all'] += 1 
    
    for outcome,count in outcomes.most_common():
        logger.info(f"{outcome}:{count}")