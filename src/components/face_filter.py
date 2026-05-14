from __future__ import annotations 

import cv2 
import numpy as np 

from insightface.app import FaceAnalysis


from src.components.base import BaseFilter
from src.settings.config import FaceThresholds
from src.entities.crop import Crop 
from src.entities.stage_results import StageResult
from src.exceptions.custom_exceptions import ModelLoadError
from src.logger.custom_logger import logger

class FaceFilter(BaseFilter):
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
        """Detects visible faces inside the detected person region using InsightFace SCRFD detector.
        Ensures the face is large, confident, and geometrically consistent so that
        heavily occluded, cropped, or invalid faces are filtered out early."""
        
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

        areas = np.array([(f.bbox[2] - f.bbox[0])*(f.bbox[3] - f.bbox[1]) 
                          for f in faces])   

        person_bbox_norm =  crop.extras.get("person_bbox")
        
        if person_bbox_norm is None:
            valid_indices = list(range(len(faces)))
        
        else:
            px1 = person_bbox_norm[0] * crop.width 
            py1 = person_bbox_norm[1] * crop.height
            px2 = person_bbox_norm[2] * crop.width
            py2 = person_bbox_norm[3] * crop.height

            valid_indices = []

            for i,f in enumerate(faces):
                cx = (f.bbox[0] + f.bbox[2]) / 2
                cy = (f.bbox[1] + f.bbox[3]) / 2 

                if px1 <= cx <= px2 and py1 <= cy <= py2:
                    valid_indices.append(i)
        
        if not valid_indices:
            return self._build_result(crop_id=crop.crop_id,
                                      passed=False,
                                      reason="no_face_inside_person_bbox",
                                      metrics={"n_faces":float(n_faces),
                                               "top_face_confidence":0.0,
                                               "top_face_area_ratio":0.0,
                                               "all_landmarks_in_bbox":0.0,})

        valid_areas = areas[valid_indices]
        local_best =  int(valid_areas.argmax())
        best_idx = valid_indices[local_best]
        best = faces[best_idx] 

        top_face_confidence = float(best.det_score)
        top_face_area_ratio = float(areas[best_idx]) / crop_area

        x1,y1,x2,y2  = best.bbox 
        kps_x = best.kps[:,0]
        kps_y = best.kps[:,1]
        
        all_inside = bool(np.all((kps_x >= x1) & (kps_x <= x2) & (kps_y >= y1) & (kps_y <= y2)))
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