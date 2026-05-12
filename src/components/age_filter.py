from __future__ import annotations
from collections import Counter

import cv2 
import torch 
from transformers import (AutoConfig,
                          AutoImageProcessor,
                          AutoModelForImageClassification)

from src.components.base import BaseFilter
from src.components.quality_filter import QualityFilter
from src.components.person_detector import PersonDetector
from src.components.pose_filter import PoseFilter
from src.components.face_filter import FaceFilter

from src.entities.crop import Crop 
from src.entities.stage_results import StageResult
from src.utils.io import ImageLoader

from src.settings.config import (AgeThresholds,
                                 QualityThresholds,
                                 PoseThresholds,
                                 FaceThresholds,
                                 PersonDetectionThresholds)

from src.exceptions.custom_exceptions import ModelLoadError
from src.logger.custom_logger import logger
from src.constants import IMAGES_DIR,AGE_MODEL_ID,DEVICE

class AgeFilter(BaseFilter):
    """Predicting age and gender with MiVOLO v2, reject minors"""
    name = "age"
    
    def __init__(self,
                 thresholds: AgeThresholds,
                 device: str | None = None) -> None:
        self.thresholds = thresholds
        self.device = device or DEVICE 
        logger.info(f"Loading MiVOLOv2 {AGE_MODEL_ID} on {self.device}")

        try:
            self.config = AutoConfig.from_pretrained(AGE_MODEL_ID,
                                                     trust_remote_code=True)
            
            self.processor = AutoImageProcessor.from_pretrained(AGE_MODEL_ID,
                                                                trust_remote_code=True)
            
            self.model = AutoModelForImageClassification.from_pretrained(AGE_MODEL_ID,
                                                                         trust_remote_code=True,
                                                                         dtype=torch.float16).to(self.device)
            self.model.eval()
        
        except Exception as e:
            raise ModelLoadError(f"Failed to load MiVOLOv2 from {AGE_MODEL_ID} :{e}") from e
        
        logger.info(f"MiVOLOv2 ready in fp16,device: {self.device}")

    def _apply(self,crop: Crop) -> StageResult:
        face_bbox = crop.extras.get("face_bbox")
        person_bbox = crop.extras.get("person_bbox")

        if face_bbox is None or person_bbox is None:
            return self._build_result(crop_id=crop.crop_id,
                                      passed=False,
                                      reason="missing_upstream_bbox",
                                      metrics=self._empty_metrics())
        
        H,W = crop.height,crop.width
        face_x1,face_y1,face_x2,face_y2 = self._denormalize(face_bbox,W,H)
        body_x1,body_y1,body_x2,body_y2 = self._denormalize(person_bbox,W,H)

        # Clamp coordinates to image bounds
        face_x1 = max(0, min(face_x1, W - 1))
        face_x2 = max(0, min(face_x2, W))

        face_y1 = max(0, min(face_y1, H - 1))
        face_y2 = max(0, min(face_y2, H))

        body_x1 = max(0, min(body_x1, W - 1))
        body_x2 = max(0, min(body_x2, W))

        body_y1 = max(0, min(body_y1, H - 1))
        body_y2 = max(0, min(body_y2, H))

        # Validate crop geometry
        if face_x2 <= face_x1 or face_y2 <= face_y1:
            return self._build_result(crop_id=crop.crop_id,
                                      passed=False,
                                      reason="invalid_face_crop",
                                      metrics=self._empty_metrics())

        if body_x2 <= body_x1 or body_y2 <= body_y1:
            return self._build_result(crop_id=crop.crop_id,
                                      passed=False,
                                      reason="invalid_body_crop",
                                      metrics=self._empty_metrics())

        face_rgb = crop.image[face_y1:face_y2, face_x1:face_x2]
        body_rgb = crop.image[body_y1:body_y2, body_x1:body_x2]

        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        body_bgr = cv2.cvtColor(body_rgb, cv2.COLOR_RGB2BGR)

        faces_input = self.processor(images=[face_bgr])["pixel_values"]
        body_input = self.processor(images=[body_bgr])["pixel_values"]

        faces_input = faces_input.to(dtype=self.model.dtype, device=self.device)
        body_input = body_input.to(dtype=self.model.dtype, device=self.device)

        with torch.no_grad():
            output = self.model(faces_input=faces_input,
                                body_input=body_input,
                                return_dict=True)
        
        predicted_age = float(output.age_output[0].item())
        gender_idx = int(output.gender_class_idx[0].item())
        gender_confidence = float(output.gender_probs[0].item())
        gender_label = self.config.gender_id2label[gender_idx]

        metrics: dict[str, float] = {"predicted_age": predicted_age,
                                     "gender_idx": float(gender_idx),
                                     "gender_confidence": gender_confidence,
                                     "face_crop_height": float(face_rgb.shape[0]),
                                     "face_crop_width": float(face_rgb.shape[1]),
                                     "body_crop_height": float(body_rgb.shape[0]),
                                     "body_crop_width": float(body_rgb.shape[1])}

        t = self.thresholds
        failures: list[str] = []

        if predicted_age < t.min_age:
            failures.append(f"age={predicted_age:.1f}<{t.min_age} ({gender_label})")
        
        if gender_confidence < t.min_gender_confidence:
            failures.append(f"gender_conf={gender_confidence:.2f}<{t.min_gender_confidence}")

        passed = len(failures) == 0
        reason = "ok" if passed else " | ".join(failures)

        return self._build_result(crop_id=crop.crop_id,
                                  passed=passed,
                                  reason=reason,
                                  metrics=metrics)

    @staticmethod
    def _denormalize(bbox_norm: tuple[float,float,float,float],
                     W: int,H: int) -> tuple[int,int,int,int]:
        """Converts normalized [0,1] bbox to pixel coordinates"""
        x1,y1,x2,y2 = bbox_norm
        return (int(x1*W),int(y1*H),int(x2*W),int(y2*H))
    
    @staticmethod
    def _empty_metrics() -> dict[str,float]:
        return {"predicted_age":0.0,
                "gender_idx":0.0,
                "gender_confidence":0.0,
                "face_crop_height":0.0,
                "face_crop_width":0.0,
                "body_crop_height":0.0,
                "body_crop_width":0.0,}
    
    def _build_result(self,
                      crop_id: str,
                      passed: bool,
                      reason: str,
                      metrics: dict[str, float]) -> StageResult:
        
        return StageResult(stage_name=self.name,
                           crop_id=crop_id,
                           passed=passed,
                           reason=reason,
                           metrics=metrics)

# if __name__ == "__main__":
    
#     quality_filter = QualityFilter(QualityThresholds())
#     detector = PersonDetector(PersonDetectionThresholds())
#     pose = PoseFilter(PoseThresholds())
#     face = FaceFilter(FaceThresholds())
#     age = AgeFilter(AgeThresholds())
#     loader = ImageLoader()

    
#     outcomes: Counter[str] = Counter()

#     for path in sorted(IMAGES_DIR.glob("*.png")):
#         crop = loader.load(path)

#         if not quality_filter.apply(crop).passed:
#             outcomes["rejected_at_quality"] += 1
#             continue
#         if not detector.apply(crop).passed:
#             outcomes["rejected_at_detection"] += 1
#             continue
#         if not pose.apply(crop).passed:
#             outcomes["rejected_at_pose"] += 1
#             continue
#         if not face.apply(crop).passed:
#             outcomes["rejected_at_face"] += 1
#             continue
#         if not age.apply(crop).passed:
#             outcomes["rejected_at_age"] += 1
#             continue

#         outcomes["passed_all"] += 1

#     for outcome, count in outcomes.most_common():
#         logger.info(" {} : {}", outcome, count)