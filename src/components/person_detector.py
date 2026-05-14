from __future__ import annotations

from ultralytics import YOLO

from src.components.base import BaseFilter 
from src.settings.config import PersonDetectionThresholds


from src.entities.crop import Crop
from src.entities.stage_results import StageResult

from src.constants import DEVICE 

from src.exceptions.custom_exceptions import ModelLoadError
from src.logger.custom_logger import logger 

class PersonDetector(BaseFilter):
    """Detects the person in image with YOLOv11m
    Rejects crops when:
    1. No detections clears min_confidence
    2. The best detections bbox covers less than min_bbox_area_ratio
    3. More than max_persons valid detections (noisy multi-person) 
    """

    name = "person_detection"
    PERSON_CLASS_ID = 0 

    def __init__(self,
                 thresholds: PersonDetectionThresholds,
                 model_path: str = "yolo11m.pt",
                 device: str | None = None) -> None:
        
        self.thresholds = thresholds
        self.device = device or DEVICE
        logger.info(f"Loading YOLO weights from {model_path} on device {self.device}")

        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
        
        except Exception as e:
            raise ModelLoadError(f"Failed to load YOLO model from {model_path}:{e}") from e
        
        logger.info("YOLO model ready")
    
    def _apply(self,crop: Crop) -> StageResult:
        results = self.model(crop.image,
                             conf=self.thresholds.min_confidence,
                             classes=[self.PERSON_CLASS_ID],
                             verbose=False)
        
        boxes = results[0].boxes
        n_raw = len(boxes)

        if n_raw > 0:
            xyxyn = boxes.xyxyn.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            areas = (xyxyn[:,2] - xyxyn[:,0]) * (xyxyn[:,3] - xyxyn[:,1]) 
            # we use xyxyn instead of xyxy since its more better for area filtering (easier to communicate it)
            # area ratio per detection (x2-x1)*(y2-y1)
            # ex detected person occupies 50% of image 
        else:
            xyxyn = confs = areas = None
        
        # apply area-ratio filter on top of YOLO confs filter 
        if n_raw > 0:
            valid_mask = areas >= self.thresholds.min_bbox_area_ratio
            n_valid = int(valid_mask.sum())
        else:
            n_valid = 0 
        
        # pick the best detection largest among valid 
        if n_valid > 0:
            valid_areas = areas[valid_mask]
            valid_confs = confs[valid_mask]
            valid_boxes = xyxyn[valid_mask]
            best_idx = int(valid_areas.argmax())
            top_confidence = float(valid_confs[best_idx])
            top_area = float(valid_areas[best_idx])
            best_bbox = tuple(float(v) for v in valid_boxes[best_idx])
        
        else:
            top_confidence = 0.0 
            top_area = 0.0 
            best_bbox = None 
        
        metrics: dict[str,float] = {"n_detections_raw":float(n_raw),
                                    "n_detections_valid":float(n_valid),
                                    "top_confidence":top_confidence,
                                    "top_bbox_area_ratio":top_area}
        
        # decision
        failures: list[str] = []

        if n_valid == 0:
            if n_raw == 0:
                failures.append("no_person_detected")
            else:
                failures.append(f"top_bbox_area={areas.max():.2f}"
                                f"<{self.thresholds.min_bbox_area_ratio}")
        
        elif n_valid > self.thresholds.max_persons:
            failures.append(f"n_persons={n_valid} > {self.thresholds.max_persons}")
        
        passed = len(failures) == 0 
        reason = "ok" if passed else " | ".join(failures)

        if passed and best_bbox is not None:
            crop.extras["person_bbox"] = best_bbox
        
        return StageResult(stage_name=self.name,
                           crop_id=crop.crop_id,
                           passed=passed,
                           reason=reason,
                           metrics=metrics)

