from __future__ import annotations

from enum import Enum
from pydantic import BaseModel,ConfigDict,model_validator 

class ViolationReason(str,Enum):
    """Reason am image should be excluded from the curated dataset."""
    BLURRY = "blurry" # quality filter stage
    NO_PERSON = "no_person" # yolo person detection
    NOT_FULL_BODY = "not_full_body" # yolo pose detection
    FACE_HIDDEN = "face_hidden" # face stage
    MINOR = "minor" # age stage
    ADVERTISEMENT = "advertisement" # ad filter stage 

class ValidationLabel(BaseModel):
    """Ground truth label for a single image in validation set."""
    model_config = ConfigDict(frozen=True,extra="forbid")

    crop_id: str 
    should_keep: bool 
    violation_reason: ViolationReason | None = None 
    notes: str = ""

    @model_validator(mode="after") # custom validation logic this runs after validating all fields 
    def _check_consistency(self) -> "ValidationLabel":
        """Rejects rows where should_keep and violation_reason disagree"""
        if self.should_keep and self.violation_reason is not None:
            raise ValueError(f"crop {self.crop_id!r}:should_keep=True but violation_reason="
                             f"{self.violation_reason!r} - kept images have no violation")
        
        if not self.should_keep and self.violation_reason is None:
            raise ValueError(f"crop {self.crop_id!r} should_keep=False requires a violation_reason")
        return self