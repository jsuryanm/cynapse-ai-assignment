from __future__ import annotations

from enum import Enum
from pydantic import BaseModel,ConfigDict,model_validator 

class ViolationReason(str,Enum):
    """Reason an image should be excluded from the curated dataset.
    Each constant maps to its corresponding stage where violation occurs.
    ex, blurry:quality, no_person:yolo detection, not_full_body: yolo-pose detection,
   etc"""

    BLURRY = "blurry" 
    NO_PERSON = "no_person" 
    NOT_FULL_BODY = "not_full_body" 
    FACE_HIDDEN = "face_hidden" 
    MINOR = "minor" 
    ADVERTISEMENT = "advertisement"  

class ValidationLabel(BaseModel):
    """Ground truth label for a single image in validation set."""
    model_config = ConfigDict(frozen=True,extra="forbid")

    crop_id: str 
    should_keep: bool 
    violation_reason: ViolationReason | None = None 
    notes: str = ""

    @model_validator(mode="after") 
    def _check_consistency(self) -> "ValidationLabel":
        """Rejects rows where should_keep and violation_reason disagree"""
        if self.should_keep and self.violation_reason is not None:
            raise ValueError(f"crop {self.crop_id!r}:should_keep=True but violation_reason="
                             f"{self.violation_reason!r} - kept images have no violation")
        
        if not self.should_keep and self.violation_reason is None:
            raise ValueError(f"crop {self.crop_id!r} should_keep=False requires a violation_reason")
        return self