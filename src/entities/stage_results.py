from __future__ import annotations 

from pydantic import BaseModel,ConfigDict,Field 

class StageResult(BaseModel):
    """The verdict of one filter applied to one crop.

    Every `BaseFilter.apply(crop)` returns a `StageResult`. The pipeline
    collects these into a `FinalDecision`.

    Attributes
    ----------
    stage_name : str
        Stage identifier — must match the filter's `name` property.
        E.g. "quality", "person_detection", "pose", "face", "age", "ad_filter".
    crop_id : str
        Mirrors `Crop.crop_id` so a `StageResult` is meaningful in isolation.
    passed : bool
        True if the crop satisfied this stage's criteria.
    reason : str
        Short, human-readable explanation. For passed=True a brief
        confirmation; for passed=False the failing condition.
        E.g. "blur_variance=45.2 < threshold=80.0".
    metrics : dict[str, float]
    elapsed_ms : float
    """

    model_config = ConfigDict(frozen=True,extra="forbid")
    # frozen=True obj is immutable and extra="forbid" extra fields aren't allowed

    stage_name: str 
    crop_id: str 
    passed: bool
    reason: str 
    metrics: dict[str,float] = Field(default_factory=dict)
    elapsed_ms: float = 0.0
