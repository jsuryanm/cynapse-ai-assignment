from __future__ import annotations 

from pydantic import BaseModel,ConfigDict,Field 

class StageResult(BaseModel):
    """Stores the execution result of components 
    Attributes
    ----------
    stage_name : str, which stage (component) we are currently running  
    (e.g, quality check,yolo person detection,etc)
    crop_id : str, image name
    passed : bool
        True if the image satisfied this stage's criteria.
    reason : str
        Short, human-readable explanation. 
        If img passes stage then reason=ok,
        else reason for failing is given from component.
    metrics : dict[str, float] E.g. {"blur_variance": 45.2, "brightness": 120.4, ...}.
    elapsed_ms : float
    """

    model_config = ConfigDict(frozen=True,extra="forbid")

    stage_name: str 
    crop_id: str 
    passed: bool
    reason: str 
    metrics: dict[str,float] = Field(default_factory=dict)
    elapsed_ms: float = 0.0
