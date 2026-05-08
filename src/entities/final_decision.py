from __future__ import annotations 

from typing import Any 

from pydantic import BaseModel,ConfigDict,Field 

from src.entities.stage_results import StageResult

class FinalDecision(BaseModel):
    """Pipeline's final verdict on a single crop.
    Aggregates all StageResults for one crop and provides downstream analysis
    args:
    crop_id: image 
    source_path: original str path of image
    kept: True if it passes every stage
    rejected: name of stage where image is rejected
    stage_results: All stages in execution order
    total_elapsed_ms: Sum of StagedResults execution time"""

    model_config = ConfigDict(frozen=True,extra="forbid")

    crop_id: str
    source_path: str
    kept: bool
    rejected_at_stage: str | None
    stage_results: list[StageResult] = Field(default_factory=list)
    total_elapsed_ms: float = 0.0

    def to_flat_record(self) -> dict[str,Any]:
        """Flatten into a single row suitable for Parquet/CSV
        Stage level details collapse into prefixed columns."""
        
        record: dict[str,Any] = {"crop_id":self.crop_id,
                                 "source_path":self.source_path,
                                 "kept":self.kept,
                                 "rejected_at_stage":self.rejected_at_stage,
                                 "total_elapsed_ms":self.total_elapsed_ms}
        
        for result in self.stage_results:
            prefix = result.stage_name
            record[f"{prefix}_passed"] = result.passed
            record[f"{prefix}_reason"] = result.reason
            record[f"{prefix}_elapsed_ms"] = result.elapsed_ms

            for metric_name,metric_value in result.metrics.items():
                record[f"{prefix}_{metric_name}"] = metric_value