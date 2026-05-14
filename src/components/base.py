from __future__ import annotations

import time 
from abc import ABC,abstractmethod

from src.entities.crop import Crop 
from src.entities.stage_results import StageResult
from src.exceptions.custom_exceptions import StageExecutionError
from src.logger.custom_logger import logger

class BaseFilter(ABC):
    """Subclasses implement _appy(crop) with their domain logic.
    apply(crop) wrapper handles timing, logging and exceptions"""

    name: str = ""

    @abstractmethod
    def _apply(self,crop: Crop) -> StageResult:
        """Run the stage's logic on a single crop
        
        Subclasses must return:
        - return a `StageResult` with `stage_name == self.name`
        - populate `metrics` with every value used in the decision
        - never raise on legitimate "rejected" cases — those are normal
        outcomes and must come back as `passed=False`
        - only raise for unexpected programming errors (which the wrapper
        converts to `StageExecutionError`)"""

    def apply(self,crop: Crop) -> StageResult:
        """Wraps _apply with timing and error handling.
        Subclasses don't override this. They only override _apply"""
        start = time.perf_counter()

        try:
            result = self._apply(crop)
        
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0 
            logger.exception(f"Stage {self.name} crashed on crop_id:{crop.crop_id} after {elapsed_ms:.1f}ms")
            raise StageExecutionError(f"Stage {self.name} failed for crop_id: {crop.crop_id}: {e}",
                                      stage_name=self.name) from e 
        
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if result.stage_name != self.name :
            raise StageExecutionError(f"Stage {self.name} returned StageResult with mismatched"
                                      f"stage_name: {result.stage_name!r}",
                                      stage_name=self.name)
        
        result = result.model_copy(update={"elapsed_ms":elapsed_ms})
        logger.info(f"Stage {self.name} | crop_id: {crop.crop_id} | passed: {result.passed} | {elapsed_ms:.1f} ms | {result.reason}")
        return result