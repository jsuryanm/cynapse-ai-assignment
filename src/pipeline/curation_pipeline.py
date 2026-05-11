from __future__ import annotations 

import time  
from pathlib import Path 

from src.components.base import BaseFilter
from src.entities.crop import Crop 
from src.entities.final_decision import FinalDecision
from src.entities.stage_results import StageResult
from src.exceptions.custom_exceptions import StageExecutionError
from src.logger.custom_logger import logger 
from src.utils.io import ImageLoader

class CurationPipeline:
    """Runs the dataset curation pipeline over image paths. 
    Component filters are passed through list. Returns list[FinalDecision]"""

    def __init__(self,filters: list[BaseFilter],loader: ImageLoader):
        if not filters:
            raise ValueError("Pipeline needs atleast one filter")
        
        self.filters = filters
        self.loader = loader
        logger.info(f"Pipeline configured with {len(filters)} stages:{[f.name for f in filters]}")

    def run(self,paths: list[Path]) -> list[FinalDecision]:
        """Apply the component pipeline to every image.
        Returns FinalDecision object per image"""

        logger.info(f"Running pipeline: processing {len(paths)} images")
        decisions: list[FinalDecision] = []

        for i,path in enumerate(paths,start=1):
            decision = self._process_one(path)
            decisions.append(decision)

            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(paths)}")
            
        n_kept = sum(1 for d in decision if d.kept)
        logger.info(f"Pipeline run complete: {n_kept} kept, {len(decisions) - n_kept} rejected")
        return decisions 
    
    def _process_one(self,path: Path) -> FinalDecision:
        """Run pipeline on single image"""
        start = time.perf_counter()

        try:
            image = self.loader.load(path)
        except Exception as e:
            logger.error(f"Failed to load {path.name}: {e}")
            return FinalDecision(crop_id=path.stem,
                                 source_path=str(path),
                                 kept=False,
                                 rejected_at_stage="load",
                                 stage_results=[],
                                 total_elapsed_ms=0.0)

        results: list[StageResult] = []
        rejected_at: str | None = None 

        for filter_ in self.filters:
            try:
                result = filter_.apply(image)
            except StageExecutionError as e:
                logger.error(f"Stage {filter_.name} error occured on {image.crop_id}:{e}") 
                rejected_at = filter_.name
                break 
            
            results.append(result)
            if not result.passed:
                rejected_at = filter_.name 
                break 
        
        total_ms = (time.perf_counter() - start) * 1000 
        return FinalDecision(crop_id=image.crop_id,
                             source_path=str(path),
                             kept=(rejected_at is None),
                             stage_results=results,
                             total_elapsed_ms=total_ms)