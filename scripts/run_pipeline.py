# scripts/run_pipeline.py
"""End-to-end pipeline runner.

Usage:
    uv run python scripts/run_pipeline.py
    uv run python scripts/run_pipeline.py --data-dir data/raw --limit 50
    uv run python scripts/run_pipeline.py --skip-dedup
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
import yaml

from src.components.ad_filter import AdFilter
from src.components.age_filter import AgeFilter
from src.components.face_filter import FaceFilter
from src.components.person_detector import PersonDetector
from src.components.pose_filter import PoseFilter
from src.components.quality_filter import QualityFilter
from src.data.integrity import DedupReport, deduplicate_paths
from src.logger.custom_logger import configure_logger, logger
from src.pipeline.curation_pipeline import CurationPipeline
from src.services.artifact_writer import write_all_artifacts
from src.services.run_manifest import build_manifest
from src.settings.config import (CLIPAdFilterThresholds,
                                 AgeThresholds,
                                 FaceThresholds,
                                 PersonDetectionThresholds,
                                 PoseThresholds,
                                 QualityThresholds)
from src.utils.io import ImageLoader
from src.constants import IMAGES_DIR,PROJECT_ROOT_DIR,DEFAULT_THRESHOLDS_PATH,DATA_INTERIM


PROJECT_ROOT = PROJECT_ROOT_DIR
DATA_RAW = IMAGES_DIR
THRESHOLDS_PATH = DEFAULT_THRESHOLDS_PATH

app = typer.Typer(help="Curation pipeline runner.")


def _make_run_id() -> str:
    """Timestamp-based run ID. Format: 2026-05-11_115226"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _load_thresholds_dict() -> dict:
    with THRESHOLDS_PATH.open() as f:
        return yaml.safe_load(f)


@app.command()
def main(data_dir: Path = typer.Option(DATA_RAW, help="Folder of input crops."),
         limit: Optional[int] = typer.Option(None, help="If set, process only first N crops."),
         skip_dedup: bool = typer.Option(False, help="Don't dedup input paths."),
         log_level: str = typer.Option("INFO", help="Console log level.")) -> None:
    """Run the full orchestrator pipeline"""
    
    run_id = _make_run_id()
    log_file = configure_logger(log_level=log_level)
    run_folder = DATA_INTERIM / run_id

    logger.info("=" * 60)
    logger.info("Curation pipeline run: {}", run_id)
    logger.info("Data dir: {}", data_dir)
    logger.info("Output:   {}", run_folder)
    logger.info("=" * 60)

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Run the deduplication pipeline
    all_paths = sorted(p for p in data_dir.glob("*.png"))
    logger.info("Found {} candidate files", len(all_paths))

    if skip_dedup:
        unique_paths = all_paths
        dedup_report = DedupReport(
            n_input=len(all_paths), n_unique=len(all_paths), duplicate_groups={}
        )
    else:
        unique_paths, dedup_report = deduplicate_paths(all_paths)

    if limit is not None:
        unique_paths = unique_paths[:limit]
        logger.info("Limited to first {} crops", limit)

    # Load all the components
    
    thresholds_dict = _load_thresholds_dict()

    filters = [
        QualityFilter(QualityThresholds(**thresholds_dict["quality"])),
        PersonDetector(PersonDetectionThresholds(**thresholds_dict["person_detection"])),
        PoseFilter(PoseThresholds(**thresholds_dict["pose"])),
        FaceFilter(FaceThresholds(**thresholds_dict["face"])),
        AgeFilter(AgeThresholds(**thresholds_dict["age"])),
        AdFilter(CLIPAdFilterThresholds(**thresholds_dict["clip_ad_filter"])),
    ]

    stage_order = [f.name for f in filters]

    # Run Pipeline
    pipeline = CurationPipeline(filters=filters, loader=ImageLoader())
    decisions = pipeline.run(unique_paths)

    #  Build manifest + write artifacts
    finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    crop_id_to_source = {p.stem: p for p in unique_paths}
    manifest = build_manifest(run_id=run_id,
                              started_at=started_at,
                              finished_at=finished_at,
                              data_dir=str(data_dir),
                              dedup_report=dedup_report,
                              decisions=decisions,
                              stage_order=stage_order,
                              config_snapshot=thresholds_dict)

    write_all_artifacts(decisions=decisions,
                        crop_id_to_source=crop_id_to_source,
                        manifest=manifest,
                        log_file=log_file,
                        run_folder=run_folder,)

    logger.info("Done. Run artifacts at: {}", run_folder)


if __name__ == "__main__":
    app()