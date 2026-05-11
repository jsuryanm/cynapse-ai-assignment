from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from src.components.ad_filter import AdFilter
from src.components.age_filter import AgeFilter
from src.components.face_filter import FaceFilter
from src.components.person_detector import PersonDetector
from src.components.pose_filter import PoseFilter
from src.components.quality_filter import QualityFilter
from src.components.base import BaseFilter

from src.data.integrity import DedupReport, deduplicate_paths

from src.logger.custom_logger import configure_logger, logger

from src.pipeline.curation_pipeline import CurationPipeline

from src.services.artifact_writer import write_all_artifacts
from src.services.run_manifest import build_environment, build_manifest

from src.settings.config import (AppConfig,
                                 Thresholds,
                                 load_app_config,
                                 load_thresholds)
from src.utils.io import ImageLoader


app = typer.Typer(help="Curation pipeline runner.")


def _make_run_id() -> str:
    """Timestamp-based run ID. Format: 2026-05-11_142525"""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _build_filters(thresholds: Thresholds) -> list[BaseFilter]:
    """Construct the cascade in canonical order.

    Order matters: cheap OpenCV checks run first so we never spend
    YOLO/CLIP compute on crops we were going to drop anyway.
    """
    return [
        QualityFilter(thresholds.quality),
        PersonDetector(thresholds.person_detection),
        PoseFilter(thresholds.pose),
        FaceFilter(thresholds.face),
        AgeFilter(thresholds.age),
        AdFilter(thresholds.clip_ad_filter),
    ]


@app.command()
def main(
    data_dir: Optional[Path] = typer.Option(
        None,
        help="Override paths.data_dir from config.yaml.",
    ),
    limit: Optional[int] = typer.Option(None, help="If set, process only first N crops."),
    skip_dedup: bool = typer.Option(False, help="Don't dedup input paths."),
    log_level: str = typer.Option("INFO", help="Console log level."),
) -> None:
    """Run the full curation cascade."""
    # ---------- Load typed configuration ----------
    app_config: AppConfig = load_app_config()
    thresholds: Thresholds = load_thresholds()

    # CLI flag overrides config.yaml when provided.
    effective_data_dir = data_dir.resolve() if data_dir else app_config.paths.data_dir

    # ---------- Setup ----------
    run_id = _make_run_id()
    log_file = configure_logger(log_level=log_level)
    run_folder = app_config.paths.runs_dir / run_id

    logger.info("=" * 60)
    logger.info("Curation pipeline run: {}", run_id)
    logger.info("Data dir:   {}", effective_data_dir)
    logger.info("Run folder: {}", run_folder)
    logger.info("=" * 60)

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---------- Enumerate + dedup ----------
    all_paths = sorted(effective_data_dir.glob("*.png"))
    logger.info("Found {} candidate files", len(all_paths))

    if skip_dedup:
        unique_paths = all_paths
        dedup_report = DedupReport(
            n_input=len(all_paths),
            n_unique=len(all_paths),
            duplicate_groups={},
        )
    else:
        unique_paths, dedup_report = deduplicate_paths(all_paths)

    if limit is not None:
        unique_paths = unique_paths[:limit]
        logger.info("Limited to first {} crops", limit)

    # ---------- Build filters ----------
    filters = _build_filters(thresholds)
    stage_order = [f.name for f in filters]

    # ---------- Run cascade ----------
    pipeline = CurationPipeline(filters=filters, loader=ImageLoader())
    decisions = pipeline.run(unique_paths)

    # ---------- Build manifest + write artifacts ----------
    finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    crop_id_to_source = {p.stem: p for p in unique_paths}

    manifest = build_manifest(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        dedup_report=dedup_report,
        decisions=decisions,
        stage_order=stage_order,
    )
    config_snapshot = thresholds.model_dump(mode="json")
    environment = build_environment(data_dir=str(effective_data_dir))

    write_all_artifacts(
        decisions=decisions,
        crop_id_to_source=crop_id_to_source,
        manifest=manifest,
        config_snapshot=config_snapshot,
        environment=environment,
        dedup_report=dedup_report,
        log_file=log_file,
        run_folder=run_folder,
    )


    logger.info("Done. Run artifacts at: {}", run_folder)


if __name__ == "__main__":
    app()