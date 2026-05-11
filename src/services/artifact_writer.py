from __future__ import annotations 

import json 
import random 
import shutil 

from collections import defaultdict 
from pathlib import Path 
from typing import Any 

import pandas as pd 

from src.entities.final_decision import FinalDecision
from src.logger.custom_logger import logger 

SAMPLES_PER_STAGE = 20 # how many rejected samples to keep per stage

def _ensure_folder(path: Path) -> Path:
    path.mkdir(parents=True,exist_ok=True)

def write_decisions_parquet(decisions: list[FinalDecision],path: Path) -> None:
    records = [d.to_flat_record() for d in decisions]
    df = pd.DataFrame.from_records(records)
    df.to_parquet(path,index=False)
    logger.info(f"Wrote {len(df)} rows to {path.name}")

def copy_kept_crops(decisions: list[FinalDecision],
                    crop_id_to_source: dict[str,Path],
                    kept_folder: Path) -> int:
    """Copy each kept crop in kept_folder. Returns count of copied files."""
    _ensure_folder(kept_folder)
    n = 0
    for d in decisions:
        if not d.kept:
            continue
        source = crop_id_to_source.get(d.crop_id)
        if source is None or not source.exists():
            logger.warning("Kept crop missing source path: {}", d.crop_id)
            continue
        shutil.copy2(source, kept_folder / source.name)
        n += 1
    logger.info("Copied {} kept crops to {}", n, kept_folder)
    return n

def copy_rejected_samples(decisions: list[FinalDecision],
                          crop_id_to_source: dict[str, Path],
                          samples_root: Path,
                          samples_per_stage: int = SAMPLES_PER_STAGE,
                          seed: int = 42) -> dict[str, int]:
    """For each stage, copy up to `samples_per_stage` random rejected crops
    into samples_root/rejected_at_<stage>/. Random with fixed seed for
    reproducibility."""
    _ensure_folder(samples_root)
    rng = random.Random(seed)

    # Group rejected crops by stage
    by_stage: dict[str, list[FinalDecision]] = defaultdict(list)
    for d in decisions:
        if not d.kept and d.rejected_at_stage is not None:
            by_stage[d.rejected_at_stage].append(d)

    counts: dict[str, int] = {}
    for stage, items in by_stage.items():
        folder = _ensure_folder(samples_root / f"rejected_at_{stage}")
        sample = rng.sample(items, min(samples_per_stage, len(items)))
        copied = 0
        for d in sample:
            source = crop_id_to_source.get(d.crop_id)
            if source is None or not source.exists():
                continue
            shutil.copy2(source, folder / source.name)
            copied += 1
        counts[stage] = copied
        logger.info("Sampled {} rejected crops for stage '{}'", copied, stage)
    return counts

def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Serialize the manifest dict to JSON."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Wrote manifest to {}", path.name)


def copy_log_file(log_file: Path, run_folder: Path) -> None:
    """Copy the active log file into the run folder for archival.
    Best-effort — if the log file isn't where expected, we skip."""
    if log_file is None or not log_file.exists():
        logger.warning("Log file not found at {}, skipping archive", log_file)
        return
    shutil.copy2(log_file, run_folder / "run.log")
    logger.info("Archived log file to {}/run.log", run_folder.name)


def write_all_artifacts(
    *,
    decisions: list[FinalDecision],
    crop_id_to_source: dict[str, Path],
    manifest: dict[str, Any],
    log_file: Path | None,
    run_folder: Path,
) -> None:
    """Public entry point. Writes all artifacts for one run.

    Layout produced:
        <run_folder>/
            manifest.json
            decisions.parquet
            run.log
            kept/
                <kept crop files>
            samples/
                rejected_at_<stage>/
                    <sample crop files>
    """
    _ensure_folder(run_folder)

    write_decisions_parquet(decisions, run_folder / "decisions.parquet")
    copy_kept_crops(decisions, crop_id_to_source, run_folder / "kept")
    copy_rejected_samples(decisions, crop_id_to_source, run_folder / "samples")
    write_manifest(manifest, run_folder / "manifest.json")
    if log_file is not None:
        copy_log_file(log_file, run_folder)

    logger.info("All artifacts written to {}", run_folder)