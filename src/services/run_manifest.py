from __future__ import annotations

import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from src.data.integrity import DedupReport
from src.entities.final_decision import FinalDecision


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tool_versions() -> dict[str, str]:
    """Capture versions of important runtime dependencies."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in ("torch", "ultralytics", "transformers", "insightface"):
        try:
            mod = __import__(pkg.replace("-", "_"))
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


def _build_stage_stats(decisions: list[FinalDecision],
                       stage_order: list[str]) -> list[dict[str, Any]]:
    """Compute the number of images passes and fail each stage.
    For each stage, returns the number of crops that entered, the number
    that passed, the number rejected, and the per-stage rejection rate.
    """
    rejected_by_stage: Counter[str] = Counter()
    for d in decisions:
        if not d.kept and d.rejected_at_stage is not None:
            rejected_by_stage[d.rejected_at_stage] += 1

    unknown = set(rejected_by_stage) - set(stage_order)
    if unknown:
        raise ValueError(f"Decisions reference unknown stages {unknown}; "
                         f"expected one of {stage_order}")

    stats: list[dict[str, Any]] = []
    n_in = len(decisions)
    for stage in stage_order:
        n_rejected = rejected_by_stage.get(stage, 0)
        n_passed = n_in - n_rejected
        rejection_rate = (n_rejected / n_in) if n_in > 0 else 0.0
        stats.append({"stage": stage,
                      "n_in": n_in,
                      "n_passed": n_passed,
                      "n_rejected": n_rejected,
                      "rejection_rate": round(rejection_rate, 4)})
        n_in = n_passed  
    return stats


def build_manifest(run_id: str,
                   started_at: str,
                   finished_at: str,
                   data_dir: str,
                   dedup_report: DedupReport,
                   decisions: list[FinalDecision],
                   stage_order: list[str],
                   config_snapshot: dict[str, Any],) -> dict[str, Any]:
    """Assemble manifest dict for a full pipeline run."""
    outcomes: Counter[str] = Counter()
    for d in decisions:
        if d.kept:
            outcomes["passed_all"] += 1
        else:
            outcomes[f"rejected_at_{d.rejected_at_stage}"] += 1

    started_dt = datetime.fromisoformat(started_at.rstrip("Z"))
    finished_dt = datetime.fromisoformat(finished_at.rstrip("Z"))
    elapsed_seconds = (finished_dt - started_dt).total_seconds()

    stage_stats = _build_stage_stats(decisions, stage_order)

    return {"run_id": run_id,
            "started_at": started_dt,
            "finished_at": finished_dt,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "data_dir": data_dir,
            "dedup": dedup_report.to_dict(),
            "outcomes": dict(outcomes),
            "stage_stats": stage_stats,
            "config_snapshot": config_snapshot,
            "tool_versions": _tool_versions()}