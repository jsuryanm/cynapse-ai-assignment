from __future__ import annotations 

import platform 
import sys 

from collections import Counter
from datetime import datetime,timezone
from typing import Any 

from src.data.integrity import DedupReport
from src.entities.final_decision import FinalDecision

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _tool_versions() -> dict[str,str]:
    """Capture versions of important runtime dependencies"""
    
    versions: dict[str,str] = {"python":sys.version.split()[0],
                              "platform":platform.platform()} # platform.platform() returns os information

    for pkg in ("torch", "ultralytics", "transformers", "insightface", "open_clip_torch"):
        try:
            mod = __import__(pkg.replace("-", "_"))
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


def build_manifest(run_id: str,
                   started_at: str,
                   finished_at: str,
                   data_dir: str,
                   dedup_report: DedupReport,
                   decisions: list[FinalDecision],
                   config_snapshot: dict[str,Any]) -> dict[str,Any]:
    """Assemble manifest dict for running the entire pipeline"""
    outcomes: Counter[str] = Counter()

    for d in decisions:
        if d.kept:
            outcomes['passed_all'] += 1 
        else:
            outcomes[f"rejected_at_{d.rejected_at_stage}"] += 1
        
    started_dt = datetime.fromisoformat(started_at.rstrip("Z"))
    finished_dt = datetime.fromisoformat(finished_at.rstrip("Z"))
    elapsed_seconds = (finished_dt - started_dt).total_seconds()

    return {"run_id":run_id,
            "started_at":started_dt,
            "finished_at":finished_dt,
            "elapsed_seconds":round(elapsed_seconds,2),
            "data_dir":data_dir,
            "dedup":dedup_report.to_dict(),
            "outcomes":dict(outcomes),
            "config_snapshot":config_snapshot,
            "tool_versions":_tool_versions()}