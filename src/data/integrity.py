# src/data/integrity.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from src.logger.custom_logger import logger


@dataclass(frozen=True, slots=True)
class DedupReport:
    n_input: int
    n_unique: int
    duplicate_groups: dict[str, list[str]] = field(default_factory=dict)

    @property
    def n_removed(self) -> int:
        """Number of redundant files removed. Derived, never stored."""
        return self.n_input - self.n_unique

    @property
    def n_duplicate_groups(self) -> int:
        """Number of hash groups containing more than one file."""
        return len(self.duplicate_groups)
    
    def to_summary_dict(self) -> dict:
        return {"n_input": self.n_input,
                "n_unique": self.n_unique,
                "n_removed": self.n_removed,
                "n_duplicate_groups": self.n_duplicate_groups}


    def to_dict(self) -> dict:
        """Full view including per-group filenames. Suitable for sidecar files."""
        return {**self.to_summary_dict(),
                "duplicate_groups": {
                    h[:8]: filenames
                    for h, filenames in self.duplicate_groups.items()},
                }


def _file_hash(path: Path) -> str:
    """MD5 of file contents — fast and sufficient for exact-duplicate detection."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def deduplicate_paths(paths: list[Path]) -> tuple[list[Path], DedupReport]:
    """Return paths with duplicates removed. Only the first path in each group is kept."""
    hash_to_paths: dict[str, list[Path]] = {}
    for path in paths:
        h = _file_hash(path)
        hash_to_paths.setdefault(h, []).append(path)

    unique_paths = [group[0] for group in hash_to_paths.values()]

    duplicate_groups = {h: [p.name for p in group]
                        for h, group in hash_to_paths.items()
                        if len(group) > 1}

    report = DedupReport(n_input=len(paths),
                         n_unique=len(unique_paths),
                         duplicate_groups=duplicate_groups,)

    logger.info(
        "Deduplication: {} -> {} ({} redundant copies in {} groups)",
        report.n_input,
        report.n_unique,
        report.n_removed,
        report.n_duplicate_groups,
    )
    return unique_paths, report