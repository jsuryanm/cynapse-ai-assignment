from __future__ import annotations 

import hashlib 
from dataclasses import dataclass,field
from pathlib import Path 

from src.logger.custom_logger import logger

@dataclass(frozen=True,slots=True)
class DedupReport:
    """Summary of a deduplication pass"""
    n_input: int 
    n_unique: int 
    duplicate_groups: dict[str,list[str]] = field(default_factory=list)

    @property
    def n_remove(self) -> int:
        return self.n_input - self.n_unique
    
    def to_dict(self) -> dict:
        return {"n_input":self.n_input,
                "n_unique":self.n_unique,
                "n_removed":len(self.duplicate_groups),
                "duplicate_groups":{
                    h[:8]:filenames
                    for h,filenames in self.duplicate_groups.items()
                }}
    
def _file_hash(path: Path) -> str:
    """MD5 file contents, fast and sufficient for duplicate detection"""
    return hashlib.md5(path.read_bytes()).hexdigest()

def deduplicate_paths(paths: list[Path]) -> tuple[list[Path],DedupReport]:
    """Return paths with duplicates removed only first path in each group is kept"""
    
    hash_to_paths: dict[str,list[Path]] = {}
    for path in paths:
        h = _file_hash(path)
        hash_to_paths.setdefault(h,[]).append(path)

    unique_paths = [group[0] for group in hash_to_paths.values()] 
    
    duplicate_groups =  {
        h: [p.name for p in group]
        for h,group in hash_to_paths.items()
        if len(group) > 1
    } 

    report = DedupReport(n_input=len(paths),
                         n_unique=len(unique_paths),
                         duplicate_groups=duplicate_groups)
    
    logger.info(f"Deuplication of {report.n_input} images to {report.n_unique} unique images having ({report.n_remove} redundant copies in {len(duplicate_groups)} groups)")
    return unique_paths,report 