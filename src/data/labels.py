from __future__ import annotations 

from collections import Counter 
from pathlib import Path 

import pandas as pd 

from src.entities.validation_label import ValidationLabel
from src.exceptions.custom_exceptions import ValidationError
from src.logger.custom_logger import logger 

REQUIRED_COLUMNS = {"crop_id","should_keep","violation_reason"}

def load_validation_csv(csv_path: Path) -> list[ValidationLabel]:
    """Load and validate ground-truth labels from CSV"""
    if not csv_path.exists():
        raise ValidationError(f"Validation labels file not found: {csv_path}")
    
    df = pd.read_csv(csv_path,dtype={"crop_id":str})
    
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing: 
        raise ValidationError(f"{csv_path.name} missing required columns: {sorted(missing)}")
    
    df["violation_reason"] = df["violation_reason"].where(df["violation_reason"].notna(),None)

    if "notes" not in df.columns:
        df["notes"] = ""
    else:
        df["notes"] = df["notes"].fillna("")
    
    labels: list[ValidationLabel] = []

    for i,record in enumerate(df.to_dict(orient="records"),start=2): 
        # converts df to list of dicts containing row data
        try:
            labels.append(ValidationLabel(**record))
        except Exception as e:
            raise ValidationError(f"{csv_path.name} row {i}: {e}") from e
    
    crop_ids = [label.crop_id for label in labels]
    duplicates = [cid for cid, n in Counter(crop_ids).items() if n > 1]

    if duplicates:
        raise ValidationError(f"{csv_path.name} has duplicate crop_ids: {sorted(duplicates)}")
    
    logger.info(f"Loaded {len(labels)} from {csv_path.name}")
    return labels