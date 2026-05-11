from __future__ import annotations 

from collections import defaultdict,Counter 
from dataclasses import dataclass 
from pathlib import Path 
from typing import Any 

import pandas as pd 
from sklearn.metrics import confusion_matrix as sk_confusion_matrix 

from src.entities.validation_label import ValidationLabel,ViolationReason
from src.logger.custom_logger import logger
from src.exceptions.custom_exceptions import ValidationError

VIOLATION_TO_EXPECTED_STAGE: dict[ViolationReason, str] = {
    ViolationReason.BLURRY: "quality",
    ViolationReason.NO_PERSON: "person_detection",
    ViolationReason.NOT_FULL_BODY: "pose",
    ViolationReason.FACE_HIDDEN: "face",
    ViolationReason.MINOR: "age",
    ViolationReason.ADVERTISEMENT: "ad_filter",
}

