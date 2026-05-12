from __future__ import annotations 

from dataclasses import dataclass,field 
from pathlib import Path 
from typing import Any 

import numpy as np 

# slots only allows the attributes defined in class 
@dataclass(slots=True)
class Crop:
    """A crop is an input image that every layer in component consumes. 
    It bundles loaded image as np array with metadata 
    that downstream stages record may reference"""

    crop_id: str # filename without extension.Used as primary key in decisions table
    source_path: Path # Original file path
    image: np.ndarray
    height: int 
    width: int 
    extras: dict[str,Any] = field(default_factory=dict)

    @property # allows us to use method like an attribute
    def shape(self) ->tuple[int,int,int]:
        return self.image.shape 