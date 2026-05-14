from __future__ import annotations 

from dataclasses import dataclass,field 
from pathlib import Path 
from typing import Any 

import numpy as np 

@dataclass(slots=True)
class Crop:
    """A crop is an input image that every layer in component consumes. 
    It bundles loaded image as np array with metadata 
    that downstream stages record may reference"""

    crop_id: str 
    source_path: Path 
    image: np.ndarray
    height: int 
    width: int 
    extras: dict[str,Any] = field(default_factory=dict)

    @property 
    def shape(self) ->tuple[int,int,int]:
        return self.image.shape 