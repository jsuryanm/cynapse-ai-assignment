from __future__ import annotations

from pathlib import Path 

import numpy as np

from PIL import Image,ImageOps,UnidentifiedImageError

from src.entities.crop import Crop 
from src.logger.custom_logger import logger 
from src.exceptions.custom_exceptions import ImageProcessingError

class ImageLoader:
    """Single source of truth for loading image crops goes through this class.
    Centralizing loading guarentees consistent behavior:
    1. Corrupt files raises ImageProcessingError
    2. EXIF orientation applied (Determines how an img should be rotated to appear correctly)
    3. Output is always RGB (H,W,3)
    4. No image preprocessing (no resizing, no normalization, no CLAHE)
    """

    SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg","jpeg",".png"})

    def load(self,path: Path) -> Crop:
        """Load single image and return Crop entity"""

        if not path.exists():
            raise ImageProcessingError(f"Image not found: {path}")
        
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ImageProcessingError(f"Unsupported extension {path.suffix!r} for {path.name}"
                                       f"expected one of {sorted(self.SUPPORTED_EXTENSIONS)}")
        
        try:
            with Image.open(path) as probe:
                probe.verify()
        except (UnidentifiedImageError,OSError) as e:
            raise ImageProcessingError(f"Corrupt or truncated image: {path}") from e 
        
        try:
            with Image.open(path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                pil_img = pil_img.convert("RGB")
                array = np.asarray(pil_img,dtype=np.uint8)
            
        except (UnidentifiedImageError,OSError) as e:
            raise ImageProcessingError(f"Failed to decode image: {path}") from e 
        
        if array.ndim != 3 or array.shape[2] != 3:
            raise ImageProcessingError(f"Unexpected Image shape: {array.shape} for {path}"
                                       f"Expected (H,W,3) RGB.")
        
        height,width = int(array.shape[0]),int(array.shape[1])
        
        crop = Crop(crop_id=path.stem,
                    source_path=path,
                    image=array,
                    height=height,
                    width=width)
        
        logger.info(f"Loaded {path.name} | shape: ({width,height,3}) | bytes: {array.nbytes}")

        return crop 

# if __name__ == "__main__":
#     PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]
#     DATA_DIR = PROJECT_ROOT_DIR / "data"
#     test_img_path = DATA_DIR / "raw" /  "crop (1088).png"

#     img_loader = ImageLoader()
#     test_crop = img_loader.load(test_img_path)
#     print(test_crop)