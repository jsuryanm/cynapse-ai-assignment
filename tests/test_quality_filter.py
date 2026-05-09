from __future__ import annotations 

from pathlib import Path

import numpy as np 
import pytest 

from src.constants import IMAGES_DIR
from src.entities.crop import Crop 
from src.components.quality_filter import QualityFilter
from src.settings.config import QualityThresholds

# creates reusable test data, instead rebuilding 
# thresholds pytest will automatically inject it. 
 
@pytest.fixture 
def thresholds() -> QualityThresholds:
    """Thresholds tuned during EDA"""
    return QualityThresholds()

def _make_crop(image: np.ndarray,
               crop_id: str = "test") -> Crop:
    """Builds Crop obj from np array"""
    h,w = image.shape[:2]
    return Crop(crop_id=crop_id,
                source_path=Path(f"/tmp/{crop_id}.png"),
                image=image,
                height=h,
                width=w)

def test_passes_good_crop(thresholds: QualityThresholds) -> None:
    """sample noisy crop should pass the test checks"""
    rng =  np.random.default_rng(seed=42)
    image = rng.integers(50,200,size=(400,200,3),dtype=np.uint8)
    result = QualityFilter(thresholds).apply(_make_crop(image,"good"))

    assert result.passed is True # assert checks if condition is True 
    assert result.stage_name == "quality"
    assert result.reason == "ok"
    assert set(result.metrics) == {"width","height","aspect","brightness","blur_variance"}

def test_rejects_too_small(thresholds: QualityThresholds) -> None:
    """Near black images fail brightness check"""
    image = np.full((400, 200, 3), 5, dtype=np.uint8)
    result = QualityFilter(thresholds).apply(_make_crop(image, "dark"))

    assert result.passed is False
    assert "brightness" in result.reason

def test_rejects_too_blurry(thresholds: QualityThresholds) -> None:
    """Uniform-color image has zero Laplacian variance."""
    image = np.full((400, 200, 3), 128, dtype=np.uint8)
    result = QualityFilter(thresholds).apply(_make_crop(image, "uniform"))

    assert result.passed is False
    assert "blur" in result.reason

def test_collects_multiple_failure_reasons(thresholds: QualityThresholds) -> None:
    """Crop that fails multiple criteria reports all of them."""
    image = np.full((50, 30, 3), 5, dtype=np.uint8)  # tiny + dark + uniform
    result = QualityFilter(thresholds).apply(_make_crop(image, "bad"))

    assert result.passed is False
    assert "|" in result.reason  # multiple failures joined
    # Metrics still populated even on failure
    assert result.metrics["width"] == 30.0
    assert result.metrics["height"] == 50.0


def test_metrics_present_on_pass(thresholds: QualityThresholds) -> None:
    """Even passing crops carry full metrics for downstream tuning."""
    rng = np.random.default_rng(seed=0)
    image = rng.integers(50, 200, size=(400, 200, 3), dtype=np.uint8)
    result = QualityFilter(thresholds).apply(_make_crop(image, "good"))

    assert result.metrics["blur_variance"] > 0
    assert 0 <= result.metrics["brightness"] <= 255
