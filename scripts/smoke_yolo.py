from __future__ import annotations 

import torch 
from ultralytics import YOLO 
import time

from src.logger.custom_logger import logger 
from src.utils.io import ImageLoader
from src.constants import PROJECT_ROOT_DIR,IMAGES_DIR
from src.settings.config import ModelsConfig


model_config = ModelsConfig()

model_name = model_config.yolo_detection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Loading {model_name} on {DEVICE}")
t0 = time.perf_counter()

model = YOLO(model_name).to(DEVICE)
load_ms = (time.perf_counter() - t0) * 1000 
logger.info(f"Model loaded in {load_ms:.0f} ms")

loader = ImageLoader()
sample_paths = sorted(IMAGES_DIR.glob("*.png"))[:2]
logger.info(f"Inspecting samples: {[p.name for p in sample_paths]}")

crop_a = loader.load(sample_paths[0])
t0 = time.perf_counter()
results_a = model(crop_a.image,
                  conf=0.25,
                  classes=[0],
                  verbose=False)

cold_ms = (time.perf_counter() - t0) * 1000 
logger.info(f"Cold inference: {cold_ms:.0f} ms")

crop_b = loader.load(sample_paths[1])
t0 = time.perf_counter()

results_b = model(crop_b.image,
                  conf=0.25,
                  classes=[0],
                  verbose=False)

warm_ms = (time.perf_counter() - t0) * 1000
logger.info(f"Warm inference {warm_ms:.0f} ms")

print("\n" + "=" * 60)
print(f"INSPECTING RESULT FOR {crop_a.crop_id}")
print("=" * 60)

print(f"Type of results: {type(results_a)}")
print(f"Length of results: {len(results_a)}")

result = results_a[0]
print(f"Type of results[0]: {type(result)}")
print(f"Type of results.boxes: {type(result.boxes)}")
print(f"Number of person detections: {len(result.boxes)}")

if len(result.boxes) > 0: 
    print(f"Pixel coords (x1,y1,x2,y2): {result.boxes.xyxy.cpu().numpy()}")
    print(f"normalized [0-1]: {result.boxes.xyxyn.cpu().numpy()}")
    print(f"confidence per detection: {result.boxes.conf.cpu().numpy()}")
    print(f"class index per detection: {result.boxes.cls.cpu().numpy()}")
    print(f"crop dimensions (h,w,c): {crop_a.image.shape}")
else:
    print("No person detections this fails in stage 1")

print("\n" + "=" * 60)
print("TIMING SUMMARY")
print("=" * 60)
print(f"Model load:     {load_ms:>7.0f} ms  (one-time at startup)")
print(f"Cold inference: {cold_ms:>7.0f} ms  (first call — GPU warmup)")
print(f"Warm inference: {warm_ms:>7.0f} ms  (subsequent calls)")
