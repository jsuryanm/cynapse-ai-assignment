from pathlib import Path 
import torch

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "raw"
DEFAULT_CONFIG_PATH = PROJECT_ROOT_DIR / "config" / "config.yaml"
DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT_DIR / "config" / "thresholds.yaml"


HEAD_KPT_INDICES = (0,1,2,3,4) 
SHOULDER_KPT_INDICES = (5, 6) 
HIP_KPT_INDICES = (11, 12) 
KNEE_KPT_INDICES = (13, 14) 

LANDMARK_NAMES = ["right_eye", "left_eye", "nose", "right_mouth", "left_mouth"]
COLORS = ["red", "blue", "green", "orange", "purple"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AGE_MODEL_ID = "iitolstykh/mivolo_v2"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

