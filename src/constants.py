from pathlib import Path 

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "raw"

HEAD_KPT_INDICES = (0,1,2,3,4)
TORSO_KPT_INDICES =  (5,6,11,12)
LEG_KPT_INDICES = (13,14,15,16)

LANDMARK_NAMES = ["right_eye", "left_eye", "nose", "right_mouth", "left_mouth"]
COLORS = ["red", "blue", "green", "orange", "purple"]


# if __name__ == '__main__':
#     print(PROJECT_ROOT_DIR)
#     print(DATA_DIR)
#     print(IMAGES_DIR)