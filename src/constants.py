from pathlib import Path 

PROJECT_ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "raw"

if __name__ == '__main__':
    print(PROJECT_ROOT_DIR)
    print(DATA_DIR)
    print(IMAGES_DIR)