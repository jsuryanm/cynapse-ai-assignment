import os 
from pathlib import Path
import logging 

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]: %(message)s")

project_name = "curation"

list_of_files = [
    ".github/workflows/.gitkeep",
    "config/config.yaml",
    "config/thresholds.yaml",
    
    "data/raw/.gitkeep",
    "data/labels/validation.csv",
    "data/README.md",
    
    "notebooks/01_eda.ipynb",
    "notebooks/02_stage0_quality.ipynb",
    "notebooks/03_stage1_person_detection.ipynb",
    "notebooks/04_stage2_pose_fullbody.ipynb",
    "notebooks/05_stage3_face_visibility.ipynb",
    "notebooks/06_stage4_age_filter.ipynb",
    "notebooks/07_stage5_ad_clip.ipynb",
    "notebooks/08_threshold_tuning.ipynb",
    "notebooks/09_final_evaluation.ipynb",

    "src/__init__.py",
    "src/constants.py",

    "src/logger/__init__.py",
    "src/logger/custom_logger.py",
    
    "src/settings/__init__.py",
    "src/settings/config.py",

    "src/exceptions/__init__.py",
    "src/exceptions/custom_exceptions.py",
    
    "src/entities/__init__.py",
    "src/entities/crop.py",
    "src/entities/stage_results.py",
    "src/entities/final_decision.py",

    "src/data/__init__.py",
    "src/data/loader.py",
    "src/data/integrity.py",

    "src/components/__init__.py",
    "src/components/base.py",
    "src/components/quality_filter.py",
    "src/components/person_detector.py",
    "src/components/pose_filter.py",
    "src/components/face_filter.py",
    "src/components/age_filter.py",
    "src/components/ad_filter.py",

    "src/pipeline/__init__.py",
    "src/pipeline/curation_pipeline.py",
    "src/pipeline/evaluation_pipeline.py",

    "src/utils/__init__.py",
    "src/utils/io.py",
    "src/utils/visualization.py",
    "src/utils/metrics.py",

    "src/services/__init__.py",
    "src/services/run_manifest.py",
    "src/services/artifact_writer.py",

    "scripts/run_pipeline.py",
    "scripts/evaluate.py",
    "scripts/label_helper.py",

    "backend/__init__.py",
    "backend/schemas.py",
    "backend/main.py",

    "frontend/__init__.py",
    "frontend/app.py",

    "tests/__init__.py",
    "tests/conftest.py",
    "tests/test_quality_filter.py",
    "tests/test_person_detector.py",
    "tests/test_pose_filter.py",
    "tests/test_face_filter.py",
    "tests/test_age_filter.py",
    "tests/test_ad_filter.py",

    "requirements.txt",]


for file_path in list_of_files:
    file_path =  Path(file_path)
    file_dir,file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path,"w") as f:
            pass
            logging.info(f"Creating an empty file: {file_path}")
    
    else:
        logging.info(f"{file_name} already exists")