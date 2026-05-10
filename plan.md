# Assignment Compliance Audit

## Executive Verdict

The project is a strong partial prototype, but it does not yet meet all take-home assignment requirements or deliverables.

What is already present shows a sensible computer-vision filtering strategy: EDA, image quality checks, YOLO person detection, YOLO pose-based full-body filtering, InsightFace face visibility filtering, and a MiVOLO age-estimation smoke test. However, the repo does not yet contain a working end-to-end curation pipeline, production age filtering, production advertisement filtering, final validation metrics, or a design/results write-up sufficient to prove the required `>90%` dataset requirement coverage.

## Requirement-by-Requirement Status

| Assignment requirement | Status | Evidence | Current gap |
| --- | --- | --- | --- |
| Full-body person crops only | Partially Met | `src/components/person_detector.py` detects person crops with YOLO, and `src/components/pose_filter.py` checks visible head, torso, and leg keypoints. `config/thresholds.yaml` defines the person and pose thresholds. | This is implemented as component-level logic, but it is not yet wired into an end-to-end production pipeline. It also has no formal validation proving the full-body requirement is met above `90%`. |
| Face visible, frontal or side view | Partially Met | `src/components/face_filter.py` uses InsightFace/SCRFD detection and checks face confidence, face area ratio, and landmarks. Notebook `notebooks/05_stage3_face_visibility.ipynb` contains exploratory results. | The face filter exists, but there is no final labelled validation set or final evaluation report proving coverage. |
| Exclude crops from advertisements | Not Met | `config/thresholds.yaml` defines intended CLIP prompts for `clip_ad_filter`. | `src/components/ad_filter.py` is empty, `notebooks/07_stage5_ad_clip.ipynb` is empty, and no ad-filter results are available. |
| Exclude young children below teenager | Not Met | `config/thresholds.yaml` defines `age.min_age: 16`, and `notebooks/06_stage4_age_filter.ipynb` contains a MiVOLO smoke test on one crop. | `src/components/age_filter.py` is empty. There is no production age filter, no batch age results, and no validation for child exclusion. |
| Achieve `>90%` dataset requirement coverage | Not Proven | Exploratory notebooks/logs show stage counts through face filtering. | `data/labels/validation.csv` is empty, `notebooks/09_final_evaluation.ipynb` is empty, and there are no precision/recall/coverage metrics against labelled ground truth. |
| Computationally efficient and scalable | Partially Met | The staged design is efficient in principle: cheap OpenCV quality checks run first, followed by heavier YOLO, pose, face, age, and CLIP stages. Config includes runtime settings such as `device`, `batch_size`, and `num_workers`. | Current production code processes components individually. There is no implemented batch runner, manifest writer, decisions file, or benchmark showing throughput. |
| Minimize human labelling | Partially Met | The proposed approach uses pretrained conventional/deep CV models and only reserves labels for validation/tuning. | The validation label file is empty, and there is no implemented active-learning or label-helper workflow despite `scripts/label_helper.py` existing as an empty file. |
| Require minimal parameter tuning/generalize across datasets | Partially Met | Thresholds are centralized in `config/thresholds.yaml`, and notebooks include some sensitivity analysis for face thresholds. | `notebooks/08_threshold_tuning.ipynb` is empty, and there is no final evidence that the thresholds generalize. |
| Conventional CV allowed | Met | `src/components/quality_filter.py` uses OpenCV brightness and Laplacian blur checks. | No major gap for this specific allowance. |
| Deep-learning CV models allowed | Met | YOLO, YOLO-pose, InsightFace, MiVOLO, and CLIP are listed or explored. | Age and CLIP are not yet productionized. |
| Do not use Vision-Language Models | Met | The repo uses computer-vision models and CLIP-style image/text similarity, not Gemini, GPT, InternVL, Qwen-VL, or other VLM APIs. | Keep this constraint explicit in the final README/design write-up. |
| Working codebase executable end-to-end | Not Met | Individual components and notebooks exist. | `src/pipeline/curation_pipeline.py`, `src/pipeline/evaluation_pipeline.py`, `scripts/run_pipeline.py`, and `scripts/evaluate.py` are empty. `main.py` only prints a hello-world message. |
| Experimental results showing effectiveness | Not Proven | Logs and notebooks show exploratory counts through quality, detection, pose, and face stages. | Final evaluation artifacts are missing. `artifacts` currently only contains `artifacts/eda/01_eda_metrics.parquet`; no final decisions file, metrics report, or curated output manifest exists. |
| Brief system design and key decisions | Partially Met | The code structure and notebooks imply a staged cascade design. | `README.md` is very short and does not yet explain system design, key tradeoffs, thresholds, validation method, or final results. |

## What Is Already Built

### Dataset and EDA

- The raw dataset is present under `data/raw`.
- The repo currently has `1,147` image files in `data/raw`.
- EDA work exists in `notebooks/01_eda.ipynb`.
- EDA artifact `artifacts/eda/01_eda_metrics.parquet` exists.
- The EDA notebook indicates `1,147 / 1,148` loadable candidates, meaning one source item was likely corrupt or empty during inspection.

### Stage 0: Image Quality

- `src/components/quality_filter.py` is implemented.
- It checks minimum width, minimum height, aspect ratio, brightness range, and Laplacian blur variance.
- It returns a `StageResult` with metrics and rejection reasons.
- It uses conventional OpenCV techniques, which are allowed by the assignment.
- `tests/test_quality_filter.py` has real unit tests.

Important note: `config/thresholds.yaml` comments describe `min_aspect_ratio` as `width/height`, but the code calculates `height / width`. The code behavior appears aligned with tall person crops, but the comment is misleading and should be corrected later.

### Stage 1: Person Detection

- `src/components/person_detector.py` is implemented.
- It uses YOLOv11m with COCO person class `0`.
- It rejects crops with no valid person, too-small person area, or too many valid people.
- It stores the best normalized person bounding box in `crop.extras["person_bbox"]`.
- Exploratory notebook/log evidence shows Stage 1 results after quality filtering.

### Stage 2: Full-Body Pose Filtering

- `src/components/pose_filter.py` is implemented.
- It uses YOLO pose keypoints.
- It checks head, torso, and leg keypoint visibility.
- This directly targets the assignment requirement that crops contain full-body people with no missing feet or hands.
- Notebook `notebooks/04_stage2_pose_fullbody.ipynb` contains exploratory pose diagnostics and counts.

### Stage 3: Face Visibility

- `src/components/face_filter.py` is implemented.
- It uses InsightFace/SCRFD face detection.
- It checks detection confidence, face area ratio, and whether landmarks fall inside the face box.
- It stores a normalized face box in `crop.extras["face_bbox"]`.
- Notebook `notebooks/05_stage3_face_visibility.ipynb` contains exploratory face diagnostics and threshold sensitivity checks.

### Stage 4: Age Exploration

- `notebooks/06_stage4_age_filter.ipynb` contains a MiVOLO smoke test.
- The notebook successfully selects a crop that passed stages 0-3, builds face/person crops, runs MiVOLO, and prints a plausible age prediction.
- This is useful evidence that the intended age-filter approach is feasible.

However, this is not yet production implementation.

### Shared Interfaces and Configuration

- `src/components/base.py` defines `BaseFilter`, which wraps stage execution with timing, logging, and exception handling.
- `src/entities/stage_results.py` defines `StageResult`.
- `src/entities/final_decision.py` defines `FinalDecision` and a `to_flat_record()` method for tabular outputs.
- `src/settings/config.py` defines config and threshold models.
- `config/config.yaml` and `config/thresholds.yaml` centralize runtime/model/threshold settings.

These are good foundations for the final pipeline.

### Exploratory Stage Counts

Notebook and log evidence shows the system has been explored through the first four filtering stages. The most useful notebook evidence found so far is:

| Stage checkpoint | Approximate count |
| --- | ---: |
| Raw/loadable images | `1,147` |
| Unique image records used in some notebooks | `1,001` |
| Quality survivors | `784` |
| Person detection survivors | `461` |
| Pose/full-body survivors | `208` |
| Face visibility survivors | `124` |

These numbers are useful for development diagnostics, but they are not final assignment metrics because they are not evaluated against labelled requirement coverage.

## What Is Missing

### End-to-End Pipeline

The following files are currently empty:

- `src/pipeline/curation_pipeline.py`
- `src/pipeline/evaluation_pipeline.py`
- `scripts/run_pipeline.py`
- `scripts/evaluate.py`

Because these are empty, the project does not yet provide a working end-to-end executable codebase. The current code can run individual stages and notebooks, but it does not yet produce a complete curated dataset, `decisions.parquet`, `manifest.json`, or final accepted/rejected output directories.

### Production Age Filter

`src/components/age_filter.py` is empty.

The MiVOLO notebook proves feasibility on one crop, but the production system still needs:

- Model loading.
- Face/person crop preparation.
- Batch or per-image inference.
- Age threshold decision logic.
- `StageResult` metrics and rejection reasons.
- Tests.

Until this exists, the assignment requirement to exclude young children is not met.

### Production Advertisement Filter

`src/components/ad_filter.py` is empty.

`config/thresholds.yaml` defines CLIP ad-filter prompts, but the project still needs:

- CLIP model loading.
- Image preprocessing.
- Real-person vs advertisement similarity scoring.
- Decision logic using `similarity_margin`.
- `StageResult` metrics and rejection reasons.
- Tests.

Until this exists, the assignment requirement to exclude advertisement-origin crops is not met.

### Final Evaluation and `>90%` Coverage Proof

The assignment explicitly expects `>90%` dataset requirement coverage. This is not proven yet.

Current blockers:

- `data/labels/validation.csv` is empty.
- `notebooks/09_final_evaluation.ipynb` is empty.
- `src/pipeline/evaluation_pipeline.py` is empty.
- `scripts/evaluate.py` is empty.
- No final precision, recall, coverage, false-positive, or false-negative metrics were found.

The project should not claim `>90%` coverage until labelled validation data and final metrics exist.

### Threshold Tuning

`notebooks/08_threshold_tuning.ipynb` is empty.

Some threshold sensitivity work exists in earlier notebooks, especially the face notebook, but the project still lacks a consolidated tuning notebook/report showing how thresholds were chosen and why they generalize.

### Backend and Frontend

The following files are empty:

- `backend/main.py`
- `backend/schemas.py`
- `frontend/app.py`

These are not required by the assignment unless presented as part of the deliverable. If kept, they should either be implemented or excluded from the final story to avoid looking unfinished.

### Documentation

`README.md` currently only contains a one-line project description.

The final submission still needs:

- How to install dependencies.
- How to place/download the dataset.
- How to run the full pipeline.
- How to run evaluation.
- System design explanation.
- Key design decisions.
- Final results table.
- Limitations and failure modes.
- Explicit statement that no VLMs are used.

## Testing Status

Running `pytest -q` currently reports:

```text
5 passed, 1 warning
```

The warning is a pytest cache warning caused by denied access while writing under `.pytest_cache`.

The passing tests are only for `QualityFilter`. These files are empty and do not yet test their corresponding behavior:

- `tests/test_person_detector.py`
- `tests/test_pose_filter.py`
- `tests/test_face_filter.py`
- `tests/test_age_filter.py`
- `tests/test_ad_filter.py`
- `tests/conftest.py`

There are also no tests yet for:

- End-to-end curation orchestration.
- Final decision flattening/output writing.
- Evaluation metrics.
- CLI scripts.
- Backend/frontend code.

## Current Compliance Conclusion

The project currently satisfies the early research/prototyping part of the assignment well, especially for the first three real filtering stages:

- Quality.
- Person detection.
- Full-body pose.
- Face visibility.

It does not yet satisfy the full deliverable requirements because the production pipeline stops before age/ad filtering and final evaluation. The most important current truth is:

> The project partially addresses the assignment, but it cannot yet be submitted as meeting all requirements because the end-to-end system and `>90%` coverage evidence are missing.

## Recommended Next Work

1. Implement `src/pipeline/curation_pipeline.py`.
   - Load configs and thresholds.
   - Run stages in order.
   - Stop at first failed stage.
   - Produce `FinalDecision` records.
   - Write `decisions.parquet` and `manifest.json`.

2. Implement `scripts/run_pipeline.py`.
   - Provide a simple CLI command to run the full pipeline from `data/raw`.
   - Save outputs under `artifacts`.

3. Implement `src/components/age_filter.py`.
   - Move the MiVOLO smoke-test logic into production code.
   - Return `StageResult` with predicted age and threshold decision.

4. Implement `src/components/ad_filter.py`.
   - Use CLIP/open-clip prompts already defined in `config/thresholds.yaml`.
   - Return real/ad similarity metrics and threshold decision.

5. Create a small labelled validation set.
   - Populate `data/labels/validation.csv`.
   - Include labels for full body, face visible, ad/non-ad, child/adult, and final keep/reject.

6. Implement `src/pipeline/evaluation_pipeline.py` and `scripts/evaluate.py`.
   - Compare pipeline decisions against validation labels.
   - Report coverage, precision, recall, false accepts, and false rejects.
   - Only claim `>90%` if the metrics prove it.

7. Complete `notebooks/08_threshold_tuning.ipynb` and `notebooks/09_final_evaluation.ipynb`.
   - Use them to document threshold selection and final measured performance.

8. Expand tests.
   - Keep model-heavy tests mocked or smoke-level.
   - Add unit tests for decision logic.
   - Add integration tests for pipeline orchestration using fake filters.

9. Update `README.md`.
   - Explain setup, usage, system design, model choices, results, and limitations.

## Final Submission Readiness

| Deliverable | Current readiness |
| --- | --- |
| Working executable codebase | Not ready |
| Experimental results | Not ready |
| Design explanation | Partially ready in code/notebooks, not ready in README |
| Requirement coverage proof | Not ready |
| Minimal human-labelling story | Partially ready conceptually, not implemented |
| No-VLM compliance | Ready, based on current model choices |

