# Computer Vision Dataset Curation Pipeline

This project implements an end-to-end computer vision pipeline for curating a
noisy person-crop dataset. The goal is to automatically filter the dataset so
the final training set contains only valid full-body person crops with visible
faces, while rejecting advertisements and young children.

## Assignment Requirements

The final dataset must satisfy these requirements:

| Requirement | Handling |
|---|---|
| Full-body person crops only | YOLO person detection + YOLO pose keypoint checks |
| Face must be visible | InsightFace SCRFD face detection |
| Exclude advertisements | CLIP zero-shot ad/mannequin filter |
| Exclude young children | MiVOLOv2 age estimation |

The assignment also asks for:

- greater than 90% dataset requirement coverage
- computationally efficient and scalable filtering
- minimal human labelling
- minimal parameter tuning with reasonable generalization
- no Vision-Language Models such as Gemini, GPT, InternVL, or Qwen-VL

This solution uses conventional and deep-learning computer vision models only:
YOLO, InsightFace, MiVOLO, and CLIP.

## Current Status

The project is functionally complete for the take-home assignment.

| Component | Status |
|---|---|
| Quality filter | Done |
| Person detector | Done |
| Full-body pose filter | Done |
| Face visibility filter | Done |
| Age/minor filter | Done |
| Advertisement filter | Done |
| End-to-end pipeline runner | Done |
| Evaluation script | Done |
| Final validation run | Done |

## Pipeline Design

The system is a cascading filter pipeline. Each crop must pass every stage to be
kept.

```text
raw crops
  -> optional deduplication
  -> quality filter
  -> person detector
  -> pose / full-body filter
  -> face visibility filter
  -> age filter
  -> ad / mannequin filter
  -> curated kept crops
```

The cascade is used because it is efficient: cheap checks run first, and
expensive models are only applied to crops that survive earlier stages.

Each stage returns an auditable decision with metrics and a rejection reason.
The final output includes one decision record per evaluated crop.

## Quick Start

Install dependencies:

```bash
uv sync
```

Place the dataset images under:

```text
data/raw/*.png
```

Run the normal production pipeline:

```bash
python scripts/run_pipeline.py
```

Run final validation reporting:

```bash
python scripts/run_pipeline.py --skip-dedup
python scripts/evaluate.py
```

Final validation uses `--skip-dedup` so every labelled crop in
`data/labels/validation.csv` receives a direct pipeline decision. The production
pipeline can still use deduplication to avoid duplicate images in the curated
dataset.

## Outputs

Each run writes a timestamped folder under:

```text
artifacts/runs/<run_id>/
```

Important outputs:

| Output | Purpose |
|---|---|
| `manifest.json` | run metadata and stage counts |
| `decisions.parquet` | one row per crop with final decision and stage trace |
| `evaluation.json` | validation metrics from `scripts/evaluate.py` |
| `kept/` | accepted crops |
| `samples/` | representative rejected samples by stage |
| `run.log` | run log |

## Final Validation Results

Final validation was run with:

```bash
python scripts/run_pipeline.py --skip-dedup
python scripts/evaluate.py
```

Overall performance:

| Metric | Value |
|---|---:|
| Crops evaluated | 208 |
| Labels without decisions | 0 |
| True positives | 16 |
| False negatives | 14 |
| True negatives | 170 |
| False positives | 8 |
| Accuracy | 89.42% |
| Precision keep | 66.67% |
| Recall keep | 53.33% |
| F1 keep | 59.26% |
| Coverage | 95.51% |

The main assignment target is greater than 90% dataset requirement coverage.
This run achieves 95.51%, so the main coverage requirement is met.

Per-violation coverage:

| Violation | Labelled | Rejected | Coverage |
|---|---:|---:|---:|
| no_person | 38 | 38 | 100.00% |
| blurry | 24 | 24 | 100.00% |
| face_hidden | 29 | 29 | 100.00% |
| advertisement | 34 | 33 | 97.06% |
| not_full_body | 45 | 40 | 88.89% |
| minor | 8 | 6 | 75.00% |

Some labelled violations are rejected before their dedicated stage because the
pipeline is a cascade. For example, many advertisement crops fail person, pose,
face, or quality checks before reaching the CLIP ad filter. The primary metric
is therefore overall reject coverage.

## Minimal Tuning And Generalization

The pipeline is designed to require minimal parameter tuning:

- It uses pretrained general-purpose computer vision models.
- No custom model is trained on the assignment dataset.
- Tunable parameters are centralized in `config/thresholds.yaml`.
- The exposed thresholds are few and interpretable.
- The same pipeline can be run on another similar person-crop dataset without
  changing model code.

Cross-dataset generalization is supported by the use of pretrained YOLO,
InsightFace, MiVOLO, and CLIP models. The current evidence is based on the
provided dataset; an external held-out dataset would be useful future evidence.

## Design Decisions

### Cascade Instead Of One Large Model

A cascade short-circuits early. This keeps the pipeline efficient and makes each
rejection reason easy to inspect.

### YOLO For Person And Pose

YOLOv11m provides fast person detection, and YOLOv11m-pose provides the
keypoints needed to check whether the crop is full-body.

### InsightFace For Face Visibility

Face detection is handled separately because pose face keypoints can be noisy
for small or side-view faces.

### MiVOLO For Age

MiVOLO is used because it is designed for age estimation from face and body
signals. This avoids using a forbidden Vision-Language Model.

### CLIP For Advertisements

CLIP is used as a zero-shot image-text embedding model to separate real-person
crops from advertisement/mannequin-like images without training a custom
classifier.

## Limitations

- `minor` coverage is based on only 8 labelled examples because the dataset has
  limited minor samples. This makes the minor percentage less stable than larger
  categories.
- `not_full_body` coverage is 88.89%, slightly below 90%, although total reject
  coverage remains above the assignment target.
- Generalization is designed for through pretrained models and minimal
  thresholds, but it has only been evaluated on the provided dataset.
- Increasing strictness could improve reject coverage further, but may reduce
  the number of valid crops kept.

## Tests

Run:

```bash
python -m pytest -q
```

The repository includes unit tests for core filtering behavior and the final
verification command should pass before submission.

## Project Structure

```text
config/
  config.yaml
  thresholds.yaml
data/
  raw/
  labels/validation.csv
scripts/
  run_pipeline.py
  evaluate.py
src/
  components/
  pipeline/
  services/
  data/
  entities/
tests/
artifacts/runs/
```

## License

This project is part of a private take-home assignment and is not currently
licensed for redistribution.
