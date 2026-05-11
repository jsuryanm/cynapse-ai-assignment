# Computer Vision Dataset Curation Pipeline

A production-style, multi-stage inference pipeline that automatically curates a
noisy person-crop dataset down to crops that meet strict downstream 
requirements.


---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Approach](#approach)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Implementation Status](#implementation-status)
5. [Quick Start](#quick-start)
6. [Project Structure](#project-structure)
7. [Configuration](#configuration)
8. [Running the Pipeline](#running-the-pipeline)
9. [Outputs & Run Artifacts](#outputs--run-artifacts)
10. [Logging](#logging)
11. [Evaluation](#evaluation)
12. [Testing](#testing)
13. [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)
14. [Results](#results)
15. [Limitations & Future Work](#limitations--future-work)
16. [VLM Compliance](#vlm-compliance)
17. [Tech Stack](#tech-stack)

---

## Problem Statement

We are given a noisy dataset of person crops. The final training dataset must
satisfy **four requirements**:

| # | Requirement |
|---|---|
| 1 | Full-body person crops only — no missing feet or hands |
| 2 | Face must be visible — frontal or side view |
| 3 | Exclude crops originating from advertisements / mannequins |
| 4 | Exclude crops of young children (below teenager, < 16 years) |

**Constraints**

- Achieve **> 90% dataset requirement coverage** on a labelled validation slice.
- Minimise manual labelling.
- Generalise across datasets with minimal threshold re-tuning.
- Be computationally efficient and scalable.
- **No Vision-Language Models** (Gemini, GPT-4V, InternVL, Qwen-VL, etc.).

Conventional CV (OpenCV) and deep-learning CV models (detection, pose, face,
classification, CLIP) **are** allowed.

---

## Approach

The system is a **6-stage cascading filter** (`Pipeline` pattern). Each stage
is an independent, swappable component implementing a common `BaseFilter`
interface. A crop must pass *every* stage to be accepted.

**Why a cascade and not a single multi-task model?**

- **Cheap before expensive.** Stage 0 (OpenCV quality) is microseconds per
  image. Stage 1 (YOLO) is ~10 ms. Stage 5 (CLIP) is ~30 ms. Putting the cheap
  filters first means most rejects never reach the expensive stages —
  dramatically reducing total compute.
- **Each requirement maps to a dedicated, *interpretable* signal.** When a crop
  is rejected we know *which* requirement it failed, not just that "the model
  said no". This is essential for debugging and threshold tuning.
- **Modularity.** Swapping YOLOv11m for YOLOv12, or InsightFace SCRFD for
  RetinaFace, is a one-line change. The pipeline doesn't care.
- **Beats VLM dependence.** A monolithic VLM would also have been forbidden
  by the assignment, but even if it weren't, this design is far cheaper, more
  auditable, and more controllable.

---

## Pipeline Architecture

```
        ┌─────────────────────────────────────────────────────────────┐
        │                     Raw person crops                        │
        │                  (data/raw/*.png — 1147 imgs)               │
        └───────────────────────────────┬─────────────────────────────┘
                                        │
                              deduplication (perceptual hash)
                                        │
        ┌───────────────────────────────▼─────────────────────────────┐
        │  Stage 0 │ Quality Filter                              cheap│
        │            OpenCV: size, aspect ratio, brightness, blur     │
        └───────────────────────────────┬─────────────────────────────┘
                                        │
        ┌───────────────────────────────▼─────────────────────────────┐
        │  Stage 1 │ Person Detector                                  │
        │            YOLOv11m (COCO class 0), bbox area, count        │
        └───────────────────────────────┬─────────────────────────────┘
                                        │
        ┌───────────────────────────────▼─────────────────────────────┐
        │  Stage 2 │ Full-body Pose Filter                            │
        │            YOLOv11m-pose keypoints — head/torso/legs visible│
        └───────────────────────────────┬─────────────────────────────┘
                                        │
        ┌───────────────────────────────▼─────────────────────────────┐
        │  Stage 3 │ Face Visibility Filter                           │
        │            InsightFace SCRFD: detection conf + landmarks    │
        └───────────────────────────────┬─────────────────────────────┘
                                        │
        ┌───────────────────────────────▼─────────────────────────────┐
        │  Stage 4 │ Age Filter                                       │
        │            MiVOLO v2 — reject crops with predicted age < 16 │
        └───────────────────────────────┬─────────────────────────────┘
                                        │
        ┌───────────────────────────────▼─────────────────────────────┐
        │  Stage 5 │ Ad / Mannequin Filter                       slow │
        │            CLIP ViT-B/32 zero-shot — real-person vs ad probe│
        └───────────────────────────────┬─────────────────────────────┘
                                        │
        ┌───────────────────────────────▼─────────────────────────────┐
        │                       Curated dataset                       │
        │       manifest.json + decisions.parquet + kept/*.png        │
        └─────────────────────────────────────────────────────────────┘
```

Each `StageResult` carries metrics (numbers it measured) and a `rejected_reason`
string. The final `FinalDecision` per crop carries the full per-stage trace,
making every accept/reject decision auditable.

---

## Implementation Status

| Stage / Component | Status | Notes |
|---|---|---|
| EDA |  Done | `notebooks/01_eda.ipynb`, `artifacts/eda/01_eda_metrics.parquet` |
| Stage 0 — Quality Filter |  Done | `src/components/quality_filter.py` + tests |
| Stage 1 — Person Detector |  Done | `src/components/person_detector.py` (YOLOv11m) |
| Stage 2 — Pose Filter |  Done | `src/components/pose_filter.py` (YOLOv11m-pose) |
| Stage 3 — Face Filter | Done | `src/components/face_filter.py` (InsightFace SCRFD) |
| Stage 4 — Age Filter |  In progress | MiVOLO smoke test done in notebook; production component pending |
| Stage 5 — Ad/CLIP Filter |  In progress | Prompts defined in `thresholds.yaml`; component pending |
| End-to-end pipeline |  Done | `src/pipeline/curation_pipeline.py` |
| CLI runner |  Done | `scripts/run_pipeline.py` (Typer) |
| Artifact writer |  Done | `src/services/artifact_writer.py` |
| Evaluation pipeline |  In progress | `src/pipeline/evaluation_pipeline.py` |
| Validation labels |  In progress | `data/labels/validation.csv` |
| Threshold tuning report |  In progress | `notebooks/08_threshold_tuning.ipynb` |

---

## Quick Start

### 1. Prerequisites

- **Python ≥ 3.12**
- **CUDA 12.8** (recommended; CPU also supported via `config.yaml`)
- [**uv**](https://docs.astral.sh/uv/) — fast Python package manager

### 2. Clone and install

```bash
git clone <repo-url>
cd cynapse-ai-assignment

# install mivolo
# Note: This does not get tracked into pyproject.toml
uv pip install --no-deps --no-build-isolation 'mivolo @ git+https://github.com/WildChlamydia/MiVOLO.git'


# Create a virtual env and install all locked dependencies
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock` and installs the exact dependency
graph used during development — including PyTorch built against CUDA 12.8.

### 3. Place the dataset

Download the noisy crops and place them under:

```
data/raw/*.png
```

(See the assignment for the dataset link. The dataset is **not** checked into
git — see `.gitignore`.)

### 4. Run the pipeline

```bash
uv run python scripts/run_pipeline.py
```

That's it. The pipeline will:

- Discover all PNGs under `data/raw/`
- Deduplicate them
- Run all 6 stages
- Write a timestamped run folder to `data/interim/<run_id>/`
- Stream logs to console **and** write a structured log file to `logs/`

---

## Project Structure

```
cynapse-ai-assignment/
│
├── config/
│   ├── config.yaml           # runtime / paths / models / pipeline behaviour
│   └── thresholds.yaml       # all tunable numerical knobs per stage
│
├── data/
│   ├── raw/                  # input crops (gitignored)
│   ├── interim/              # per-run output folders
│   └── labels/               # validation labels (gitignored)
│
├── artifacts/                # cached models, EDA outputs, plots
├── logs/                     # YYYY-MM-DD/HH-MM-SS.log files
│
├── notebooks/                # EDA, per-stage exploration, threshold tuning
│   ├── 01_eda.ipynb
│   ├── 02_stage0_quality.ipynb
│   ├── 03_stage1_person_detection.ipynb
│   ├── 04_stage2_pose_fullbody.ipynb
│   ├── 05_stage3_face_visibility.ipynb
│   ├── 06_stage4_age_filter.ipynb
│   ├── 07_stage5_ad_clip.ipynb
│   ├── 08_threshold_tuning.ipynb
│   └── 09_final_evaluation.ipynb
│
├── src/
│   ├── components/           # one file per stage; all extend BaseFilter
│   │   ├── base.py
│   │   ├── quality_filter.py
│   │   ├── person_detector.py
│   │   ├── pose_filter.py
│   │   ├── face_filter.py
│   │   ├── age_filter.py
│   │   └── ad_filter.py
│   ├── pipeline/             # orchestrators (curation + evaluation)
│   ├── services/             # artifact writing, run manifests
│   ├── entities/             # dataclasses: Crop, StageResult, FinalDecision
│   ├── data/                 # image loading, dedup, integrity
│   ├── utils/                # io, visualization, metrics
│   ├── settings/             # pydantic config models + YAML loaders
│   ├── exceptions/           # custom exception hierarchy
│   ├── logger/               # loguru setup
│   └── constants.py          # project-wide constants (paths, kpt indices)
│
├── scripts/
│   ├── run_pipeline.py       # CLI: run the full cascade
│   ├── evaluate.py           # CLI: evaluate against labelled validation set
│   └── label_helper.py       # CLI helper for assisted labelling
│
├── tests/                    # pytest suites per component
│
├── pyproject.toml            # uv / setuptools config
├── uv.lock                   # locked dependencies
└── README.md
```

**Architectural conventions**

- **`components/`** = pure inference logic per stage. Stateless, swappable.
- **`pipeline/`** = orchestration. Knows about all stages and how to chain them.
- **`services/`** = I/O side-effects (writing parquet, copying files, building manifests).
- **`entities/`** = shared dataclasses moved between layers. No logic.
- **`settings/`** = pydantic models that load and *validate* YAML config.

This separation is critical: it means `components/quality_filter.py` knows
nothing about files, parquet, or even other stages. That's what makes it
unit-testable in 5 lines.

---

## Configuration

The project uses **two YAML files** for configuration, both validated by
pydantic v2 at startup. Invalid config fails fast with a clear error — never
mid-pipeline three hours into a run.

### `config/config.yaml` — application settings

```yaml
runtime:
  device: "cuda"           # or "cpu"
  batch_size: 16
  num_workers: 2

paths:
  data_dir: "data/raw"
  artifacts_dir: "artifacts"
  labels_dir: "data/labels"
  models_cache: "artifacts/models"

models:
  yolo_detection: "yolo11m.pt"
  yolo_pose: "yolo11m-pose.pt"
  face_detector: "buffalo_l"
  age_estimator: "iitolstykh/mivolo_v2"
  clip_model: "openai/clip-vit-base-patch32"

pipeline:
  save_rejected: true
  save_visualizations: true
  decisions_filename: "decisions.parquet"
  manifest_filename: "manifest.json"
```

### `config/thresholds.yaml` — tunable knobs

Every numerical threshold for every stage lives here. **No magic numbers in
code.** This file is what you would tune for a new dataset.

```yaml
quality:
  min_aspect_ratio: 1.3        # h/w — person crops are tall
  min_width: 80
  min_height: 150
  min_brightness: 15.0         # reject near-black
  max_brightness: 240.0        # reject near-white
  min_blur_variance: 10.0      # Laplacian variance — low = blurry

person_detection:
  min_confidence: 0.5
  min_bbox_area_ratio: 0.4     # person area / crop area
  max_persons: 5

pose:
  keypoint_confidence: 0.5
  min_head_kpts: 1             # ≥ 1 of {nose, eyes, ears}
  min_shoulder_kpts: 2
  min_hip_kpts: 2
  min_knee_kpts: 2

face:
  min_detection_confidence: 0.6
  min_face_area_ratio: 0.005
  require_all_landmarks_in_bbox: true

age:
  min_age: 16
  min_gender_confidence: 0.6

clip_ad_filter:
  similarity_margin: 0.0       # real-score must beat ad-score by this margin
  real_person_prompts: [ ... ] # 5 candid-photo descriptions
  ad_prompts:        [ ... ]   # 6 advertisement / mannequin descriptions
```

**Why split into two files?** `config.yaml` answers *"what runs and where"*.
`thresholds.yaml` answers *"how strict is each filter"*. Threshold tuning
should never require touching application config.

---

## Running the Pipeline

### Full run

```bash
uv run python scripts/run_pipeline.py
```

### Useful CLI flags

```bash
# Process only the first 100 crops (smoke test)
uv run python scripts/run_pipeline.py --limit 100

# Custom data directory
uv run python scripts/run_pipeline.py --data-dir path/to/crops

# Skip deduplication
uv run python scripts/run_pipeline.py --skip-dedup

# Verbose console output
uv run python scripts/run_pipeline.py --log-level DEBUG
```

Every run gets a unique `run_id` of the form `YYYY-MM-DD_HHMMSS`. Multiple runs
in the same day never collide.

---

## Outputs & Run Artifacts

After every run, a self-contained run folder is written to
`data/interim/<run_id>/`:

```
data/interim/2026-05-11_143022/
├── manifest.json              # run metadata: config snapshot, counts, timing
├── decisions.parquet          # one row per crop with full per-stage trace
├── run.log                    # archived copy of the run's log file
├── kept/                      # symlinks/copies of accepted crops
│   ├── crop_001.png
│   └── ...
└── samples/
    ├── rejected_at_quality/   # representative rejected samples per stage
    ├── rejected_at_person_detection/
    ├── rejected_at_pose/
    ├── rejected_at_face/
    ├── rejected_at_age/
    └── rejected_at_ad/
```

**Why this layout?**

- `manifest.json` makes every run reproducible — it snapshots the exact
  thresholds used.
- `decisions.parquet` is a tabular trace: pandas can load it instantly and
  every metric measured by every stage is queryable.
- `samples/rejected_at_<stage>/` lets you eyeball *why* the pipeline rejected
  things. This is the single most useful debugging artifact.

---

## Logging

We use [`loguru`](https://github.com/Delgan/loguru) for all logging — never
`print()`.

```
logs/
└── 2026-05-11/
    ├── 14-30-22.log
    └── 16-04-51.log
```

- Console: colored, INFO-level by default.
- File: structured, DEBUG-level, 50 MB rotation, 30-day retention.
- Every run also archives its log into `data/interim/<run_id>/run.log`.

Logged events include: stage start/end, per-stage counts, threshold values
used, decisions taken, timing per image, warnings, and full tracebacks on
errors.

---

## Evaluation

> 🚧 In progress.

The evaluation pipeline computes assignment-level metrics against a manually
labelled validation slice (`data/labels/validation.csv`).

**Metrics**

- **Coverage** — fraction of *acceptable* crops the pipeline keeps.
- **Precision** — fraction of kept crops that are actually acceptable.
- **Per-requirement recall / precision** — full-body, face visible,
  non-advertisement, age ≥ 16.
- **Per-stage funnel** — how many crops survive each stage.
- **Latency** — ms/image per stage and end-to-end.

```bash
uv run python scripts/evaluate.py --run-id 2026-05-11_143022
```

---

## Design Decisions & Tradeoffs

### 1. Why a cascade rather than parallel filters?

A cascade short-circuits as soon as any requirement fails. A parallel design
would run *all* expensive models on every crop — wasteful, since ~30 % of
crops fail Stage 0 (cheap OpenCV checks) alone.

### 2. Why YOLOv11m for both detection and pose?

YOLOv11m is fast, accurate, and ships with a pose variant that uses the same
backbone. Loading two YOLO models has minor memory cost but identical
preprocessing — keeping the codebase simple.

### 3. Why InsightFace SCRFD for faces (not the YOLO pose head)?

The pose model's face keypoints (eyes, nose, ears) are noisy and miss small
faces. SCRFD is purpose-built for face detection and provides a face bounding
box + 5 landmarks, which gives us a much more reliable face-visibility signal.

### 4. Why MiVOLO for age?

MiVOLO is purpose-built for age estimation from full-body crops and handles
side-view / occluded faces better than face-only age models. The assignment
forbids VLMs, so we can't ask a VLM "is this a child?".

### 5. Why zero-shot CLIP for ads/mannequins?

We have no labelled ad/non-ad data. Training a binary classifier would require
labelling. CLIP gives us a strong zero-shot prior using prompt engineering —
no labelling, no fine-tuning, and prompts are easy to inspect and tune. The
similarity-margin threshold gives a single tunable knob.

### 6. Why pydantic-validated YAML configs?

Pydantic catches typos and out-of-range values at startup, not three hours
into a run. Example: `device: "gtx1080"` is rejected immediately with a clear
message rather than producing a cryptic CUDA error during inference.

### 7. Why a `BaseFilter` ABC?

It enforces a single contract — `apply(crop) -> StageResult` — across all
stages. The pipeline doesn't care if the filter is OpenCV-based or
PyTorch-based; it just calls `apply`. This is the Strategy pattern.

### 8. Why deduplication?

The raw set contains exact and near-duplicate crops. Without dedup, identical
crops would inflate every count and bias evaluation.

---

## Results

> 🚧 Final metrics depend on the validation set being labelled and the
> evaluation pipeline being wired up.

**Per-stage funnel** observed during development on the full 1,147-crop set:

| Checkpoint | Surviving crops |
|---|---:|
| Loadable images | 1,147 |
| After deduplication | ~1,001 |
| After Stage 0 (Quality) | 784 |
| After Stage 1 (Person) | 461 |
| After Stage 2 (Pose) | 208 |
| After Stage 3 (Face) | 124 |
| After Stage 4 (Age) | *pending* |
| After Stage 5 (Ad/CLIP) | *pending* |

The funnel shape — biggest drops at pose and face — matches the assignment's
strictness on the full-body and face-visibility requirements.

---

## Limitations & Future Work

- **Validation set is small.** A larger labelled slice would tighten metric
  confidence intervals.
- **CLIP ad-filter is prompt-engineered, not benchmarked.** Adding a small
  labelled ad subset would let us pick optimal prompts and a calibrated
  `similarity_margin`.
- **Single-GPU inference.** Multi-GPU batching is left to a follow-up — the
  cascade design makes this a per-stage optimisation, not an architectural
  change.
- **Pose-based full-body check** uses keypoint presence. A learned full-body
  classifier conditioned on keypoints would likely be more robust.
- **MiVOLO age boundary at 16** has predictable error near the threshold.
  Calibration on a held-out age set would improve precision/recall trade-off.
- **No active learning loop.** A real production system would route uncertain
  decisions (near-threshold cases) to a labelling queue for review.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Modern typing, performance |
| Package manager | uv | 10–100× faster than pip; deterministic lockfile |
| Deep learning | PyTorch 2.11 (CUDA 12.8) | Deep Learning framework |
| Detection / Pose | Ultralytics YOLOv11 | Best speed/accuracy at this scale |
| Face | InsightFace (SCRFD) | Production-grade face detector |
| Transformers | MiVOLOv2,openai/clip-vit-base-patch32 | Reproducible CLIP weights and MiVOLO |
| Config | pydantic v2 + PyYAML | Typed validation, fail-fast |
| CLI | Typer + Rich | Type-driven CLI, pretty output |
| Logging | loguru | Zero-config, rotating, structured |
| Data | pandas + pyarrow | Parquet I/O, fast filtering |
| Imaging | opencv-python-headless + Pillow | Headless-friendly for servers |
| Visualisation | matplotlib | Plotting Data Distribution and images |

---

## License

This project is part of a private take-home assignment and is not currently
licensed for redistribution.