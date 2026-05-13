# Person-Crop Dataset Curation Pipeline

This project demonstrates multi-stage computer-vision pipeline that filters a noisy person-crop dataset
down to crops suitable for training. The passing criteria for images is that the image must contain a full-body person crop, with visible face (frontal or side view). The images that contain partial body crops, advertisement/mannequin, minors are automatically filtered out.

## Headline result

Evaluated on a hand-labelled validation subset of **207 crops** sampled from a
1,001-image run:

| Metric                              | Value      |
| ----------------------------------- | ---------- |
| Coverage (recall on rejects)        | **96.72%** |
| Assignment target                   | > 90%      |
| End-to-end runtime (1,001 crops)    | ~50 s on a single CUDA GPU |

A detailed methodology, per-violation breakdown, and discussion of design
decisions live in [`REPORT.md`](./REPORT.md).

`Note`: I have excluded duplicate images during the complete pipeline run and evaluation.

## Setup

Requires Python 3.12 and a CUDA-capable GPU.
Dependencies are managed with [`uv`](https://docs.astral.sh/uv/).

`Note`: This setup will work on CPU also but it will be slower.

```bash
# Clone and enter the repo
git clone https://github.com/jsuryanm/dataset-curation-pipeline.git
cd dataset-curation-pipeline

# Install dependencies into a project-local virtualenv
uv sync

# Install mivolo as this is not tracked not pyproject.toml
uv pip install --no-deps --no-build-isolation "mivolo @ git+https://github.com/WildChlamydia/MiVOLO.git"


# Place the noisy person dataset at:
#   data/raw/*.png
```

The three deep-learning model families (Ultralytics YOLO, InsightFace SCRFD, HuggingFace
Transformers CLIP and MiVOLO) download their own weights on first run into `artifacts/models/`
or each library's default cache. No manual download is required.

## How to run

The pipeline is split into two CLI entry points so curation and evaluation can
be re-run independently of each other.

### 1. Run the curation cascade

```bash
python scripts/run_pipeline.py
```

Useful flags:

```bash
--data-dir PATH        # override the input directory from config.yaml
--limit N              # process only the first N crops (smoke-test)
--skip-dedup           # skip MD5 deduplication of input files
--log-level INFO       # console verbosity
```

Each run produces a timestamped folder under `artifacts/runs/`:

```
artifacts/runs/2026-05-12_204240/
├── manifest.json            # slim per-run summary (counts, stage stats, timing)
├── config_snapshot.json     # exact thresholds used for this run
├── environment.json         # python, torch, library versions
├── dedup_groups.json        # MD5 hash groups (full duplicate listing)
├── decisions.parquet        # one row per crop with stage-by-stage metrics
├── run.log                  # archived copy of the run log
├── kept/                    # stores all images that pass the all stages the pipeline  
└── samples/rejected_at_<stage>/   # 20 random rejects per stage
```

### 2. Evaluate a run against ground-truth labels

```bash
python scripts/evaluate.py
```

By default this evaluates the most recent run folder against
`data/labels/validation.csv`. Override either with:

```bash
--run-folder PATH      # evaluate a specific run instead of the latest
--labels PATH          # use a different labels CSV
```

The evaluator writes `evaluation.json` into the run folder and prints a Rich
table to stdout with overall metrics and per-violation coverage.

### 3. Generate the submission plots

The notebook `notebooks/08_final_evaluation.ipynb` re-runs the evaluator,
loads `evaluation.json`, and saves four plots into
`<run_folder>/final_evaluation_plots/`:

- `overall_metrics.png`
- `confusion_matrix.png`
- `per_violation_coverage.png`
- `violation_stage_breakdown.png`

These same plots are embedded in `REPORT.md`.

## Project layout

```
.
├── config/
│   ├── config.yaml           # paths, runtime, model identifiers
│   └── thresholds.yaml       # all tunable numerical thresholds
├── data/
│   ├── raw/                  # input crops (*.png)
│   └── labels/validation.csv # hand-labelled ground truth
├── scripts/
│   ├── run_pipeline.py       # CLI: curation cascade
│   └── evaluate.py           # CLI: evaluate a run vs. labels
├── notebooks/                # Experiment run for each component 
│ 
├── src/
│   ├── components/           # one BaseFilter subclass per pipeline stage
│   ├── pipeline/             # cascade orchestration + evaluation logic
│   ├── services/             # artifact + manifest writers
│   ├── settings/             # pydantic config loaders
│   ├── entities/             # Crop, StageResult, FinalDecision, ValidationLabel
│   ├── exceptions/           # custom exception hierarchy
│   ├── data/                 # label loading, deduplication
│   ├── logger/               # loguru configuration
│   └── utils/                # image I/O
├── artifacts/runs/           # per-run output folders (generated)
├── logs/YYYY-MM-DD/          # timestamped run logs (generated)
├── REPORT.md                 # design rationale, methodology, results
└── README.md
```

## Configuration

Two YAML files, both validated by pydantic v2 schemas at load time so bad keys
or types fail loudly before any model is loaded:

- `config/config.yaml` — paths, device,model identifiers, runtime settings.
  Mapped to `AppConfig` in `src/settings/config.py`.
- `config/thresholds.yaml` — every numerical knob across the six stages.
  Mapped to `Thresholds` in the same module.

Changing a threshold never requires a code edit. A copy of the thresholds
actually used by each run is written to `config_snapshot.json` so any run is
exactly reproducible from its own artifacts.

## Logs

Whenever a python script is run, it creates a fresh timestamped log file at
`logs/YYYY-MM-DD/HH-MM-SS.log`. The log records every stage decision per image,
whether image passes or fails stage,the reason for failure/passing, and the date-time, which is what made debugging threshold
choices and stage ordering tractable. The active log file is also archived
into the run folder inside `artifacts/runs/YYYY-mm-dd_hhmmss/runs.log`.