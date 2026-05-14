# Person Dataset Curation Pipeline

## 1. Introduction

Modern AI applications such as surveillance, retail analytics, and autonomous systems rely on high-quality person datasets for accurate computer vision model training. However, large-scale datasets are often noisy and contain incomplete body crops, invisible faces, advertisement images, and unsuitable age groups. Reviewing each crop by hand is slow, expensive, and inconsistent between annotators, and does not scale to the dataset sizes that modern
training pipelines require.

This project builds an automated curation pipeline for person body crops. The
pipeline keeps an image only when it satisfies all four target requirements:

- the crop contains a **full-body** person,
- the **face is visible** from the front or side profile view,
- the crop shows a **real person** rather than an advertisement or mannequin,
- the person is a **teenager or adult**, not a young child.


## 2. Problem Definition and Algorithm

### 2.1 Task Definition

**Input.** A directory of noisy person-crop images.

**Output.** Two groups of images and a structured decision record per crop:

- `kept` — images that satisfy all four requirements,
- `rejected` — images that are removed by the first stage that failed.

Each decision record contains an image id `(crop_id)`, the final keep/reject
verdict, the stage that produced a rejection (if any), the numeric metrics
computed at every stage the crop visited, and per-stage timing. These records
are written to a Parquet file so the full run is auditable after the fact.

The primary evaluation metric is **coverage**, also referred to as the recall
on the reject class. Coverage answers a single question:

> For all the images that should be removed, what fraction did the pipeline
> successfully remove?

For a worked example, suppose a validation set contains 100 invalid images.
If the pipeline rejects 95 of them and lets 5 through, coverage is:

```text
coverage = invalid crops correctly rejected / all invalid crops
         = 95 / 100
         = 95%
```

Coverage is preferred over plain accuracy for this project because the two
error types differ sharply in cost. Letting an invalid crop reach the curated
dataset can degrade any downstream model trained on it, while accidentally
rejecting a valid crop is cheap to recover from when raw input is plentiful.
Coverage measures the costly error class directly.

### 2.2 Algorithm Definition

The pipeline is designed as a sequential filtering system, where simple and computationally cheap checks run first, followed by using deep-learning models for filtering only when necessary. This early-exit design improves efficiency because invalid crops are discarded as soon as a failure condition is detected.

![Pipeline Workflow](images/project-workflow.png)

#### Design principles

Three principles drive every choice in this section.

**Cheap before expensive.** The cheapest checks come first. The first quality check by 
OpenCV and runs in well under a millisecond per crop, removing 21.7% of
inputs (217 of 1,001) before any deep learning model is loaded. The expensive
deep-learning models then run only on the survivors.

**One stage, one requirement.** Each stage maps to exactly one of the four
target requirements (plus the two upstream sanity checks). This is why every
rejected crop has a single `rejected_at_stage` value, and it is what makes
the per-violation analysis in Section 3 possible. A fused single-score
classifier would offer no equivalent breakdown.

**Specialised models for specialised checks.** Face detection, age
estimation, and advertisement classification each have a model already
trained on the right distribution. Using them directly is cheaper, more
accurate, and more interpretable than fine-tuning a single multi-task model
on a small custom dataset.

#### Stage details

**1. Quality (OpenCV heuristics).** This stage removes visually unusable images .

  - Minimum size checks `(min_width=80, min_height=150` ensure the image contains enough visual detail.
  - Aspect ratio threshold `(min_aspect_ratio=1.3)` ensures the crop roughly matches the shape of a standing human body.
  - Brightness thresholds `(15–240)` reject extremely dark or overexposed images.
  - Blur detection `(min_blur_variance=10)` uses Laplacian variance to detect blurry or out-of-focus images.

**2. Person detection (YOLO11m).** This stage verifies whether image contains a clear visible person. It removes images, without persons, crowded scenes containing multiple overlapping people. The detector applies:
  - Person confidence threshold `(min_confidence=0.5)` ensures that the detection is reliable. 
  - Bounding box area `(min_bbox_area=0.4)` threshold ensures the detected person occupies atleast 40% space in crop.
  - Maximum person limit `(max_persons=5)` rejects crowded scenes with many overlapping people. 
  

**3. Full-body check (YOLO11m-pose).** This stage ensures the person’s full body is visible using pose keypoints.
  - Keypoint confidence threshold `(keypoint_confidence=0.5)` ensures detected joints are reliable.
  - Requires at least:
    - 1 visible head keypoint `(min_head_kpts=1)`
    - 2 shoulder keypoints `(min_shoulder_kpts=2)`
    - 2 hip keypoints `(min_hip_kpts=2)`
    - 2 knee keypoints `(min_knee_kpts=2)`
  - Images missing important body regions are rejected as partial-body crops.

**4. Face visibility (SCRFD, via InsightFace).** This stage verifies that a clear and visible face is present.

  - Face detection confidence `(min_detection_confidence=0.6)` ensures reliable face detection.
  - Minimum face area ratio `(min_face_area_ratio=0.005)` ensures the face is large enough for analysis.
  - Landmark validation `(require_all_landmarks_in_bbox=true)` ensures all facial landmarks lie within the detected face box.

  - Images with hidden, occluded, or extremely small faces are rejected.

**5. Age filter (MiVOLOv2).** This stage removes images predicted to contain young children.

  - Minimum age threshold `(min_age=16)` rejects crops predicted to depict minors.
  - Gender-confidence threshold `(min_gender_confidence=0.6)` acts as a reliability check for the model prediction.
  - Uses both facial and body information for more robust age estimation.

**6. Advertisement filter (openai/clip-vit-base-patch32, zero-shot).** This stage removes advertisements, mannequins, and studio-style promotional images.

  - CLIP compares the image against:
    - Real-person prompts (e.g. candid street photos, casual human photographs)

    - Advertisement prompts (e.g. mannequins, fashion shoots, retail displays)

  - If advertisement similarity exceeds real-person similarity (similarity_margin=0.0), the image is rejected.

  - This zero-shot approach allows semantic filtering without training a custom classifier.



#### Two design choices worth flagging

**Why two YOLO models, not just YOLO11m-pose?** YOLO11m-pose is itself a
person detector, so using it for both stages would reduce code. The stages
are kept separate because they answer different questions. YOLO11m cheaply
removes no-person and tiny-person crops before any pose computation is
needed; YOLO11m-pose then focuses only on the harder question of body
visibility. This separation is what allows the per-violation breakdown in
Section 3 to distinguish "no person" failures from "partial body" failures.

**Why MD5, not perceptual hashing, for deduplication?** The raw dataset
contained 1,147 files but only 1,001 unique ones — 146 redundant copies.
Exact-byte MD5 hashing catches these. Perceptual hashing would also collapse
visually-similar but distinct photographs, which is not what we want — two
different photos of the same person are still distinct training examples.

#### How the pipeline works

- A dark crop is rejected at the quality stage and never reaches the next stage which is YOLO detection. A crop with no detectable person is rejected by YOLO11m and never reaches
pose or age estimation. 

- A high-quality, full-body crop with a visible face
reaches MiVOLOv2 and CLIP, because only at that point are age and
advertisement checks meaningful.

## 3. Experimental Evaluation

### 3.1 Methodology

All results below come from run `2026-05-12_204240`.

| Item                          | Value |
| ----------------------------- | ----: |
| Unique pipeline decisions     | 1,001 |
| Labelled validation crops     |   207 |
| Labels without decisions      |     0 |
| Kept crops in the full run    |   117 |
| End-to-end runtime            |  ~50s |

The 207 hand-labelled crops were compared against the pipeline's decisions
for the same crop IDs. Each label captures whether the crop should be kept,
and if not, which of six violation categories applies (`blurry`, `no_person`,
`not_full_body`, `face_hidden`, `minor`, `advertisement`). Schema validation
at load time prevents inconsistent labels (e.g. `should_keep=True` paired
with a violation reason) from entering the evaluation.

The validation set contains 183 crops that should be rejected and 24 crops
that should be kept. As noted in Section 2.1, the dominant error to avoid is
an invalid crop leaking into the kept set, so coverage on the reject class
is the primary metric.

For this run, coverage is:

```text
coverage = correctly rejected invalid crops / all labelled invalid crops
         = 177 / (177 + 6)
         = 96.72%
```

Five secondary metrics are also reported for completeness:

| Metric                       | Meaning in this project                                                          |
| ---------------------------- | -------------------------------------------------------------------------------- |
| Accuracy                     | Fraction of labelled crops classified correctly overall.                          |
| Precision (keep)             | Of crops the pipeline kept, the fraction that were actually valid.                |
| Recall (keep)                | Of valid crops, the fraction the pipeline successfully kept.                      |
| F1 (keep)                    | Harmonic mean of keep-precision and keep-recall.                                  |
| **Coverage (reject recall)** | **Primary metric. Of invalid crops, the fraction the pipeline rejected.**         |

### 3.2 Results

![Overall metrics](artifacts/runs/2026-05-12_204240/final_evaluation_plots/overall_metrics.png)

| Metric                       | Value      |
| ---------------------------- | ---------- |
| Accuracy                     | 92.75%     |
| Precision (keep)             | 71.43%     |
| Recall (keep)                | 62.50%     |
| F1 (keep)                    | 66.67%     |
| **Coverage (reject recall)** | **96.72%** |

![Confusion matrix](artifacts/runs/2026-05-12_204240/final_evaluation_plots/confusion_matrix.png)

| Outcome                                 | Count |
| --------------------------------------- | ----: |
| True positives (correctly kept)         |    15 |
| False negatives (valid crop rejected)   |     9 |
| True negatives (correctly rejected)     |   177 |
| False positives (invalid crop leaked)   |     6 |

![Per-violation coverage](artifacts/runs/2026-05-12_204240/final_evaluation_plots/per_violation_coverage.png)

| Violation type   | Labelled | Rejected | Coverage  |
| ---------------- | -------: | -------: | --------: |
| `blurry`         |       26 |       26 |   100.00% |
| `no_person`      |       28 |       28 |   100.00% |
| `advertisement`  |       40 |       40 |   100.00% |
| `not_full_body`  |       49 |       48 |    97.96% |
| `face_hidden`    |       32 |       29 |    90.62% |
| `minor`          |        8 |        6 |    75.00% |

### 3.3 Discussion

The headline result supports the central hypothesis of the project: a
staged cascade of pretrained CV models, with no VLM and no model training,
can remove the large majority of invalid person crops while remaining fast
and interpretable. Five of the six violation categories sit at or above the
90% target, and four reach 100% on this validation set.

The system is also intentionally **conservative**. The confusion matrix
shows that 9 of the 24 valid crops are incorrectly rejected (a keep-recall
of 62.5%), while only 6 of the 183 invalid crops leak through. This
asymmetry is the right trade-off for curation: the cost of a single
mannequin or minor entering a downstream training set is high, while a
discarded valid crop is cheap to replace from the larger raw pool.

**The weakest category is `minor`, at 75% coverage.** Two of the eight
labelled minors slip through. Three factors contribute. First, MiVOLOv2's
accuracy degrades on side-profile and small-face crops, several of which
appear in the labelled minor subset. Second, the threshold (16 years) sits
at the teenager–adult boundary, the region where age-estimation uncertainty
is largest. Tightening the threshold to 18 would catch additional minors at
the direct cost of false-rejecting young adults — a real trade-off, not a
free improvement. Third, with only 8 labelled examples in this category,
the 75% figure has a wide confidence interval and should not be
over-interpreted as a precise estimate.

**The `advertisement` category catches the eye for a different reason.**
Coverage is 100%, but a per-stage breakdown of where those rejections occur
(available in `evaluation.json`) shows that only 2.5% of labelled
advertisements are caught at the dedicated CLIP stage. The remaining 97.5%
are rejected earlier — most by person detection, because mannequins fail to
register as confident persons, and by pose, because many advertisements are
mid-body crops. This is not redundancy: the small fraction CLIP does catch
are studio-quality ad images that successfully pass every earlier check,
and removing the CLIP stage would let those through. The current results
also imply that on a dataset of cleaner advertisements (editorial
photography of real models, for example), the CLIP stage would become the
primary defence rather than a backstop.

## 4. Limitations and Future Work

Listed in roughly decreasing order of expected impact.

- **Strengthen age filtering** — the weakest stage. Two practical paths:
  (a) gate MiVOLOv2 behind a face-quality check so it only runs on
  reliable frontal faces; or (b) ensemble MiVOLOv2 with a second age model
  and reject only when both agree. Either would primarily address the
  side-profile failure mode identified in Section 3.3 and should narrow the
  gap on the `minor` category.

- **Move from per-crop to batched inference.** YOLO, SCRFD, MiVOLOv2, and
  CLIP all support batched inputs. Batching crops in groups of 16–32 would
  amortise GPU launch overhead and is expected to roughly double end-to-end
  throughput. The change is localised to the filter classes and the
  pipeline orchestrator.

- **Re-evaluate the CLIP advertisement filter in isolation.** Because most
  labelled ads in this dataset are caught earlier in the cascade, the
  current results understate CLIP's contribution. A controlled ablation
  that bypasses earlier stages for the ad-labelled subset would yield a
  true per-stage reading and would directly inform whether the prompt bank
  needs revision.

- **Add a human-in-the-loop queue for borderline crops.** Crops scoring
  within a small margin of any threshold (e.g. age 16 ± 1 year; CLIP score
  within 0.02 of the decision boundary) are the cases where a few seconds
  of human review would add the most signal per minute. The current
  pipeline makes a hard decision on every crop.

- **Validate threshold generalisation on a second dataset.** Every numeric
  threshold was tuned against this single noisy dataset. The
  keypoint-visibility logic and prompt design were chosen to generalise
  rather than overfit, but generalisation remains an empirical claim until
  tested on independent data.

- **Strengthen ground truth.** The 207 validation labels were authored by a
  single annotator. A production deployment should at minimum use
  double-annotation with adjudication for disagreements, particularly for
  borderline categories (`minor`, mild `face_hidden`, mild blur) where
  inter-rater disagreement is most likely.

## 5. Conclusion

This project provides an automated curation pipeline for noisy person-crop
datasets. On a 1,001-crop run, the pipeline keeps 117 crops in ~50 seconds
on a single consumer GPU and reaches **96.72% coverage** against a
hand-labelled validation set of 207 crops, clearing the 90% target with
margin. The core design strength is the early-exit cascade: cheap heuristic
checks remove obvious failures before any GPU model runs, while specialised
pretrained models handle person detection, pose, face visibility, age, and
advertisement filtering in turn. The most pressing weakness is age
filtering on side profiles, which Section 5 outlines a concrete path to
address.
