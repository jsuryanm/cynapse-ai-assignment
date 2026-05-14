from __future__ import annotations 

from collections import defaultdict,Counter 
from dataclasses import dataclass 
from pathlib import Path 
from typing import Any 

import pandas as pd 
from sklearn.metrics import confusion_matrix as sk_confusion_matrix 

from src.entities.validation_label import ValidationLabel,ViolationReason
from src.logger.custom_logger import logger
from src.exceptions.custom_exceptions import ValidationError

VIOLATION_TO_EXPECTED_STAGE: dict[ViolationReason, str] = {ViolationReason.BLURRY: "quality",
                                                           ViolationReason.NO_PERSON: "person_detection",
                                                           ViolationReason.NOT_FULL_BODY: "pose",
                                                           ViolationReason.FACE_HIDDEN: "face",
                                                           ViolationReason.MINOR: "age",
                                                           ViolationReason.ADVERTISEMENT: "ad_filter"}

@dataclass(frozen=True,slots=True)
class ConfusionMatrix:
    tp: int 
    fp: int 
    tn: int 
    fn: int 

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn
    
    @property
    def accuracy(self) -> float:
        """Ratio of correct predictions"""
        return (self.tp + self.tn) / self.total if self.total else 0.0
    
    @property
    def precision_keep(self) -> float:
        """Ratio tp predictions that were actually correct"""
        denom = self.tp + self.fp 
        return self.tp / denom if denom else 0.0 
    
    @property
    def recall_keep(self) -> float:
        """How many tp were correctly identified by the model"""
        denom = self.tp + self.fn 
        return self.tp / denom if denom else 0.0
    
    @property
    def f1_keep(self) -> float: 
        p,r = self.precision_keep,self.recall_keep
        return 2*p*r / (p + r) if (p + r) else 0.0 
    
    @property
    def coverage(self) -> float:
        """Of bad crops in the validation set, what fraction did we reject.
        This is recall-on-the-reject-class"""
        denom = self.tn + self.fp
        return self.tn / denom if denom else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn},
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "precision_keep": round(self.precision_keep, 4),
            "recall_keep": round(self.recall_keep, 4),
            "f1_keep": round(self.f1_keep, 4),
            "coverage": round(self.coverage, 4),
        }

@dataclass(frozen=True, slots=True)
class ViolationCoverage:
    """Per-violation diagnostics."""

    violation: str
    n_labeled: int
    n_rejected: int
    n_rejected_at_expected_stage: int
    stage_breakdown: dict[str, int]

    @property
    def coverage(self) -> float:
        return self.n_rejected / self.n_labeled if self.n_labeled else 0.0

    @property
    def coverage_at_expected_stage(self) -> float:
        if not self.n_labeled:
            return 0.0
        return self.n_rejected_at_expected_stage / self.n_labeled

    def to_dict(self) -> dict[str, Any]:
        return {"violation": self.violation,
                "n_labeled": self.n_labeled,
                "n_rejected": self.n_rejected,
                "coverage": round(self.coverage, 4),
                "n_rejected_at_expected_stage": self.n_rejected_at_expected_stage,
                "coverage_at_expected_stage": round(self.coverage_at_expected_stage, 4),
                "stage_breakdown": self.stage_breakdown}


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Complete output of one evaluation pass."""

    n_labels: int
    n_decisions: int
    n_evaluated: int
    n_unmatched_labels: int
    unmatched_label_ids: list[str]
    confusion: ConfusionMatrix
    violations: list[ViolationCoverage]

    def to_dict(self) -> dict[str, Any]:
        return {"n_labels": self.n_labels,
                "n_decisions": self.n_decisions,
                "n_evaluated": self.n_evaluated,
                "n_unmatched_labels": self.n_unmatched_labels,
                "unmatched_label_ids": self.unmatched_label_ids,
                "overall": self.confusion.to_dict(),
                "violations": [v.to_dict() for v in self.violations]}


def evaluate_run(decisions_path: Path,
                 labels: list[ValidationLabel]) -> EvaluationResult:
    """Compare pipeline decisions against ground-truth labels.
    Only labeled crops contribute to the metrics. Unmatched labels are
    reported separately.
    """
    if not decisions_path.exists():
        raise ValidationError(f"Decisions file not found: {decisions_path}")

    df = pd.read_parquet(decisions_path)
    required = {"crop_id", "kept", "rejected_at_stage"}
    missing = required - set(df.columns)
    if missing:
        raise ValidationError(
            f"{decisions_path.name} missing required columns: {sorted(missing)}"
        )

    logger.info("Evaluating: {} decisions vs. {} labels", len(df), len(labels))

    decisions_by_id: dict[str, dict[str, Any]] = {
        row["crop_id"]: {
            "kept": bool(row["kept"]),
            "rejected_at": row["rejected_at_stage"],
        }
        for row in df.to_dict(orient="records")
    }

    matched: list[tuple[ValidationLabel, dict[str, Any]]] = []
    unmatched_ids: list[str] = []
    for label in labels:
        decision = decisions_by_id.get(label.crop_id)
        if decision is None:
            unmatched_ids.append(label.crop_id)
        else:
            matched.append((label, decision))

    if unmatched_ids:
        logger.warning(
            "{} labeled crops have no matching decision (likely removed by dedup "
            "or absent from data_dir). First few: {}",
            len(unmatched_ids),
            unmatched_ids[:5],
        )

    confusion = _confusion_matrix(matched)
    violations = _per_violation_coverage(matched)

    return EvaluationResult(
        n_labels=len(labels),
        n_decisions=len(df),
        n_evaluated=len(matched),
        n_unmatched_labels=len(unmatched_ids),
        unmatched_label_ids=unmatched_ids,
        confusion=confusion,
        violations=violations,
    )


def _confusion_matrix(
    matched: list[tuple[ValidationLabel, dict[str, Any]]],
) -> ConfusionMatrix:
    """Computes the confusion matrix between ground-truth labels and pipeline predictions.
    Measures how well the curation pipeline keeps valid crops and rejects invalid ones
    by calculating TP, FP, TN, and FN statistics used for evaluation metrics."""
    if not matched:
        return ConfusionMatrix(tp=0, fp=0, tn=0, fn=0)

    y_true = [label.should_keep for label, _ in matched]
    y_pred = [decision["kept"] for _, decision in matched]

    cm = sk_confusion_matrix(y_true, y_pred, labels=[True, False])
    tp, fn = int(cm[0, 0]), int(cm[0, 1])
    fp, tn = int(cm[1, 0]), int(cm[1, 1])
    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)


def _per_violation_coverage(
    matched: list[tuple[ValidationLabel, dict[str, Any]]],
) -> list[ViolationCoverage]:
    """Computes rejection coverage separately for each violation category in the dataset.
    Tracks whether invalid crops were rejected and whether they failed at the expected
    pipeline stage, helping diagnose weak filters and stage-level failure patterns."""
    
    by_violation: dict[ViolationReason, list[dict[str, Any]]] = defaultdict(list)
    for label, decision in matched:
        if label.violation_reason is not None:
            by_violation[label.violation_reason].append(decision)

    coverages: list[ViolationCoverage] = []
    for violation, decisions in by_violation.items():
        n_labeled = len(decisions)
        n_rejected = sum(1 for d in decisions if not d["kept"])
        expected_stage = VIOLATION_TO_EXPECTED_STAGE[violation]
        n_at_expected = sum(
            1 for d in decisions
            if not d["kept"] and d["rejected_at"] == expected_stage
        )
        stage_breakdown = Counter(
            d["rejected_at"] for d in decisions if not d["kept"]
        )
        coverages.append(
            ViolationCoverage(
                violation=violation.value,
                n_labeled=n_labeled,
                n_rejected=n_rejected,
                n_rejected_at_expected_stage=n_at_expected,
                stage_breakdown=dict(stage_breakdown),
            )
        )

    coverages.sort(key=lambda v: v.coverage, reverse=True)
    return coverages




    
    