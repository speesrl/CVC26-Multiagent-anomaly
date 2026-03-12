"""
Metrics utilities for surveillance anomaly detection experiments.

This module is intentionally lightweight but matches the API expected by
the evaluation scripts in this repository:

    from metrics import evaluate_anomaly_detection, EvaluationResult, plot_roc_curve

It provides:
  - EvaluationResult: dataclass with all key metrics
  - evaluate_anomaly_detection: computes frame- and video-level metrics
  - plot_roc_curve: optional ROC plot helper (only used if matplotlib is installed)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_fscore_support,
)


@dataclass
class EvaluationResult:
    frame_auc: float
    video_auc: float
    average_precision: float
    f1_at_optimal: float
    accuracy_at_optimal: float
    optimal_threshold: float
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray


def _video_level_scores(
    scores: np.ndarray, labels: np.ndarray, video_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate frame-level scores to video-level scores by taking the
    maximum score per video, which is the standard protocol for
    UCF-Crime-style benchmarks.
    """
    unique_vids = np.unique(video_ids)
    vid_scores = []
    vid_labels = []

    for vid in unique_vids:
        mask = video_ids == vid
        vid_scores.append(scores[mask].max())
        # A video is anomalous if any frame is anomalous
        vid_labels.append(int(labels[mask].max()))

    return np.asarray(vid_scores, dtype=float), np.asarray(vid_labels, dtype=int)


def evaluate_anomaly_detection(
    scores: Sequence[float],
    labels: Sequence[int],
    video_ids: Optional[Sequence[int]] = None,
) -> EvaluationResult:
    """
    Compute standard anomaly-detection metrics for frame-level scores.

    Args:
        scores: anomaly scores per frame (higher = more anomalous)
        labels: binary ground-truth labels per frame (0 = normal, 1 = anomaly)
        video_ids: optional integer video IDs per frame; if provided,
                   video-level AUC is computed by max-pooling scores.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    assert scores.shape == labels.shape, "scores and labels must have same shape"
    assert scores.ndim == 1, "scores and labels must be 1D arrays"

    # Frame-level ROC / AUC
    frame_auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Average precision (PR AUC)
    average_precision = (
        average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0
    )

    # Choose optimal threshold with Youden's J statistic (TPR - FPR)
    j_scores = tpr - fpr
    j_idx = int(np.argmax(j_scores))
    optimal_threshold = float(thresholds[j_idx])

    # Binarise scores at optimal threshold and compute F1 / accuracy
    y_pred = (scores >= optimal_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, y_pred, average="binary", zero_division=0
    )
    accuracy = float((y_pred == labels).mean())

    # Video-level AUC (optional)
    if video_ids is not None:
        video_ids = np.asarray(video_ids, dtype=int)
        assert video_ids.shape == scores.shape, "video_ids must align with scores"
        vid_scores, vid_labels = _video_level_scores(scores, labels, video_ids)
        if len(np.unique(vid_labels)) > 1:
            video_auc = roc_auc_score(vid_labels, vid_scores)
        else:
            video_auc = 0.0
    else:
        # Fallback: treat frame-level AUC as video-level AUC when no IDs are given
        video_auc = frame_auc

    return EvaluationResult(
        frame_auc=float(frame_auc),
        video_auc=float(video_auc),
        average_precision=float(average_precision),
        f1_at_optimal=float(f1),
        accuracy_at_optimal=accuracy,
        optimal_threshold=optimal_threshold,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
    )


def plot_roc_curve(result: EvaluationResult, title: str = "ROC Curve"):
    """
    Convenience helper to plot the ROC curve, used only in notebooks / paper
    figure generation. This is a no-op if matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        # Silently skip plotting when matplotlib is unavailable.
        return

    plt.figure()
    plt.plot(result.fpr, result.tpr, label=f"AUC = {result.frame_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

