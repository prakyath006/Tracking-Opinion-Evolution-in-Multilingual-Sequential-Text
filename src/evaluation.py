"""
=============================================================================
Module 7 — Evaluation & Metrics
=============================================================================
Computes all evaluation metrics for the opinion evolution tracker:

Standard Metrics:
  • Accuracy, Precision, Recall, F1 (macro, weighted)
  • Confusion matrix
  • Per-class metrics

Custom Metric (Novel Contribution):
  • Sequence Consistency Score (SCS) — measures how coherent the model's
    predictions are across a user's review sequence.

Author : Opinion Evolution Tracking Project
Date   : 2026
=============================================================================
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Standard Classification Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Parameters
    ----------
    y_true : List[int]
        Ground truth labels.
    y_pred : List[int]
        Predicted labels.
    label_names : List[str], optional
        Human-readable names for each class.
    average : str
        Averaging mode for P/R/F1 ('macro', 'weighted', 'micro').
        
    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy, precision, recall, f1, and per-class metrics.
    """
    # Filter out ignore indices (-1)
    valid = [(t, p) for t, p in zip(y_true, y_pred) if t != -1]
    if not valid:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    y_true_valid, y_pred_valid = zip(*valid)
    
    acc = accuracy_score(y_true_valid, y_pred_valid)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_valid, y_pred_valid, average=average, zero_division=0
    )
    
    metrics = {
        "accuracy": acc,
        f"precision_{average}": precision,
        f"recall_{average}": recall,
        f"f1_{average}": f1,
    }
    
    # Per-class metrics
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true_valid, y_pred_valid, average=None, zero_division=0
    )
    
    unique_labels = sorted(set(y_true_valid) | set(y_pred_valid))
    for i, label in enumerate(unique_labels):
        if i < len(precision_per):
            name = label_names[label] if label_names and label < len(label_names) else f"class_{label}"
            metrics[f"precision_{name}"] = precision_per[i]
            metrics[f"recall_{name}"] = recall_per[i]
            metrics[f"f1_{name}"] = f1_per[i]
            metrics[f"support_{name}"] = int(support_per[i])
    
    return metrics


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Parameters
    ----------
    y_true : List[int]
        Ground truth labels.
    y_pred : List[int]
        Predicted labels.
    label_names : List[str], optional
        Class names for display.
        
    Returns
    -------
    np.ndarray
        Confusion matrix of shape [num_classes, num_classes].
    """
    valid = [(t, p) for t, p in zip(y_true, y_pred) if t != -1]
    if not valid:
        return np.array([])
    
    y_true_valid, y_pred_valid = zip(*valid)
    return confusion_matrix(y_true_valid, y_pred_valid)


def get_classification_report(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
) -> str:
    """
    Get a formatted classification report string.
    
    Parameters
    ----------
    y_true : List[int]
        Ground truth labels.
    y_pred : List[int]
        Predicted labels.
    label_names : List[str], optional
        Class names.
        
    Returns
    -------
    str
        Formatted classification report.
    """
    valid = [(t, p) for t, p in zip(y_true, y_pred) if t != -1]
    if not valid:
        return "No valid predictions to evaluate."
    
    y_true_valid, y_pred_valid = zip(*valid)
    
    return classification_report(
        y_true_valid, y_pred_valid,
        target_names=label_names,
        zero_division=0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Novel Metric: Sequence Consistency Score (SCS)
# ──────────────────────────────────────────────────────────────────────────────

def sequence_consistency_score(
    predicted_labels: List[List[int]],
    seq_lens: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Sequence Consistency Score (SCS) — Novel Metric.
    
    Measures how consistent (non-fluctuating) the model's predictions are
    across each user's review sequence. A model that predicts smooth
    transitions gets a higher score than one that oscillates.
    
    Formula:
        SCS_i = 1 - (num_label_flips_i / (seq_len_i - 1))
        SCS   = mean(SCS_i) over all sequences
    
    Example:
        Sequence [Pos, Pos, Neg, Neg] -> 1 flip / 3 transitions = SCS = 0.67
        Sequence [Pos, Neg, Pos, Neg] -> 3 flips / 3 transitions = SCS = 0.00
        Sequence [Pos, Pos, Pos, Pos] -> 0 flips / 3 transitions = SCS = 1.00
    
    Why is this useful?
    - Real opinion evolution is typically smooth, not erratic
    - A model with high accuracy but low SCS is making inconsistent predictions
    - This metric captures sequential coherence that F1 alone cannot
    
    Parameters
    ----------
    predicted_labels : List[List[int]]
        Predicted labels for each sequence. Each inner list is one user's
        prediction sequence.
    seq_lens : List[int], optional
        Actual sequence lengths (to ignore padding). If None, uses full length.
        
    Returns
    -------
    Dict[str, float]
        - 'scs_mean': Average SCS across all sequences
        - 'scs_std': Standard deviation
        - 'scs_min': Minimum SCS (most inconsistent sequence)
        - 'scs_max': Maximum SCS (most consistent sequence)
        - 'num_sequences': Number of sequences evaluated
    """
    scores = []
    
    for i, seq in enumerate(predicted_labels):
        length = seq_lens[i] if seq_lens else len(seq)
        
        # Need at least 2 reviews for transitions
        if length < 2:
            scores.append(1.0)  # Single review is perfectly consistent
            continue
        
        # Trim to actual length
        seq_trimmed = seq[:length]
        
        # Count label flips (transitions where label changes)
        num_flips = sum(
            1 for j in range(1, len(seq_trimmed))
            if seq_trimmed[j] != seq_trimmed[j-1]
        )
        
        total_transitions = len(seq_trimmed) - 1
        scs = 1.0 - (num_flips / total_transitions)
        scores.append(scs)
    
    if not scores:
        return {
            "scs_mean": 0.0,
            "scs_std": 0.0,
            "scs_min": 0.0,
            "scs_max": 0.0,
            "num_sequences": 0,
        }
    
    return {
        "scs_mean": float(np.mean(scores)),
        "scs_std": float(np.std(scores)),
        "scs_min": float(np.min(scores)),
        "scs_max": float(np.max(scores)),
        "num_sequences": len(scores),
    }


def sequence_consistency_score_with_ground_truth(
    predicted_labels: List[List[int]],
    ground_truth_labels: List[List[int]],
    seq_lens: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Comparative SCS — Measures how well the model's predicted sequence
    matches the ground truth sequence in terms of transition patterns.
    
    This variant compares the TRANSITIONS (not individual labels).
    If ground truth has a flip at position i, does the prediction also flip?
    
    Parameters
    ----------
    predicted_labels : List[List[int]]
        Predicted label sequences.
    ground_truth_labels : List[List[int]]
        Ground truth label sequences.
    seq_lens : List[int], optional
        Actual sequence lengths.
        
    Returns
    -------
    Dict[str, float]
        - 'transition_accuracy': % of transitions correctly predicted
        - 'scs_predicted': SCS of predicted sequences
        - 'scs_ground_truth': SCS of ground truth sequences
        - 'scs_delta': Difference (predicted - ground_truth)
    """
    transition_correct = 0
    transition_total = 0
    
    pred_scs = sequence_consistency_score(predicted_labels, seq_lens)
    gt_scs = sequence_consistency_score(ground_truth_labels, seq_lens)
    
    for i, (pred_seq, gt_seq) in enumerate(zip(predicted_labels, ground_truth_labels)):
        length = seq_lens[i] if seq_lens else min(len(pred_seq), len(gt_seq))
        
        if length < 2:
            continue
        
        for j in range(1, length):
            pred_flip = (pred_seq[j] != pred_seq[j-1])
            gt_flip = (gt_seq[j] != gt_seq[j-1])
            
            if pred_flip == gt_flip:
                transition_correct += 1
            transition_total += 1
    
    transition_accuracy = (
        transition_correct / transition_total if transition_total > 0 else 0.0
    )
    
    return {
        "transition_accuracy": transition_accuracy,
        "scs_predicted": pred_scs["scs_mean"],
        "scs_ground_truth": gt_scs["scs_mean"],
        "scs_delta": pred_scs["scs_mean"] - gt_scs["scs_mean"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Comprehensive Evaluation Runner
# ──────────────────────────────────────────────────────────────────────────────

class EvaluationRunner:
    """
    Runs comprehensive evaluation across all tasks and metrics.
    """
    
    SENTIMENT_LABELS = ["Positive", "Negative", "Neutral/Mixed", "Unknown"]
    TREND_LABELS = ["Improving", "Declining", "Stable"]
    TRAJECTORY_LABELS = ["Improving", "Declining", "Stable", "Volatile"]
    
    def __init__(self):
        self.results = {}
    
    def evaluate_all(
        self,
        sentiment_true: List[int],
        sentiment_pred: List[int],
        trend_true: List[int],
        trend_pred: List[int],
        trajectory_true: List[int],
        trajectory_pred: List[int],
        predicted_sequences: Optional[List[List[int]]] = None,
        ground_truth_sequences: Optional[List[List[int]]] = None,
        seq_lens: Optional[List[int]] = None,
    ) -> Dict[str, Dict]:
        """
        Run full evaluation across all tasks.
        
        Returns
        -------
        Dict containing metrics for each task.
        """
        results = {}
        
        # Sentiment evaluation
        results["sentiment"] = compute_classification_metrics(
            sentiment_true, sentiment_pred,
            label_names=self.SENTIMENT_LABELS,
        )
        
        # Trend evaluation
        results["trend"] = compute_classification_metrics(
            trend_true, trend_pred,
            label_names=self.TREND_LABELS,
        )
        
        # Trajectory evaluation
        results["trajectory"] = compute_classification_metrics(
            trajectory_true, trajectory_pred,
            label_names=self.TRAJECTORY_LABELS,
        )
        
        # SCS (if sequences are provided)
        if predicted_sequences:
            results["scs"] = sequence_consistency_score(
                predicted_sequences, seq_lens
            )
            
            if ground_truth_sequences:
                results["scs_comparative"] = sequence_consistency_score_with_ground_truth(
                    predicted_sequences, ground_truth_sequences, seq_lens
                )
        
        self.results = results
        return results
    
    def print_report(self) -> str:
        """Format and print a comprehensive evaluation report."""
        lines = []
        lines.append("=" * 70)
        lines.append("  EVALUATION REPORT")
        lines.append("=" * 70)
        
        for task_name in ["sentiment", "trend", "trajectory"]:
            if task_name not in self.results:
                continue
            
            metrics = self.results[task_name]
            lines.append(f"\n--- {task_name.upper()} ---")
            lines.append(f"  Accuracy:       {metrics.get('accuracy', 0):.4f}")
            lines.append(f"  Precision (M):  {metrics.get('precision_macro', 0):.4f}")
            lines.append(f"  Recall (M):     {metrics.get('recall_macro', 0):.4f}")
            lines.append(f"  F1 (M):         {metrics.get('f1_macro', 0):.4f}")
        
        if "scs" in self.results:
            scs = self.results["scs"]
            lines.append(f"\n--- SEQUENCE CONSISTENCY SCORE (SCS) ---")
            lines.append(f"  Mean SCS:  {scs['scs_mean']:.4f}")
            lines.append(f"  Std SCS:   {scs['scs_std']:.4f}")
            lines.append(f"  Min SCS:   {scs['scs_min']:.4f}")
            lines.append(f"  Max SCS:   {scs['scs_max']:.4f}")
        
        if "scs_comparative" in self.results:
            comp = self.results["scs_comparative"]
            lines.append(f"\n--- TRANSITION ANALYSIS ---")
            lines.append(f"  Transition Accuracy: {comp['transition_accuracy']:.4f}")
            lines.append(f"  SCS Delta (pred-gt): {comp['scs_delta']:.4f}")
        
        lines.append("\n" + "=" * 70)
        
        report = "\n".join(lines)
        logger.info(report)
        return report


# ──────────────────────────────────────────────────────────────────────────────
# Execution Time Tracking (Guide Module 5)
# ──────────────────────────────────────────────────────────────────────────────

class ExecutionTimer:
    """
    Tracks execution time for training and inference.
    Measures:
      - Training time per epoch (seconds)
      - Inference latency per sample (milliseconds)
      - Total execution time
    
    Reference: Guide Module 5 (Performance Metrics — execution time)
    """

    def __init__(self):
        self.epoch_times = []         # seconds per epoch
        self.inference_times = []     # seconds per inference batch
        self.inference_samples = []   # number of samples per inference batch
        self._start_time = None
        self._total_start = None

    def start_epoch(self):
        """Call at the start of each training epoch."""
        self._start_time = time.time()

    def end_epoch(self):
        """Call at the end of each training epoch. Records elapsed time."""
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            self.epoch_times.append(elapsed)
            self._start_time = None
            return elapsed
        return 0.0

    def start_inference(self):
        """Call before running inference on a batch."""
        self._start_time = time.time()

    def end_inference(self, num_samples: int):
        """Call after inference on a batch. Records elapsed time and sample count."""
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            self.inference_times.append(elapsed)
            self.inference_samples.append(num_samples)
            self._start_time = None
            return elapsed
        return 0.0

    def start_total(self):
        """Start tracking total execution time."""
        self._total_start = time.time()

    def get_total_time(self) -> float:
        """Get total elapsed time since start_total() was called."""
        if self._total_start is not None:
            return time.time() - self._total_start
        return 0.0

    def get_metrics(self) -> Dict[str, float]:
        """
        Get all execution time metrics.

        Returns
        -------
        Dict with:
            - avg_epoch_time_sec: Average training time per epoch
            - total_training_time_sec: Sum of all epoch times
            - avg_inference_latency_ms: Average inference time per sample (ms)
            - total_inference_time_sec: Sum of all inference times
            - total_samples_inferred: Total samples processed during inference
        """
        metrics = {}

        # Training time
        if self.epoch_times:
            metrics["avg_epoch_time_sec"] = np.mean(self.epoch_times)
            metrics["total_training_time_sec"] = sum(self.epoch_times)
            metrics["num_epochs"] = len(self.epoch_times)
        else:
            metrics["avg_epoch_time_sec"] = 0.0
            metrics["total_training_time_sec"] = 0.0
            metrics["num_epochs"] = 0

        # Inference latency
        if self.inference_times and self.inference_samples:
            total_time = sum(self.inference_times)
            total_samples = sum(self.inference_samples)
            if total_samples > 0:
                metrics["avg_inference_latency_ms"] = (
                    total_time / total_samples
                ) * 1000
            else:
                metrics["avg_inference_latency_ms"] = 0.0
            metrics["total_inference_time_sec"] = total_time
            metrics["total_samples_inferred"] = total_samples
        else:
            metrics["avg_inference_latency_ms"] = 0.0
            metrics["total_inference_time_sec"] = 0.0
            metrics["total_samples_inferred"] = 0

        # Total time
        metrics["total_execution_time_sec"] = self.get_total_time()

        return metrics

    def print_report(self) -> str:
        """Print a formatted execution time report."""
        m = self.get_metrics()
        lines = [
            "\n--- EXECUTION TIME REPORT ---",
            f"  Training epochs:         {m['num_epochs']}",
            f"  Avg epoch time:          {m['avg_epoch_time_sec']:.2f} sec",
            f"  Total training time:     {m['total_training_time_sec']:.2f} sec",
            f"  Avg inference latency:   {m['avg_inference_latency_ms']:.2f} ms/sample",
            f"  Total inference time:    {m['total_inference_time_sec']:.2f} sec",
            f"  Total execution time:    {m['total_execution_time_sec']:.2f} sec",
        ]
        report = "\n".join(lines)
        logger.info(report)
        return report
