"""
=============================================================================
Evaluation Utilities — seeding, valid baselines, uncertainty, comment-level metrics
=============================================================================
Shared by scripts/compute_valid_baselines.py, scripts/export_predictions.py,
scripts/eval_comment_level.py and scripts/eval_llm_comment_level.py.

Nothing here trains or runs a model. Functions are small and pure so they can be
unit-tested (tests/test_eval_utils.py).

Design rules
  * A majority-class baseline is chosen from TRAINING labels only, then scored on
    the test labels (never the other way round).
  * Trend metrics can be restricted to observed transitions (position >= 2),
    excluding the artificial position-1 STABLE placeholder.
  * Uncertainty is a percentile bootstrap that resamples whole units (users for
    sequence heads, comments for comment-level tasks).
=============================================================================
"""
import json
import os
import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

SENTIMENT_LABELS = [0, 1, 2, 3]          # POSITIVE, NEGATIVE, MIXED, UNKNOWN (src/ontology.py)
SENTIMENT_NAMES = ["POSITIVE", "NEGATIVE", "MIXED", "UNKNOWN"]
DETERMINATE = [0, 1, 2]                  # labels shared by every corpus (Amazon has no UNKNOWN)

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMENT_LEVEL_DIR = os.path.join(WORKSPACE_ROOT, "data", "comment_level")


# ── reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA) and request deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:                                   # pure-numpy use (tests, label statistics)
        pass


# ── metrics ──────────────────────────────────────────────────────────────────
def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                           labels: Sequence[int], names: Optional[Sequence[str]] = None) -> Dict:
    """Accuracy, macro-F1 over a FIXED label set, per-class P/R/F1/support and the confusion matrix."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(labels)
    out = {"n": int(len(y_true)), "labels": labels}
    if len(y_true) == 0:
        return out
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    names = names or [str(l) for l in labels]
    out.update({
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f)),
        "per_class": {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, n in enumerate(names)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    })
    return out


def majority_baseline(train_labels: Sequence[int], test_labels: Sequence[int],
                      labels: Sequence[int], names: Optional[Sequence[str]] = None) -> Dict:
    """Predict the most frequent TRAINING class for every test item; score it on the test labels."""
    counts = np.bincount(np.asarray(train_labels, dtype=int), minlength=max(labels) + 1)
    cls = int(np.argmax(counts))
    res = classification_metrics(test_labels, [cls] * len(test_labels), labels, names)
    res["majority_class_from_train"] = cls
    res["train_class_counts"] = {int(l): int(counts[l]) for l in labels}
    return res


def observed_transition_mask(positions: Sequence[int]) -> np.ndarray:
    """True for positions >= 1 (0-based): the transitions that were actually observed.
    Position 0 carries the artificial STABLE placeholder (src/dataset.py::build_trend_labels)."""
    return np.asarray(positions) >= 1


def trend_metrics_observed_only(y_true, y_pred, positions, labels, names=None) -> Dict:
    m = observed_transition_mask(positions)
    return classification_metrics(np.asarray(y_true)[m], np.asarray(y_pred)[m], labels, names)


def gold_scs(sequences: Iterable[Sequence[int]]) -> Dict[str, float]:
    """Sequence Consistency Score of the GOLD sentiment sequences (same formula as src/evaluation.py)."""
    scores = []
    for seq in sequences:
        seq = list(seq)
        if len(seq) < 2:
            scores.append(1.0)
            continue
        flips = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
        scores.append(1.0 - flips / (len(seq) - 1))
    return {"scs_mean": float(np.mean(scores)), "scs_std": float(np.std(scores)), "num_sequences": len(scores)}


# ── uncertainty ──────────────────────────────────────────────────────────────
def bootstrap_ci(y_true: Sequence[int], y_pred: Sequence[int], metric: Callable[[np.ndarray, np.ndarray], float],
                 groups: Optional[Sequence] = None, n_resamples: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> Dict:
    """Percentile bootstrap. If `groups` is given, whole groups (e.g. users) are resampled, otherwise items."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    if groups is None:
        idx_sets = None
        n = len(y_true)
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        idx_sets = [np.flatnonzero(groups == g) for g in uniq]
        n = len(uniq)
    stats = []
    for _ in range(n_resamples):
        pick = rng.integers(0, n, size=n)
        idx = pick if idx_sets is None else np.concatenate([idx_sets[i] for i in pick])
        stats.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(metric(y_true, y_pred)), "ci_low": float(lo), "ci_high": float(hi),
            "resamples": n_resamples, "unit": "groups" if groups is not None else "items"}


def macro_f1_fn(labels: Sequence[int]) -> Callable[[np.ndarray, np.ndarray], float]:
    from sklearn.metrics import f1_score
    labels = list(labels)
    return lambda y, p: float(f1_score(y, p, labels=labels, average="macro", zero_division=0))


# ── comment-level (multilingual / cross-domain) evaluation ───────────────────
def comment_level_report(y_true: Sequence[int], y_pred: Sequence[int],
                         bootstrap: int = 0, seed: int = 0) -> Dict:
    """
    Two views of the same predictions.
      primary_3class : only comments whose gold label is POSITIVE/NEGATIVE/MIXED; macro-F1 over those three
                       classes (a predicted UNKNOWN counts as an error). Comparable across every corpus,
                       including Amazon, which has no UNKNOWN label.
      four_class     : all comments, macro-F1 over the four ontology classes (meaningful when both source
                       and target carry UNKNOWN annotations).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    keep = np.isin(y_true, DETERMINATE)
    rep = {
        "primary_3class": classification_metrics(y_true[keep], y_pred[keep], DETERMINATE, SENTIMENT_NAMES[:3]),
        "four_class": classification_metrics(y_true, y_pred, SENTIMENT_LABELS, SENTIMENT_NAMES),
    }
    if bootstrap:
        rep["primary_3class"]["macro_f1_ci95"] = bootstrap_ci(
            y_true[keep], y_pred[keep], macro_f1_fn(DETERMINATE), n_resamples=bootstrap, seed=seed)
    return rep


def load_comment_csv(corpus: str, split: str, max_rows: int = 0):
    """Texts and ontology labels of data/comment_level/<corpus>_<split>.csv (written by
    scripts/prepare_comment_level_data.py)."""
    import pandas as pd
    path = os.path.join(COMMENT_LEVEL_DIR, f"{corpus}_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run scripts/prepare_comment_level_data.py first")
    df = pd.read_csv(path, encoding="utf-8")
    if max_rows:
        df = df.sample(n=min(max_rows, len(df)), random_state=0)
    return df["text"].fillna("").astype(str).tolist(), df["label"].astype(int).tolist()


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
