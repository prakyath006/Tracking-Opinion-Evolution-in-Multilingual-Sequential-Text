"""
=============================================================================
Module 4 — Sequential Model Metrics: Confidence, Calibration, Consistency
=============================================================================
Metrics for OpinionEvolutionTracker's own predictions, on top of what
src/evaluation.py already provides. This module does NOT reimplement
Accuracy/Precision/Recall/F1/confusion-matrix or SCS -- it imports and reuses
evaluation.py's compute_classification_metrics(), compute_confusion_matrix()
and sequence_consistency_score() directly. What it adds:

  - Prediction Entropy   — Shannon entropy of each prediction's softmax
                            distribution, mean/std per head. Low entropy =
                            the model is confident (peaked distribution);
                            high entropy = the model is unsure (flat
                            distribution) — independent of whether it's
                            actually correct.
  - Expected Calibration Error (ECE) — bins predictions by confidence (10
                            equal-width bins), compares each bin's average
                            confidence to its actual accuracy, weighted by
                            bin size. Measures whether "70% confident"
                            predictions really are right ~70% of the time.
  - SCS Reliability Ratio — mean(SCS) / std(SCS) across sequences, reusing
                            evaluation.py's sequence_consistency_score()
                            output directly (it already computes both).

Does not modify model.py, classifier.py, or dataset.py, and does not
unfreeze the encoder or change training in any way -- OpinionEvolutionTracker
is constructed here with its existing default (freeze_encoder=True) purely
to produce predictions to evaluate.

Model state note
-----------------
Entropy and ECE are well-defined for ANY classifier's softmax output,
trained or not -- unlike Module 3's MLM perplexity (which needs an intact,
task-specific pretrained head to mean anything), these formulas are valid
math regardless of whether the model has been trained. This module will
load a trained checkpoint from outputs/checkpoints/best_model_<run_id>.pt
if one exists (produced by scripts/train.py); if none exists, it runs an
UNTRAINED (freshly-initialized) model instead and labels every report
section with that fact prominently, rather than silently presenting
untrained-model numbers as if they reflect real model quality. Untrained
numbers still exercise and validate the metric computation itself
correctly (e.g. an untrained model's near-uniform softmax should show high
entropy and often poor calibration -- both real, checkable properties of
the actual predictions produced).

Usage:
    python src/confidence_eval.py --domain amazon
    python src/confidence_eval.py --domain dravidian --language tamil
    -> outputs/metrics/module4_sequential_model.md
=============================================================================
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from model import OpinionEvolutionTracker
from dataset import get_amazon_dataloader, get_dravidian_sequence_dataloader
from ontology import SentimentState, TransitionType, TrajectoryType
from evaluation import (
    compute_classification_metrics,
    compute_confusion_matrix,
    sequence_consistency_score,
)

logger = logging.getLogger(__name__)

METRICS_DIR = os.path.join(WORKSPACE_ROOT, "outputs", "metrics")
CHECKPOINT_DIR = os.path.join(WORKSPACE_ROOT, "outputs", "checkpoints")
REPORT_PATH = os.path.join(METRICS_DIR, "module4_sequential_model.md")

HEADS = {
    "sentiment": SentimentState,
    "trend": TransitionType,
    "trajectory": TrajectoryType,
}


# ──────────────────────────────────────────────────────────────────────────────
# Prediction Entropy
# ──────────────────────────────────────────────────────────────────────────────

def compute_prediction_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Shannon entropy of each row of `probs` (a probability distribution over
    classes): H = -sum(p * log(p)). Natural log (nats), matching the units
    PyTorch's own cross-entropy loss already uses elsewhere in this
    pipeline, for consistency.

    Parameters
    ----------
    probs : torch.Tensor
        Shape [..., num_classes], rows summing to 1 (softmax output).

    Returns
    -------
    torch.Tensor of shape [...] (one entropy value per prediction).
    """
    eps = 1e-12
    return -(probs * torch.log(probs + eps)).sum(dim=-1)


def entropy_stats(entropies: torch.Tensor) -> Dict[str, float]:
    """Mean/std of a flat tensor of entropy values, empty-safe."""
    if entropies.numel() == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": entropies.mean().item(),
        "std": entropies.std().item() if entropies.numel() > 1 else 0.0,
        "n": entropies.numel(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Expected Calibration Error
# ──────────────────────────────────────────────────────────────────────────────

def compute_ece(
    probs: torch.Tensor,
    y_true: torch.Tensor,
    n_bins: int = 10,
) -> Dict:
    """
    Expected Calibration Error (Guo et al., 2017): bins predictions into
    `n_bins` equal-width confidence bins, and for each bin compares average
    confidence to actual accuracy, weighted by how many predictions fall in
    that bin.

    ECE = sum_b (|bin_b| / N) * |accuracy(bin_b) - avg_confidence(bin_b)|

    Parameters
    ----------
    probs : torch.Tensor
        Shape [N, num_classes], softmax output for N predictions.
    y_true : torch.Tensor
        Shape [N], true class indices. Rows with y_true == -1 (padding /
        ignore-index positions) are excluded before binning.
    n_bins : int
        Number of equal-width confidence bins over [0, 1].

    Returns
    -------
    Dict with: ece (float or None), bins (list of per-bin dicts: range,
    count, avg_confidence, accuracy).
    """
    valid = y_true != -1
    probs = probs[valid]
    y_true = y_true[valid]

    if probs.numel() == 0:
        return {"ece": None, "bins": [], "n": 0}

    confidences, predictions = probs.max(dim=-1)
    correct = (predictions == y_true).float()

    n = probs.shape[0]
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i].item(), bin_edges[i + 1].item()
        # Include the left edge always; include the right edge only for the
        # last bin, so every confidence in [0,1] falls in exactly one bin.
        if i == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)

        count = int(in_bin.sum().item())
        if count == 0:
            bins.append({"range": (lo, hi), "count": 0, "avg_confidence": None, "accuracy": None})
            continue

        bin_confidence = confidences[in_bin].mean().item()
        bin_accuracy = correct[in_bin].mean().item()
        bins.append({
            "range": (lo, hi), "count": count,
            "avg_confidence": bin_confidence, "accuracy": bin_accuracy,
        })
        ece += (count / n) * abs(bin_accuracy - bin_confidence)

    return {"ece": ece, "bins": bins, "n": n}


# ──────────────────────────────────────────────────────────────────────────────
# SCS Reliability Ratio
# ──────────────────────────────────────────────────────────────────────────────

def compute_scs_reliability_ratio(scs_result: Dict) -> Dict:
    """
    SCS Reliability Ratio = mean(SCS) / std(SCS), reusing
    evaluation.sequence_consistency_score()'s own output directly (it
    already computes scs_mean and scs_std -- this does not recompute
    per-sequence SCS itself).

    Returns
    -------
    Dict with: ratio (float or None), mean, std, note.
    """
    mean_scs = scs_result.get("scs_mean")
    std_scs = scs_result.get("scs_std")

    if mean_scs is None or std_scs is None:
        return {"ratio": None, "mean": mean_scs, "std": std_scs,
                "note": "SCS not available (no sequences scored)."}
    if std_scs == 0:
        return {"ratio": None, "mean": mean_scs, "std": std_scs,
                "note": "SCS is identical across all sequences (std=0); ratio is undefined (division by zero)."}

    return {"ratio": mean_scs / std_scs, "mean": mean_scs, "std": std_scs, "note": ""}


# ──────────────────────────────────────────────────────────────────────────────
# Model predictions
# ──────────────────────────────────────────────────────────────────────────────

def run_predictions(
    model: OpinionEvolutionTracker,
    dataloader,
    device: torch.device,
) -> Dict[str, Dict]:
    """
    Forward-pass every batch in `dataloader` through `model` (eval mode, no
    grad) and collect, per head, flattened probabilities + true labels, plus
    per-sequence sentiment predictions for SCS.

    Returns
    -------
    Dict[head_name, {"probs": Tensor[N,C], "y_true": Tensor[N]}] for
    sentiment/trend/trajectory, plus "_scs_sequences": (pred_sequences, seq_lens)
    for the sentiment head's per-sequence SCS.
    """
    model.eval()
    collected = {h: {"probs": [], "y_true": []} for h in HEADS}
    pred_sequences, seq_lens_all = [], []

    with torch.no_grad():
        for batch in dataloader:
            texts_batch = batch["texts"]
            sentiments = batch["sentiments"].to(device)
            trends = batch["trends"].to(device)
            trajectories = batch["trajectories"].to(device)
            seq_lens = batch["seq_lens"]
            padding_mask = batch["padding_mask"].to(device)

            predictions = model(texts_batch, seq_lens=seq_lens, padding_mask=padding_mask)

            sent_probs = F.softmax(predictions["sentiment_logits"], dim=-1)
            trend_probs = F.softmax(predictions["trend_logits"], dim=-1)
            traj_probs = F.softmax(predictions["trajectory_logits"], dim=-1)

            collected["sentiment"]["probs"].append(sent_probs.reshape(-1, sent_probs.shape[-1]).cpu())
            collected["sentiment"]["y_true"].append(sentiments.reshape(-1).cpu())
            collected["trend"]["probs"].append(trend_probs.reshape(-1, trend_probs.shape[-1]).cpu())
            collected["trend"]["y_true"].append(trends.reshape(-1).cpu())
            collected["trajectory"]["probs"].append(traj_probs.cpu())
            collected["trajectory"]["y_true"].append(trajectories.cpu())

            sent_preds = sent_probs.argmax(dim=-1)
            for i, sl in enumerate(seq_lens):
                pred_sequences.append(sent_preds[i, :sl].cpu().tolist())
                seq_lens_all.append(sl)

    result = {}
    for head in HEADS:
        result[head] = {
            "probs": torch.cat(collected[head]["probs"], dim=0),
            "y_true": torch.cat(collected[head]["y_true"], dim=0),
        }
    result["_scs_sequences"] = (pred_sequences, seq_lens_all)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def generate_module4_report(
    domain: str = "amazon",
    language: str = "tamil",
    model_name: str = "bert-base-multilingual-cased",
    checkpoint_path: Optional[str] = None,
    split: str = "test",
    batch_size: int = 8,
    no_cuda: bool = False,
    write: bool = True,
) -> str:
    """
    Compute the full Module 4 metric set (per-head Accuracy/Precision/
    Recall/F1/confusion matrix, SCS, prediction entropy, ECE, SCS
    Reliability Ratio) for `domain`, and write
    outputs/metrics/module4_sequential_model.md.

    Loads a trained checkpoint from outputs/checkpoints/best_model_<run_id>.pt
    if present; otherwise runs an untrained model and labels the report
    accordingly (see module docstring for why this is still meaningful).
    """
    run_id = domain if domain == "amazon" else f"dravidian_{language}"
    device = torch.device("cuda" if (torch.cuda.is_available() and not no_cuda) else "cpu")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_{run_id}.pt")

    is_trained = os.path.exists(checkpoint_path)
    used_model_name = model_name

    if is_trained:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        used_model_name = checkpoint.get("args", {}).get("model_name", model_name)
        model = OpinionEvolutionTracker(model_name=used_model_name, use_cuda=not no_cuda)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded trained checkpoint: {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    else:
        model = OpinionEvolutionTracker(model_name=model_name, use_cuda=not no_cuda)
        logger.warning(
            f"No trained checkpoint at {checkpoint_path} -- running an "
            f"UNTRAINED (freshly-initialized) model. Metrics below reflect "
            f"random initialization, not trained performance."
        )

    if domain == "amazon":
        dataloader = get_amazon_dataloader(split=split, batch_size=batch_size)
    else:
        dataloader = get_dravidian_sequence_dataloader(language=language, split=split, batch_size=batch_size)

    predictions = run_predictions(model, dataloader, device)

    lines = ["# Module 4 — Sequential Model Metrics", ""]
    lines.append(
        f"Domain: `{run_id}` | Split: `{split}` | Encoder: `{used_model_name}`"
    )
    lines.append("")
    if not is_trained:
        lines.append(
            f"⚠️ **No trained checkpoint found at `{os.path.relpath(checkpoint_path, WORKSPACE_ROOT)}`. "
            f"This model is UNTRAINED (freshly initialized).** The metrics below are "
            f"real, correctly-computed properties of this untrained model's actual "
            f"predictions — they validate that the metric code works, but do NOT "
            f"reflect trained-model performance. Run `scripts/train.py --domain "
            f"{domain}"
            + (f" --language {language}" if domain != "amazon" else "")
            + "` (on Colab, where the real encoder can be downloaded) to produce a "
              "checkpoint this report will then use automatically."
        )
        lines.append("")

    for head, taxonomy in HEADS.items():
        probs = predictions[head]["probs"]
        y_true = predictions[head]["y_true"]
        y_pred = probs.argmax(dim=-1)

        lines.append(f"## {head.capitalize()} Head")
        lines.append("")

        # ── Reused from evaluation.py ──
        metrics = compute_classification_metrics(
            y_true.tolist(), y_pred.tolist(), label_names=taxonomy.label_names(),
        )
        lines.append(f"- Accuracy: {metrics.get('accuracy', 0):.4f}")
        lines.append(f"- Precision (macro): {metrics.get('precision_macro', 0):.4f}")
        lines.append(f"- Recall (macro): {metrics.get('recall_macro', 0):.4f}")
        lines.append(f"- F1 (macro): {metrics.get('f1_macro', 0):.4f}")

        cm = compute_confusion_matrix(y_true.tolist(), y_pred.tolist(), label_names=taxonomy.label_names())
        lines.append(f"- Confusion matrix ({', '.join(taxonomy.label_names())}):")
        lines.append("```")
        lines.append(str(cm))
        lines.append("```")

        # ── New: entropy ──
        entropies = compute_prediction_entropy(probs)
        valid_mask = y_true != -1
        stats = entropy_stats(entropies[valid_mask])
        if stats["mean"] is not None:
            lines.append(f"- Prediction entropy: mean={stats['mean']:.4f} nats, "
                          f"std={stats['std']:.4f} nats (n={stats['n']})")
        else:
            lines.append("- Prediction entropy: no valid predictions")

        # ── New: ECE ──
        ece_result = compute_ece(probs, y_true)
        if ece_result["ece"] is not None:
            lines.append(f"- Expected Calibration Error: {ece_result['ece']:.4f}")
        else:
            lines.append("- Expected Calibration Error: not computed (no valid predictions)")
        lines.append("")

    # ── SCS + SCS Reliability Ratio ──
    lines.append("## Sequence Consistency Score (SCS)")
    lines.append("")
    pred_sequences, seq_lens_all = predictions["_scs_sequences"]
    scs_result = sequence_consistency_score(pred_sequences, seq_lens_all)
    lines.append(f"- SCS mean: {scs_result['scs_mean']:.4f}")
    lines.append(f"- SCS std: {scs_result['scs_std']:.4f}")
    lines.append(f"- SCS min/max: {scs_result['scs_min']:.4f} / {scs_result['scs_max']:.4f}")
    lines.append(f"- Sequences scored: {scs_result['num_sequences']}")
    lines.append("")

    reliability = compute_scs_reliability_ratio(scs_result)
    if reliability["ratio"] is not None:
        lines.append(f"**SCS Reliability Ratio: {reliability['ratio']:.2f}** "
                      f"(mean={reliability['mean']:.4f}, std={reliability['std']:.4f})")
    else:
        lines.append(f"**SCS Reliability Ratio: not computed.** {reliability['note']}")
    lines.append("")

    report = "\n".join(lines)

    if write:
        os.makedirs(METRICS_DIR, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Saved: {REPORT_PATH}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Module 4 — sequential model confidence/calibration metrics")
    parser.add_argument("--domain", type=str, default="amazon", choices=["amazon", "dravidian"])
    parser.add_argument("--language", type=str, default="tamil", choices=["tamil", "malayalam", "kannada"])
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    report = generate_module4_report(
        domain=args.domain, language=args.language, model_name=args.model_name,
        checkpoint_path=args.checkpoint_path, split=args.split,
        batch_size=args.batch_size, no_cuda=args.no_cuda,
    )
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    main()
