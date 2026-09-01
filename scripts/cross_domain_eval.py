"""
=============================================================================
Cross-Domain Evaluation Script
=============================================================================
Tests model generalization across domains:
  • Train on Domain 1 (Amazon) -> Test on Domain 2 (DravidianCodeMix)
  • Train on Domain 2 (DravidianCodeMix) -> Test on Domain 1 (Amazon)

Reports:
  • In-domain F1
  • Cross-domain F1
  • % Degradation
  • SCS comparison

This table is one of the strongest publishable results from the project,
demonstrating the model's cross-domain generalization capability.

Usage:
    python scripts/cross_domain_eval.py

Author : Opinion Evolution Tracking Project
Date   : 2026
=============================================================================
"""

import os
import sys
import json
import logging
from typing import Dict, List

import torch

# Add src/ to path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src"))

from model import OpinionEvolutionTracker
from tokenization import MultilingualTokenizer
from embeddings import DomainAdaptedEmbeddings
from dataset import (
    AmazonSequenceDataset,
    DravidianDataset,
    DravidianSequenceDataset,
    get_amazon_dataloader,
    get_dravidian_dataloader,
    get_dravidian_sequence_dataloader,
)
from evaluation import (
    compute_classification_metrics,
    sequence_consistency_score,
    EvaluationRunner,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_cross_domain(
    model: OpinionEvolutionTracker,
    source_domain: str,
    target_loader,
    device: torch.device,
) -> Dict:
    """
    Evaluate a trained model on a different domain's test set.

    Reports all three heads (sentiment, trend, trajectory) -- trend was
    previously not collected here at all (only sentiment + trajectory),
    which meant the trend head had no cross-domain numbers anywhere in the
    project even though train.py's own evaluate_epoch() already collects it
    for in-domain evaluation. Mirrors that same pattern here for parity.
    """
    model.eval()

    all_sent_true, all_sent_pred = [], []
    all_trend_true, all_trend_pred = [], []
    all_traj_true, all_traj_pred = [], []
    all_pred_sequences = []
    all_seq_lens = []

    with torch.no_grad():
        for batch in target_loader:
            texts_batch = batch["texts"]
            sentiments = batch["sentiments"]
            trends = batch["trends"]
            trajectories = batch["trajectories"]
            seq_lens = batch["seq_lens"]
            padding_mask = batch["padding_mask"].to(device)

            predictions = model(
                texts_batch, seq_lens=seq_lens, padding_mask=padding_mask
            )

            sent_preds = predictions["sentiment_logits"].argmax(dim=-1)
            trend_preds = predictions["trend_logits"].argmax(dim=-1)
            traj_preds = predictions["trajectory_logits"].argmax(dim=-1)

            for i in range(len(seq_lens)):
                sl = seq_lens[i]
                all_sent_true.extend(sentiments[i, :sl].tolist())
                all_sent_pred.extend(sent_preds[i, :sl].cpu().tolist())
                all_trend_true.extend(trends[i, :sl].tolist())
                all_trend_pred.extend(trend_preds[i, :sl].cpu().tolist())
                all_pred_sequences.append(sent_preds[i, :sl].cpu().tolist())
                all_seq_lens.append(sl)

            all_traj_true.extend(trajectories.tolist())
            all_traj_pred.extend(traj_preds.cpu().tolist())

    sent_metrics = compute_classification_metrics(all_sent_true, all_sent_pred)
    trend_metrics = compute_classification_metrics(all_trend_true, all_trend_pred)
    traj_metrics = compute_classification_metrics(all_traj_true, all_traj_pred)
    scs = sequence_consistency_score(all_pred_sequences, all_seq_lens)

    return {
        "source_domain": source_domain,
        "sentiment": sent_metrics,
        "trend": trend_metrics,
        "trajectory": traj_metrics,
        "scs": scs,
    }


def load_trained_model(run_id: str, device: torch.device) -> OpinionEvolutionTracker:
    """
    Load a checkpoint saved by scripts/train.py for a given run_id
    ("amazon", "dravidian_tamil", "dravidian_malayalam", "dravidian_kannada").

    train.py names checkpoints best_model_{run_id}.pt precisely so that
    training on more than one domain never overwrites an earlier run's
    checkpoint -- which used to share one fixed "best_model.pt" path,
    making it impossible to hold an Amazon-trained and a Dravidian-trained
    model at the same time, and therefore impossible to evaluate the
    "reverse" direction this script's own docstring always claimed to do.
    """
    checkpoint_dir = os.path.join(WORKSPACE_ROOT, "outputs", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, f"best_model_{run_id}.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No trained model found at {checkpoint_path}. Train it first, e.g.:\n"
            f"  python scripts/train.py --domain amazon\n"
            f"  python scripts/train.py --domain dravidian --language tamil"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    model_name = args.get("model_name", "bert-base-multilingual-cased")

    model = OpinionEvolutionTracker(model_name=model_name, use_cuda=(device.type == "cuda"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded {run_id} model from epoch {checkpoint.get('epoch', '?')} ({checkpoint_path})")
    return model


def get_domain_test_loader(run_id: str, batch_size: int = 8):
    """get_amazon_dataloader / get_dravidian_sequence_dataloader for a run_id's test split."""
    if run_id == "amazon":
        return get_amazon_dataloader(split="test", batch_size=batch_size)
    language = run_id.split("dravidian_", 1)[1]
    return get_dravidian_sequence_dataloader(
        language=language, task="sentiment", split="test", batch_size=batch_size,
    )


def run_cross_domain_evaluation(dravidian_language: str = "tamil"):
    """
    Run the full cross-domain evaluation protocol in BOTH directions:
      1. In-domain:   Amazon -> Amazon
      2. Cross:       Amazon -> Dravidian (tamil / malayalam / kannada)
      3. In-domain:   Dravidian(dravidian_language) -> Dravidian(dravidian_language)
      4. Cross:       Dravidian(dravidian_language) -> Amazon              [reverse]
      5. Cross:       Dravidian(dravidian_language) -> other Dravidian languages

    Requires checkpoints from BOTH `train.py --domain amazon` and
    `train.py --domain dravidian --language <dravidian_language>` to exist
    (see load_trained_model()) -- this is exactly why train.py's checkpoints
    are now domain-specific rather than one shared "best_model.pt".
    """
    logger.info("=" * 70)
    logger.info("  CROSS-DOMAIN EVALUATION (both directions)")
    logger.info("=" * 70)

    output_dir = os.path.join(WORKSPACE_ROOT, "outputs", "cross_domain")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dravidian_run_id = f"dravidian_{dravidian_language}"

    results = {}
    other_languages = [l for l in ("tamil", "malayalam", "kannada") if l != dravidian_language]

    # ── Direction 1: Amazon-trained model ──
    try:
        amazon_model = load_trained_model("amazon", device)
        for target_run_id in ["amazon", dravidian_run_id] + [f"dravidian_{l}" for l in other_languages]:
            label = f"amazon_to_{target_run_id}"
            logger.info(f"\n--- {label} ---")
            loader = get_domain_test_loader(target_run_id)
            result = evaluate_cross_domain(amazon_model, "amazon", loader, device)
            results[label] = result
            logger.info(f"  Sentiment F1: {result['sentiment'].get('f1_macro', 0):.4f} | "
                        f"Trajectory F1: {result['trajectory'].get('f1_macro', 0):.4f}")
    except FileNotFoundError as e:
        logger.warning(f"Skipping Amazon-sourced evaluations: {e}")

    # ── Direction 2 (reverse): Dravidian-trained model ──
    try:
        dravidian_model = load_trained_model(dravidian_run_id, device)
        for target_run_id in [dravidian_run_id, "amazon"] + [f"dravidian_{l}" for l in other_languages]:
            label = f"{dravidian_run_id}_to_{target_run_id}"
            logger.info(f"\n--- {label} ---")
            loader = get_domain_test_loader(target_run_id)
            result = evaluate_cross_domain(dravidian_model, dravidian_run_id, loader, device)
            results[label] = result
            logger.info(f"  Sentiment F1: {result['sentiment'].get('f1_macro', 0):.4f} | "
                        f"Trajectory F1: {result['trajectory'].get('f1_macro', 0):.4f}")
    except FileNotFoundError as e:
        logger.warning(f"Skipping {dravidian_run_id}-sourced evaluations: {e}")

    if not results:
        logger.error("No trained checkpoints found for either domain -- nothing to evaluate.")
        return {}

    # ── Summary Table ──
    logger.info("\n" + "=" * 70)
    logger.info("  CROSS-DOMAIN RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Setup':<34} {'Sent F1':<12} {'Traj F1':<12} {'SCS':<10}")
    logger.info("-" * 70)

    for setup_name, result in results.items():
        sent_f1 = result["sentiment"].get("f1_macro", 0)
        traj_f1 = result["trajectory"].get("f1_macro", 0)
        scs = result["scs"].get("scs_mean", 0)
        logger.info(f"{setup_name:<34} {sent_f1:<12.4f} {traj_f1:<12.4f} {scs:<10.4f}")

    # Degradation analysis, both directions
    logger.info("")
    for in_key, cross_key, label in [
        ("amazon_to_amazon", f"amazon_to_{dravidian_run_id}", f"Amazon -> {dravidian_language}"),
        (f"{dravidian_run_id}_to_{dravidian_run_id}", f"{dravidian_run_id}_to_amazon", f"{dravidian_language} -> Amazon"),
    ]:
        if in_key in results and cross_key in results:
            in_f1 = results[in_key]["sentiment"].get("f1_macro", 0)
            cross_f1 = results[cross_key]["sentiment"].get("f1_macro", 0)
            if in_f1 > 0:
                degradation = ((in_f1 - cross_f1) / in_f1) * 100
                logger.info(f"  Cross-domain degradation ({label}): {degradation:.1f}%")

    # Save results
    results_path = os.path.join(output_dir, f"cross_domain_results_{dravidian_language}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")

    logger.info("\n" + "=" * 70)
    logger.info("  CROSS-DOMAIN EVALUATION COMPLETE!")
    logger.info("=" * 70)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cross-domain evaluation (both directions)")
    parser.add_argument("--language", type=str, default="tamil",
                         choices=["tamil", "malayalam", "kannada"],
                         help="Dravidian language to pair against Amazon")
    cli_args = parser.parse_args()
    run_cross_domain_evaluation(dravidian_language=cli_args.language)

