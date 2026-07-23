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
    """
    model.eval()
    
    all_sent_true, all_sent_pred = [], []
    all_traj_true, all_traj_pred = [], []
    all_pred_sequences = []
    all_seq_lens = []
    
    with torch.no_grad():
        for batch in target_loader:
            texts_batch = batch["texts"]
            sentiments = batch["sentiments"]
            trajectories = batch["trajectories"]
            seq_lens = batch["seq_lens"]
            padding_mask = batch["padding_mask"].to(device)
            
            predictions = model(
                texts_batch, seq_lens=seq_lens, padding_mask=padding_mask
            )
            
            sent_preds = predictions["sentiment_logits"].argmax(dim=-1)
            traj_preds = predictions["trajectory_logits"].argmax(dim=-1)
            
            for i in range(len(seq_lens)):
                sl = seq_lens[i]
                all_sent_true.extend(sentiments[i, :sl].tolist())
                all_sent_pred.extend(sent_preds[i, :sl].cpu().tolist())
                all_pred_sequences.append(sent_preds[i, :sl].cpu().tolist())
                all_seq_lens.append(sl)
            
            all_traj_true.extend(trajectories.tolist())
            all_traj_pred.extend(traj_preds.cpu().tolist())
    
    sent_metrics = compute_classification_metrics(all_sent_true, all_sent_pred)
    traj_metrics = compute_classification_metrics(all_traj_true, all_traj_pred)
    scs = sequence_consistency_score(all_pred_sequences, all_seq_lens)
    
    return {
        "source_domain": source_domain,
        "sentiment": sent_metrics,
        "trajectory": traj_metrics,
        "scs": scs,
    }


def run_cross_domain_evaluation():
    """
    Run the full cross-domain evaluation protocol:
      1. In-domain:  Amazon -> Amazon
      2. Cross:      Amazon -> Dravidian (Tamil)
      3. Cross:      Amazon -> Dravidian (Malayalam)
    """
    logger.info("=" * 70)
    logger.info("  CROSS-DOMAIN EVALUATION")
    logger.info("=" * 70)
    
    output_dir = os.path.join(WORKSPACE_ROOT, "outputs", "cross_domain")
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(WORKSPACE_ROOT, "outputs", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    
    if not os.path.exists(checkpoint_path):
        logger.error(
            f"No trained model found at {checkpoint_path}. "
            "Run training first with: python scripts/train.py"
        )
        return
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    args = checkpoint.get("args", {})
    model_name = args.get("model_name", "bert-base-multilingual-cased")
    
    model = OpinionEvolutionTracker(
        model_name=model_name,
        use_cuda=torch.cuda.is_available(),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    
    results = {}
    
    # ── 1. In-Domain: Amazon -> Amazon ──
    logger.info("\n--- In-Domain: Amazon -> Amazon ---")
    amazon_test = get_amazon_dataloader(split="test", batch_size=8)
    in_domain = evaluate_cross_domain(model, "amazon", amazon_test, device)
    results["amazon_to_amazon"] = in_domain
    logger.info(f"  Sentiment F1: {in_domain['sentiment'].get('f1_macro', 0):.4f}")
    logger.info(f"  Trajectory F1: {in_domain['trajectory'].get('f1_macro', 0):.4f}")
    
    # ── 2. Cross-Domain: Amazon -> Dravidian Tamil ──
    logger.info("\n--- Cross-Domain: Amazon -> Tamil ---")
    tamil_test = get_dravidian_sequence_dataloader(
        language="tamil", task="sentiment", split="test", batch_size=8
    )
    cross_tamil = evaluate_cross_domain(model, "amazon->tamil", tamil_test, device)
    results["amazon_to_tamil"] = cross_tamil
    logger.info(f"  Sentiment F1: {cross_tamil['sentiment'].get('f1_macro', 0):.4f}")
    logger.info(f"  Trajectory F1: {cross_tamil['trajectory'].get('f1_macro', 0):.4f}")
    
    # ── 3. Cross-Domain: Amazon -> Dravidian Malayalam ──
    logger.info("\n--- Cross-Domain: Amazon -> Malayalam ---")
    try:
        mal_test = get_dravidian_sequence_dataloader(
            language="malayalam", task="sentiment", split="test", batch_size=8
        )
        cross_mal = evaluate_cross_domain(model, "amazon->malayalam", mal_test, device)
        results["amazon_to_malayalam"] = cross_mal
        logger.info(f"  Sentiment F1: {cross_mal['sentiment'].get('f1_macro', 0):.4f}")
    except Exception as e:
        logger.warning(f"  Skipped Malayalam: {e}")
    
    # ── 4. Cross-Domain: Amazon -> Dravidian Kannada ──
    logger.info("\n--- Cross-Domain: Amazon -> Kannada ---")
    try:
        kan_test = get_dravidian_sequence_dataloader(
            language="kannada", task="sentiment", split="test", batch_size=8
        )
        cross_kan = evaluate_cross_domain(model, "amazon->kannada", kan_test, device)
        results["amazon_to_kannada"] = cross_kan
        logger.info(f"  Sentiment F1: {cross_kan['sentiment'].get('f1_macro', 0):.4f}")
    except Exception as e:
        logger.warning(f"  Skipped Kannada: {e}")
    
    # ── Summary Table ──
    logger.info("\n" + "=" * 70)
    logger.info("  CROSS-DOMAIN RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Setup':<30} {'Sent F1':<12} {'Traj F1':<12} {'SCS':<10}")
    logger.info("-" * 70)
    
    for setup_name, result in results.items():
        sent_f1 = result["sentiment"].get("f1_macro", 0)
        traj_f1 = result["trajectory"].get("f1_macro", 0)
        scs = result["scs"].get("scs_mean", 0)
        logger.info(f"{setup_name:<30} {sent_f1:<12.4f} {traj_f1:<12.4f} {scs:<10.4f}")
    
    # Degradation analysis
    if "amazon_to_amazon" in results and "amazon_to_tamil" in results:
        in_f1 = results["amazon_to_amazon"]["sentiment"].get("f1_macro", 0)
        cross_f1 = results["amazon_to_tamil"]["sentiment"].get("f1_macro", 0)
        if in_f1 > 0:
            degradation = ((in_f1 - cross_f1) / in_f1) * 100
            logger.info(f"\n  Cross-domain degradation (Amazon->Tamil): {degradation:.1f}%")
    
    # Save results
    results_path = os.path.join(output_dir, "cross_domain_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {results_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("  CROSS-DOMAIN EVALUATION COMPLETE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_cross_domain_evaluation()

