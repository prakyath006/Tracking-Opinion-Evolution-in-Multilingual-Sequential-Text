"""
=============================================================================
Training Script — Opinion Evolution Tracker
=============================================================================
Training loop with:
  • Multi-task loss optimization
  • Early stopping
  • Learning rate scheduling
  • Checkpoint saving
  • Train/Validation/Test evaluation
  • Logging of all metrics

Usage:
    python scripts/train.py --domain amazon --epochs 20 --batch_size 16
    python scripts/train.py --domain dravidian --language tamil --epochs 15

Author : Opinion Evolution Tracking Project
Date   : 2026
=============================================================================
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add src/ to path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src"))

from model import OpinionEvolutionTracker
from classifier import MultiTaskLoss
from dataset import (
    get_amazon_dataloader,
    get_dravidian_dataloader,
)
from evaluation import (
    compute_classification_metrics,
    sequence_consistency_score,
    EvaluationRunner,
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Training Functions
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: OpinionEvolutionTracker,
    dataloader,
    loss_fn: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch on sequence data (Amazon domain).
    
    Returns
    -------
    Dict with average losses for the epoch.
    """
    model.train()
    # Keep the encoder in eval mode (frozen, no dropout changes)
    model.embedding_generator.eval()
    
    epoch_losses = {"total": 0, "sentiment": 0, "trend": 0, "trajectory": 0}
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        texts_batch = batch["texts"]          # List[List[str]]
        sentiments = batch["sentiments"].to(device)      # [batch, max_seq]
        trends = batch["trends"].to(device)              # [batch, max_seq]
        trajectories = batch["trajectories"].to(device)  # [batch]
        seq_lens = batch["seq_lens"]
        padding_mask = batch["padding_mask"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(
            texts_batch, seq_lens=seq_lens, padding_mask=padding_mask
        )
        
        # Compute multi-task loss
        targets = {
            "sentiments": sentiments,
            "trends": trends,
            "trajectories": trajectories,
        }
        losses = loss_fn(predictions, targets)
        
        # Backward pass
        losses["total_loss"].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate losses
        epoch_losses["total"] += losses["total_loss"].item()
        epoch_losses["sentiment"] += losses["sentiment_loss"].item()
        epoch_losses["trend"] += losses["trend_loss"].item()
        epoch_losses["trajectory"] += losses["trajectory_loss"].item()
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            avg_loss = epoch_losses["total"] / num_batches
            logger.info(
                f"  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f}"
            )
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= max(num_batches, 1)
    
    return epoch_losses


def evaluate_epoch(
    model: OpinionEvolutionTracker,
    dataloader,
    loss_fn: MultiTaskLoss,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on validation/test data.
    
    Returns
    -------
    Dict with losses and classification metrics.
    """
    model.eval()
    
    epoch_losses = {"total": 0, "sentiment": 0, "trend": 0, "trajectory": 0}
    num_batches = 0
    
    all_sent_true, all_sent_pred = [], []
    all_trend_true, all_trend_pred = [], []
    all_traj_true, all_traj_pred = [], []
    all_pred_sequences = []
    all_seq_lens = []
    
    with torch.no_grad():
        for batch in dataloader:
            texts_batch = batch["texts"]
            sentiments = batch["sentiments"].to(device)
            trends = batch["trends"].to(device)
            trajectories = batch["trajectories"].to(device)
            seq_lens = batch["seq_lens"]
            padding_mask = batch["padding_mask"].to(device)
            
            # Forward pass
            predictions = model(
                texts_batch, seq_lens=seq_lens, padding_mask=padding_mask
            )
            
            # Compute loss
            targets = {
                "sentiments": sentiments,
                "trends": trends,
                "trajectories": trajectories,
            }
            losses = loss_fn(predictions, targets)
            
            epoch_losses["total"] += losses["total_loss"].item()
            epoch_losses["sentiment"] += losses["sentiment_loss"].item()
            epoch_losses["trend"] += losses["trend_loss"].item()
            epoch_losses["trajectory"] += losses["trajectory_loss"].item()
            num_batches += 1
            
            # Collect predictions for metrics
            sent_preds = predictions["sentiment_logits"].argmax(dim=-1)  # [batch, seq]
            trend_preds = predictions["trend_logits"].argmax(dim=-1)
            traj_preds = predictions["trajectory_logits"].argmax(dim=-1)
            
            for i in range(len(seq_lens)):
                sl = seq_lens[i]
                all_sent_true.extend(sentiments[i, :sl].cpu().tolist())
                all_sent_pred.extend(sent_preds[i, :sl].cpu().tolist())
                all_trend_true.extend(trends[i, :sl].cpu().tolist())
                all_trend_pred.extend(trend_preds[i, :sl].cpu().tolist())
                all_pred_sequences.append(sent_preds[i, :sl].cpu().tolist())
                all_seq_lens.append(sl)
            
            all_traj_true.extend(trajectories.cpu().tolist())
            all_traj_pred.extend(traj_preds.cpu().tolist())
    
    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= max(num_batches, 1)
    
    # Classification metrics
    sent_metrics = compute_classification_metrics(all_sent_true, all_sent_pred)
    trend_metrics = compute_classification_metrics(all_trend_true, all_trend_pred)
    traj_metrics = compute_classification_metrics(all_traj_true, all_traj_pred)
    
    # SCS
    scs = sequence_consistency_score(all_pred_sequences, all_seq_lens)
    
    return {
        "losses": epoch_losses,
        "sentiment_metrics": sent_metrics,
        "trend_metrics": trend_metrics,
        "trajectory_metrics": traj_metrics,
        "scs": scs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    """Main training function."""
    
    logger.info("=" * 70)
    logger.info("  TRAINING: Opinion Evolution Tracker")
    logger.info("=" * 70)
    logger.info(f"  Domain:     {args.domain}")
    logger.info(f"  Model:      {args.model_name}")
    logger.info(f"  Epochs:     {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  LR:         {args.lr}")
    logger.info("=" * 70)
    
    # Create output directory
    output_dir = os.path.join(WORKSPACE_ROOT, "outputs", "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logs directory
    logs_dir = os.path.join(WORKSPACE_ROOT, "outputs", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # ── Data Loaders ──
    if args.domain == "amazon":
        train_loader = get_amazon_dataloader(
            split="train", batch_size=args.batch_size, max_seq_len=args.max_seq_len
        )
        val_loader = get_amazon_dataloader(
            split="val", batch_size=args.batch_size, max_seq_len=args.max_seq_len
        )
        test_loader = get_amazon_dataloader(
            split="test", batch_size=args.batch_size, max_seq_len=args.max_seq_len
        )
    else:
        logger.info("Dravidian domain uses single-text classification (no sequences).")
        logger.info("For Dravidian, use the baseline training scripts instead.")
        return
    
    # ── Model ──
    model = OpinionEvolutionTracker(
        model_name=args.model_name,
        max_token_length=args.max_token_length,
        lstm_hidden_dim=args.lstm_hidden,
        lstm_num_layers=args.lstm_layers,
        lstm_dropout=args.dropout,
        attention_type=args.attention_type,
        attention_dim=args.attention_dim,
        freeze_encoder=args.freeze_encoder,
        use_cuda=not args.no_cuda,
    )
    device = model.device
    
    # ── Loss & Optimizer ──
    loss_fn = MultiTaskLoss(
        sentiment_weight=1.0,
        trend_weight=0.5,
        trajectory_weight=1.0,
    )
    
    optimizer = AdamW(
        model.get_trainable_params(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    
    # ── Training Loop ──
    best_val_loss = float("inf")
    patience_counter = 0
    training_log = []
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        logger.info(f"\n{'='*40} Epoch {epoch}/{args.epochs} {'='*40}")
        train_losses = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch
        )
        
        # Validate
        val_results = evaluate_epoch(model, val_loader, loss_fn, device)
        val_loss = val_results["losses"]["total"]
        
        # Log epoch results
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch} | Train Loss: {train_losses['total']:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Sent F1: {val_results['sentiment_metrics'].get('f1_macro', 0):.4f} | "
            f"Val Traj F1: {val_results['trajectory_metrics'].get('f1_macro', 0):.4f} | "
            f"SCS: {val_results['scs'].get('scs_mean', 0):.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        training_log.append({
            "epoch": epoch,
            "train_loss": train_losses,
            "val_loss": val_results["losses"],
            "val_sentiment_f1": val_results["sentiment_metrics"].get("f1_macro", 0),
            "val_trajectory_f1": val_results["trajectory_metrics"].get("f1_macro", 0),
            "val_scs": val_results["scs"].get("scs_mean", 0),
            "time_seconds": epoch_time,
        })
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping & checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, checkpoint_path)
            logger.info(f"  [BEST] Checkpoint saved to {checkpoint_path}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience: {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered!")
                break
    
    # ── Final Test Evaluation ──
    logger.info("\n" + "=" * 70)
    logger.info("  FINAL TEST EVALUATION")
    logger.info("=" * 70)
    
    # Load best model
    checkpoint = torch.load(
        os.path.join(output_dir, "best_model.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_results = evaluate_epoch(model, test_loader, loss_fn, device)
    
    # Print comprehensive report
    evaluator = EvaluationRunner()
    evaluator.results = {
        "sentiment": test_results["sentiment_metrics"],
        "trend": test_results["trend_metrics"],
        "trajectory": test_results["trajectory_metrics"],
        "scs": test_results["scs"],
    }
    report = evaluator.print_report()
    
    # Save training log
    log_path = os.path.join(logs_dir, f"training_log_{args.domain}.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info(f"Training log saved to: {log_path}")
    
    # Save test results
    results_path = os.path.join(logs_dir, f"test_results_{args.domain}.json")
    with open(results_path, "w") as f:
        json.dump({
            "losses": test_results["losses"],
            "sentiment": test_results["sentiment_metrics"],
            "trend": test_results["trend_metrics"],
            "trajectory": test_results["trajectory_metrics"],
            "scs": test_results["scs"],
        }, f, indent=2)
    logger.info(f"Test results saved to: {results_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("  TRAINING COMPLETE!")
    logger.info("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# CLI Arguments
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Opinion Evolution Tracker model"
    )
    
    # Data
    parser.add_argument("--domain", type=str, default="amazon",
                        choices=["amazon", "dravidian"],
                        help="Training domain")
    parser.add_argument("--language", type=str, default="tamil",
                        choices=["tamil", "malayalam", "kannada"],
                        help="Language for Dravidian domain")
    
    # Model
    parser.add_argument("--model_name", type=str, 
                        default="bert-base-multilingual-cased",
                        help="HuggingFace model name")
    parser.add_argument("--max_token_length", type=int, default=128,
                        help="Max subword token length per review")
    parser.add_argument("--max_seq_len", type=int, default=20,
                        help="Max reviews per user sequence")
    parser.add_argument("--lstm_hidden", type=int, default=256,
                        help="Bi-LSTM hidden dimension")
    parser.add_argument("--lstm_layers", type=int, default=2,
                        help="Number of LSTM layers")
    parser.add_argument("--attention_type", type=str, default="single",
                        choices=["single", "multi"],
                        help="Attention type")
    parser.add_argument("--attention_dim", type=int, default=128,
                        help="Attention dimension")
    parser.add_argument("--freeze_encoder", action="store_true", default=True,
                        help="Freeze pre-trained encoder weights")
    parser.add_argument("--no_freeze_encoder", dest="freeze_encoder",
                        action="store_false",
                        help="Unfreeze encoder (fine-tune)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=20,
                        help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
