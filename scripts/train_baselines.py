"""
=============================================================================
Baseline Training Script — Model Comparison (Guide Module 3)
=============================================================================
src/baselines.py defines 5 comparison models (BASELINE_REGISTRY) but nothing
in the repo trained them -- scripts/train.py only ever trains
OpinionEvolutionTracker. This script fills that gap.

Baselines fall into two groups by what they consume:

  Group A -- single-review classifiers (no sequence modeling at all):
    mbert_sentence, xlmr_sentence  -- SentenceLevelTransformer
    textcnn                        -- TextCNN
    Trained on flattened (single review text, sentiment label) pairs, the
    sentiment task only. This is deliberate, not a shortcut: these baselines
    exist specifically to show what sequential modeling buys you, so they
    are evaluated on the *same* sentiment task the full model's sentiment
    head performs, with no sequence signal available to either baseline.

  Group B -- ablation variants (sequence-aware, minus one component):
    lstm_only       -- Bi-LSTM but no attention (final hidden state instead)
    attention_only  -- attention but no Bi-LSTM (over raw embeddings)
    Both consume the exact same per-user sequences as the full model
    (get_amazon_dataloader / get_dravidian_sequence_dataloader) and are
    scored on sentiment (per-position) and trajectory (per-sequence) --
    the two tasks src/baselines.py's own heads support. Neither baseline
    has a trend head, so trend is not scored for this group.

Usage:
    python scripts/train_baselines.py --baseline mbert_sentence --domain amazon
    python scripts/train_baselines.py --baseline lstm_only --domain dravidian --language tamil
    python scripts/train_baselines.py --baseline all --domain amazon   # loop every baseline

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
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src"))

from baselines import BASELINE_REGISTRY, get_baseline_model
from tokenization import MultilingualTokenizer
from embeddings import DomainAdaptedEmbeddings
from dataset import (
    AmazonSequenceDataset,
    DravidianSequenceDataset,
    get_amazon_dataloader,
    get_dravidian_sequence_dataloader,
)
from classifier import compute_class_weights
from ontology import SentimentState, TrajectoryType
from evaluation import compute_classification_metrics, compute_confusion_matrix, sequence_consistency_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GROUP_A = {"mbert_sentence", "xlmr_sentence", "textcnn"}
GROUP_B = {"lstm_only", "attention_only"}


# ══════════════════════════════════════════════════════════════════════════════
# Group A: flat single-review dataset (sentiment task only)
# ══════════════════════════════════════════════════════════════════════════════

def flatten_to_reviews(sequence_dataset) -> Tuple[List[str], List[int]]:
    """Flatten a sequence dataset's `.sequences` into (text, sentiment) pairs."""
    texts, labels = [], []
    for seq in sequence_dataset.sequences:
        texts.extend(seq["texts"])
        labels.extend(seq["sentiments"])
    return texts, labels


class FlatReviewDataset(Dataset):
    """Single-review (text, sentiment label) pairs, for Group A baselines."""

    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def build_vocab(texts: List[str], max_vocab_size: int = 20000) -> Dict[str, int]:
    """Whitespace-level vocabulary for TextCNN (no pretrained tokenizer)."""
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(max_vocab_size - len(vocab)):
        vocab[word] = len(vocab)
    return vocab


def encode_for_textcnn(texts: List[str], vocab: Dict[str, int], max_len: int = 64) -> torch.Tensor:
    ids = []
    for text in texts:
        tokens = text.lower().split()[:max_len]
        row = [vocab.get(t, vocab["<unk>"]) for t in tokens]
        row += [vocab["<pad>"]] * (max_len - len(row))
        ids.append(row)
    return torch.tensor(ids, dtype=torch.long)


def train_group_a(
    baseline_name: str,
    train_texts: List[str], train_labels: List[int],
    val_texts: List[str], val_labels: List[int],
    test_texts: List[str], test_labels: List[int],
    args: argparse.Namespace,
    run_id: str,
) -> Dict:
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    num_classes = SentimentState.num_classes()

    class_weights = compute_class_weights(train_labels, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    if baseline_name == "textcnn":
        vocab = build_vocab(train_texts)
        model = get_baseline_model(
            "textcnn", vocab_size=len(vocab), num_classes=num_classes,
            use_cuda=not args.no_cuda,
        )

        def encode(texts):
            return {"input_ids": encode_for_textcnn(texts, vocab)}
    else:
        model_name = BASELINE_REGISTRY[baseline_name]["default_args"]["model_name"]
        tokenizer = MultilingualTokenizer(model_name)
        model = get_baseline_model(
            baseline_name, num_classes=num_classes, use_cuda=not args.no_cuda,
        )

        def encode(texts):
            return tokenizer.encode_batch(texts, max_length=args.max_token_length)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def run_epoch(texts, labels, train: bool) -> Tuple[float, List[int], List[int]]:
        model.train(train)
        total_loss, n_batches = 0.0, 0
        all_true, all_pred = [], []
        indices = list(range(len(texts)))
        if train:
            import random
            random.shuffle(indices)

        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start:start + args.batch_size]
            batch_texts = [texts[i] for i in batch_idx]
            batch_labels = torch.tensor([labels[i] for i in batch_idx], dtype=torch.long, device=device)

            encoded = encode(batch_texts)
            with torch.set_grad_enabled(train):
                logits = model(**encoded)
                loss = loss_fn(logits, batch_labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            all_true.extend(batch_labels.cpu().tolist())
            all_pred.extend(logits.argmax(dim=-1).cpu().tolist())

        return total_loss / max(n_batches, 1), all_true, all_pred

    best_val_loss = float("inf")
    training_log = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, _, _ = run_epoch(train_texts, train_labels, train=True)
        val_loss, val_true, val_pred = run_epoch(val_texts, val_labels, train=False)
        val_metrics = compute_classification_metrics(val_true, val_pred)
        elapsed = time.time() - t0
        logger.info(
            f"[{baseline_name}/{run_id}] Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_metrics.get('f1_macro', 0):.4f} ({elapsed:.1f}s)"
        )
        training_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                              "val_f1_macro": val_metrics.get("f1_macro", 0), "time_seconds": elapsed})
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    test_loss, test_true, test_pred = run_epoch(test_texts, test_labels, train=False)
    test_metrics = compute_classification_metrics(
        test_true, test_pred, label_names=SentimentState.label_names(),
    )
    test_cm = compute_confusion_matrix(test_true, test_pred, label_names=SentimentState.label_names())
    logger.info(f"[{baseline_name}/{run_id}] TEST sentiment_f1_macro={test_metrics.get('f1_macro', 0):.4f}")

    return {
        "baseline": baseline_name, "run_id": run_id, "training_log": training_log,
        "test": {"sentiment": test_metrics},
        "confusion_matrices": {"sentiment": test_cm.tolist()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Group B: sequence ablation baselines (embeddings -> LSTMOnly / AttentionOnly)
# ══════════════════════════════════════════════════════════════════════════════

def encode_sequence_batch(
    tokenizer: MultilingualTokenizer,
    embedder: DomainAdaptedEmbeddings,
    texts_batch: List[List[str]],
    max_token_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Mirrors OpinionEvolutionTracker.encode_texts() -- see src/model.py."""
    batch_size = len(texts_batch)
    seq_lens = [len(t) for t in texts_batch]
    max_seq_len = max(seq_lens)
    embedding_dim = embedder.get_embedding_dim()

    all_embeddings = torch.zeros(batch_size, max_seq_len, embedding_dim, device=device)
    for i, texts in enumerate(texts_batch):
        if not texts:
            continue
        tokenized = tokenizer.encode_batch(texts, max_length=max_token_length)
        with torch.no_grad():
            emb = embedder.generate_embeddings(tokenized, strategy="cls")
        all_embeddings[i, :len(texts), :] = emb
    return all_embeddings


def train_group_b(
    baseline_name: str,
    train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
    args: argparse.Namespace,
    run_id: str,
    class_weights: torch.Tensor,
) -> Dict:
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")

    tokenizer = MultilingualTokenizer(args.model_name)
    embedder = DomainAdaptedEmbeddings(
        model_name=args.model_name, use_cuda=not args.no_cuda, finetune_layers=0,
    )
    embedder.eval()

    if baseline_name == "lstm_only":
        model = get_baseline_model(
            "lstm_only", embedding_dim=embedder.get_embedding_dim(),
            num_classes=SentimentState.num_classes(), use_cuda=not args.no_cuda,
        )
    else:
        model = get_baseline_model(
            "attention_only", embedding_dim=embedder.get_embedding_dim(),
            num_classes=SentimentState.num_classes(), use_cuda=not args.no_cuda,
        )

    sentiment_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, weight=class_weights["sentiment"].to(device))
    trajectory_loss_fn = nn.CrossEntropyLoss(weight=class_weights["trajectory"].to(device))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def run_epoch(loader, train: bool):
        model.train(train)
        total_loss, n_batches = 0.0, 0
        sent_true, sent_pred, traj_true, traj_pred = [], [], [], []
        pred_sequences, seq_lens_all = [], []

        for batch in loader:
            texts_batch = batch["texts"]
            sentiments = batch["sentiments"].to(device)
            trajectories = batch["trajectories"].to(device)
            seq_lens = batch["seq_lens"]

            embeddings = encode_sequence_batch(
                tokenizer, embedder, texts_batch, args.max_token_length, device,
            )

            with torch.set_grad_enabled(train):
                if baseline_name == "attention_only":
                    max_len = embeddings.size(1)
                    mask = torch.zeros(len(seq_lens), max_len, dtype=torch.bool, device=device)
                    for i, length in enumerate(seq_lens):
                        mask[i, :length] = True
                    out = model(embeddings, mask=mask)
                else:
                    out = model(embeddings, seq_lens=seq_lens)

                sent_logits, traj_logits = out["sentiment_logits"], out["trajectory_logits"]
                loss = (
                    sentiment_loss_fn(sent_logits.reshape(-1, sent_logits.shape[-1]), sentiments.reshape(-1))
                    + trajectory_loss_fn(traj_logits, trajectories)
                )

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            sent_preds = sent_logits.argmax(dim=-1)
            traj_preds = traj_logits.argmax(dim=-1)
            for i, sl in enumerate(seq_lens):
                sent_true.extend(sentiments[i, :sl].cpu().tolist())
                sent_pred.extend(sent_preds[i, :sl].cpu().tolist())
                pred_sequences.append(sent_preds[i, :sl].cpu().tolist())
                seq_lens_all.append(sl)
            traj_true.extend(trajectories.cpu().tolist())
            traj_pred.extend(traj_preds.cpu().tolist())

        return (total_loss / max(n_batches, 1), sent_true, sent_pred,
                traj_true, traj_pred, pred_sequences, seq_lens_all)

    best_val_loss = float("inf")
    training_log = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, *_ = run_epoch(train_loader, train=True)
        val_loss, vs_t, vs_p, vt_t, vt_p, *_ = run_epoch(val_loader, train=False)
        val_sent_f1 = compute_classification_metrics(vs_t, vs_p).get("f1_macro", 0)
        val_traj_f1 = compute_classification_metrics(vt_t, vt_p).get("f1_macro", 0)
        elapsed = time.time() - t0
        logger.info(
            f"[{baseline_name}/{run_id}] Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_sent_f1={val_sent_f1:.4f} val_traj_f1={val_traj_f1:.4f} ({elapsed:.1f}s)"
        )
        training_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                              "val_sentiment_f1": val_sent_f1, "val_trajectory_f1": val_traj_f1,
                              "time_seconds": elapsed})
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    _, ts_t, ts_p, tt_t, tt_p, pred_seqs, seq_lens_all = run_epoch(test_loader, train=False)
    sent_metrics = compute_classification_metrics(ts_t, ts_p, label_names=SentimentState.label_names())
    traj_metrics = compute_classification_metrics(tt_t, tt_p, label_names=TrajectoryType.label_names())
    sent_cm = compute_confusion_matrix(ts_t, ts_p, label_names=SentimentState.label_names())
    traj_cm = compute_confusion_matrix(tt_t, tt_p, label_names=TrajectoryType.label_names())
    scs = sequence_consistency_score(pred_seqs, seq_lens_all)
    logger.info(
        f"[{baseline_name}/{run_id}] TEST sentiment_f1={sent_metrics.get('f1_macro', 0):.4f} "
        f"trajectory_f1={traj_metrics.get('f1_macro', 0):.4f} scs={scs.get('scs_mean', 0):.4f}"
    )

    return {
        "baseline": baseline_name, "run_id": run_id, "training_log": training_log,
        "test": {"sentiment": sent_metrics, "trajectory": traj_metrics, "scs": scs},
        "confusion_matrices": {"sentiment": sent_cm.tolist(), "trajectory": traj_cm.tolist()},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════════

def train_one_baseline(baseline_name: str, args: argparse.Namespace) -> Dict:
    run_id = args.domain if args.domain == "amazon" else f"dravidian_{args.language}"
    logger.info(f"\n{'='*70}\n  Training baseline: {baseline_name} on {run_id}\n{'='*70}")

    if args.domain == "amazon":
        train_seq = AmazonSequenceDataset(split="train", max_seq_len=args.max_seq_len)
        val_seq = AmazonSequenceDataset(split="val", max_seq_len=args.max_seq_len)
        test_seq = AmazonSequenceDataset(split="test", max_seq_len=args.max_seq_len)
        train_loader = get_amazon_dataloader(split="train", batch_size=args.batch_size, max_seq_len=args.max_seq_len)
        val_loader = get_amazon_dataloader(split="val", batch_size=args.batch_size, max_seq_len=args.max_seq_len)
        test_loader = get_amazon_dataloader(split="test", batch_size=args.batch_size, max_seq_len=args.max_seq_len)
    else:
        train_seq = DravidianSequenceDataset(language=args.language, split="train")
        val_seq = DravidianSequenceDataset(language=args.language, split="val")
        test_seq = DravidianSequenceDataset(language=args.language, split="test")
        train_loader = get_dravidian_sequence_dataloader(language=args.language, split="train", batch_size=args.batch_size)
        val_loader = get_dravidian_sequence_dataloader(language=args.language, split="val", batch_size=args.batch_size)
        test_loader = get_dravidian_sequence_dataloader(language=args.language, split="test", batch_size=args.batch_size)

    if baseline_name in GROUP_A:
        train_texts, train_labels = flatten_to_reviews(train_seq)
        val_texts, val_labels = flatten_to_reviews(val_seq)
        test_texts, test_labels = flatten_to_reviews(test_seq)
        result = train_group_a(
            baseline_name, train_texts, train_labels, val_texts, val_labels,
            test_texts, test_labels, args, run_id,
        )
    else:
        sentiments, trajectories = [], []
        for seq in train_seq.sequences:
            sentiments.extend(seq["sentiments"])
            trajectories.append(seq["trajectory"])
        class_weights = {
            "sentiment": compute_class_weights(sentiments, SentimentState.num_classes()),
            "trajectory": compute_class_weights(trajectories, TrajectoryType.num_classes()),
        }
        result = train_group_b(
            baseline_name, train_loader, val_loader, test_loader, args, run_id, class_weights,
        )

    # Save result
    logs_dir = os.path.join(WORKSPACE_ROOT, "outputs", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    out_path = os.path.join(logs_dir, f"baseline_{baseline_name}_{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {out_path}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline comparison models")
    parser.add_argument("--baseline", type=str, default="all",
                         choices=list(BASELINE_REGISTRY.keys()) + ["all"])
    parser.add_argument("--domain", type=str, default="amazon", choices=["amazon", "dravidian"])
    parser.add_argument("--language", type=str, default="tamil", choices=["tamil", "malayalam", "kannada"])
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased",
                         help="Encoder for Group B baselines (lstm_only/attention_only)")
    parser.add_argument("--max_token_length", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    names = list(BASELINE_REGISTRY.keys()) if args.baseline == "all" else [args.baseline]
    all_results = {}
    for name in names:
        all_results[name] = train_one_baseline(name, args)

    logger.info("\n" + "=" * 70)
    logger.info("  BASELINE TRAINING SUMMARY")
    logger.info("=" * 70)
    for name, result in all_results.items():
        test = result["test"]
        sent_f1 = test["sentiment"].get("f1_macro", 0)
        traj_f1 = test.get("trajectory", {}).get("f1_macro", None)
        line = f"  {name:<16} sentiment_f1={sent_f1:.4f}"
        if traj_f1 is not None:
            line += f" trajectory_f1={traj_f1:.4f}"
        logger.info(line)


if __name__ == "__main__":
    main()
