"""
=============================================================================
Full Pipeline Sanity Check (no training)
=============================================================================
Loads a few real batches from both domains through the complete pipeline --
tokenizer -> embeddings -> Bi-LSTM -> attention -> classifier heads -- and
confirms tensor shapes, absence of errors, and that labels decode to
sensible ontology states. Does not train (no backward pass, no optimizer).

Encoder substitution
--------------------
This defaults to sentence-transformers/all-MiniLM-L6-v2 rather than
bert-base-multilingual-cased / xlm-roberta-base. Both src/tokenization.py's
MultilingualTokenizer and src/embeddings.py's DomainAdaptedEmbeddings read
architecture from the loaded model's own config (hidden_size, encoder layer
count) rather than hardcoding BERT/XLM-R specifics, so any AutoModel-
compatible checkpoint is a legitimate substitute for exercising the same
code path.

The substitution exists because this environment's egress bandwidth to
huggingface.co appears to throttle/stall on large binary files specifically
(observed: tokenizer files of ~1-2MB downloaded normally; a 20MB ranged
request against bert-base-multilingual-cased's weights file transferred
~1.4MB in 40s, and a full from_pretrained() call left the weights file at
0 bytes after 8+ minutes) -- mBERT (~680MB) and XLM-R (~1.1GB) are not
practically downloadable here. all-MiniLM-L6-v2 (~88MB) was already fully
cached locally from earlier work, so it exercises a REAL transformer forward
pass end-to-end with no network dependency, rather than mocking the encoder
output. Actual training must still use bert-base-multilingual-cased /
xlm-roberta-base, on a machine (e.g. Colab) with normal HuggingFace
bandwidth -- pass --model_name to override.

Usage:
    python scripts/sanity_check_pipeline.py
    python scripts/sanity_check_pipeline.py --model_name bert-base-multilingual-cased
=============================================================================
"""

import os
import sys
import argparse
import logging

import torch

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src"))

from model import OpinionEvolutionTracker
from dataset import get_amazon_dataloader, get_dravidian_sequence_dataloader
from ontology import SentimentState, TransitionType, TrajectoryType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_batch(model: OpinionEvolutionTracker, batch: dict, domain_label: str) -> None:
    texts_batch = batch["texts"]
    sentiments = batch["sentiments"]
    trends = batch["trends"]
    trajectories = batch["trajectories"]
    seq_lens = batch["seq_lens"]
    padding_mask = batch["padding_mask"]

    batch_size = len(texts_batch)
    max_seq_len = max(seq_lens)

    with torch.no_grad():
        predictions = model(texts_batch, seq_lens=seq_lens, padding_mask=padding_mask)

    sent_logits = predictions["sentiment_logits"]
    trend_logits = predictions["trend_logits"]
    traj_logits = predictions["trajectory_logits"]
    attn_weights = predictions["attention_weights"]

    # ── Shape assertions ──
    assert sent_logits.shape == (batch_size, max_seq_len, SentimentState.num_classes()), sent_logits.shape
    assert trend_logits.shape == (batch_size, max_seq_len, TransitionType.num_classes()), trend_logits.shape
    assert traj_logits.shape == (batch_size, TrajectoryType.num_classes()), traj_logits.shape
    assert sentiments.shape == (batch_size, max_seq_len)
    assert trends.shape == (batch_size, max_seq_len)
    assert trajectories.shape == (batch_size,)
    assert not torch.isnan(sent_logits).any(), "NaN in sentiment_logits"
    assert not torch.isnan(trend_logits).any(), "NaN in trend_logits"
    assert not torch.isnan(traj_logits).any(), "NaN in trajectory_logits"

    # ── Label decode sanity ──
    sent_preds = sent_logits.argmax(dim=-1)
    traj_preds = traj_logits.argmax(dim=-1)

    logger.info(f"[{domain_label}] batch_size={batch_size} max_seq_len={max_seq_len} "
                f"seq_lens={seq_lens}")
    logger.info(f"[{domain_label}] sentiment_logits {tuple(sent_logits.shape)} | "
                f"trend_logits {tuple(trend_logits.shape)} | "
                f"trajectory_logits {tuple(traj_logits.shape)} | "
                f"attention_weights {tuple(attn_weights.shape)}")

    for i in range(min(2, batch_size)):
        sl = seq_lens[i]
        true_states = [SentimentState(s).name for s in sentiments[i, :sl].tolist()]
        pred_states = [SentimentState(s).name for s in sent_preds[i, :sl].tolist()]
        true_traj = TrajectoryType(trajectories[i].item()).name
        pred_traj = TrajectoryType(traj_preds[i].item()).name
        logger.info(f"[{domain_label}] sample {i}: true_sentiments={true_states}")
        logger.info(f"[{domain_label}] sample {i}: pred_sentiments={pred_states} "
                    f"(untrained, expected near-random)")
        logger.info(f"[{domain_label}] sample {i}: true_trajectory={true_traj}, "
                    f"pred_trajectory={pred_traj} (untrained, expected near-random)")

    logger.info(f"[{domain_label}] OK -- shapes valid, no NaNs, labels decode cleanly.\n")


def main():
    parser = argparse.ArgumentParser(description="Full pipeline sanity check (no training)")
    parser.add_argument("--model_name", type=str,
                         default="sentence-transformers/all-MiniLM-L6-v2",
                         help="Encoder to sanity-check with. Defaults to a small "
                              "already-cached model to avoid large downloads; pass "
                              "bert-base-multilingual-cased for the real thing.")
    parser.add_argument("--num_batches", type=int, default=2,
                         help="Number of batches to check per domain")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("  FULL PIPELINE SANITY CHECK (no training)")
    logger.info(f"  Encoder: {args.model_name}")
    logger.info("=" * 70)

    model = OpinionEvolutionTracker(
        model_name=args.model_name,
        freeze_encoder=True,
        use_cuda=not args.no_cuda,
    )
    model.eval()

    # ── Domain 1: Amazon ──
    amazon_loader = get_amazon_dataloader(split="train", batch_size=args.batch_size)
    for i, batch in enumerate(amazon_loader):
        if i >= args.num_batches:
            break
        check_batch(model, batch, "Domain 1: Amazon")

    # ── Domain 2: Dravidian (all three languages) ──
    for lang in ["tamil", "malayalam", "kannada"]:
        loader = get_dravidian_sequence_dataloader(
            language=lang, split="train", batch_size=args.batch_size,
        )
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break
            check_batch(model, batch, f"Domain 2: {lang}")

    logger.info("=" * 70)
    logger.info("  ALL BATCHES PASSED -- pipeline runs end to end without errors.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
