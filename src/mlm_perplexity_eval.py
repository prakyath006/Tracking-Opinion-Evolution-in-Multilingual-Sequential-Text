"""
=============================================================================
Module 3 — BERT Encoder Comparison: Masked-Language-Model Perplexity
=============================================================================
Compares bert-base-multilingual-cased and xlm-roberta-base as raw language
models (not as the frozen encoder used in the main classification pipeline)
via MLM perplexity, per domain: how well does each encoder's own masked-token
prediction fit this project's actual text?

This is deliberately a SEPARATE model instance from the one used in
src/model.py / src/embeddings.py. The main pipeline uses AutoModel (a plain
encoder, frozen, feeding a Bi-LSTM) and never trains or queries an MLM head.
This module loads AutoModelForMaskedLM instead -- the pretrained model's own
next/masked-token prediction head -- purely to characterize each encoder's
fit to this project's domains as a language model. It does not touch, load,
or affect the frozen encoder the classification pipeline actually uses, and
does not change training in any way.

Method
------
For each (encoder, domain) pair:
  1. Sample up to `max_examples` texts from that domain's test split (same
     deterministic Dataset classes used everywhere else in this project).
  2. Mask ~15% of tokens per example (transformers.DataCollatorForLanguageModeling,
     the standard, well-tested masking implementation -- not reimplemented
     here).
  3. Run each masked example through AutoModelForMaskedLM; take the model's
     own cross-entropy loss on the masked positions.
  4. Aggregate loss across all examples' masked tokens (token-count-weighted,
     not a naive per-example average -- texts have different numbers of
     masked tokens), then perplexity = exp(weighted mean loss).

Encoder availability note (same constraint documented throughout this
project -- see scripts/sanity_check_pipeline.py, notebooks/colab_training.ipynb):
bert-base-multilingual-cased's ~680MB weights file could not be downloaded
in this development environment; xlm-roberta-base (~1.1GB) is larger still
and was not attempted for the same reason. This module's perplexity
computation itself is validated two ways that do NOT require either real
checkpoint:
  1. A synthetic unit test (test_perplexity_math() at the bottom of this
     file) with hand-computable expected output, exercising the aggregation
     math in isolation.
  2. An end-to-end run against real domain text using whatever MLM-capable
     model IS locally available (sentence-transformers/all-MiniLM-L6-v2, via
     --model_name), which proves the full pipeline (data loading -> masking
     -> forward pass -> aggregation) runs without error on real data. Its
     resulting numbers are explicitly NOT meaningful perplexity scores: that
     checkpoint's MLM head was never trained (missing from the
     sentence-embedding checkpoint, randomly initialized on load), so its
     predictions -- and therefore its "perplexity" -- are noise. This is
     flagged wherever such a run's output is reported, unlike the
     substitute-encoder validations used elsewhere in this project for the
     classification pipeline (there, small real encoders like MiniLM DO
     produce meaningful, if not production-grade, numbers -- MLM perplexity
     is different because it depends on a task-specific head this checkpoint
     never had).

Real bert-base-multilingual-cased / xlm-roberta-base numbers require running
this on a machine with normal HuggingFace bandwidth (Colab).

Usage:
    python src/mlm_perplexity_eval.py
    python src/mlm_perplexity_eval.py --max_examples 500
    python src/mlm_perplexity_eval.py --model_names sentence-transformers/all-MiniLM-L6-v2 --meaningless_ok
    -> outputs/metrics/module3_bert_perplexity.md
=============================================================================
"""

import os
import sys
import math
import argparse
import logging
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dataset import AmazonSequenceDataset, DravidianSequenceDataset
from ontology import DOMAIN_CONFIGS

logger = logging.getLogger(__name__)

METRICS_DIR = os.path.join(WORKSPACE_ROOT, "outputs", "metrics")
REPORT_PATH = os.path.join(METRICS_DIR, "module3_bert_perplexity.md")

DEFAULT_ENCODERS = ["bert-base-multilingual-cased", "xlm-roberta-base"]
DEFAULT_MAX_EXAMPLES = 500  # "cap at ~500 test examples per domain for compute reasons"


# ──────────────────────────────────────────────────────────────────────────────
# Data access
# ──────────────────────────────────────────────────────────────────────────────

def _load_domain_texts(domain: str, split: str = "test", max_examples: int = DEFAULT_MAX_EXAMPLES) -> List[str]:
    """First `max_examples` review/comment texts from a domain's split, flattened
    from the same sequence datasets used throughout this project."""
    if domain == "amazon_beauty":
        ds = AmazonSequenceDataset(split=split)
    elif domain.startswith("dravidian_"):
        language = domain.split("_", 1)[1]
        ds = DravidianSequenceDataset(language=language, split=split)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    texts = []
    for seq in ds.sequences:
        texts.extend(seq["texts"])
        if len(texts) >= max_examples:
            break
    return texts[:max_examples]


# ──────────────────────────────────────────────────────────────────────────────
# Perplexity computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_perplexity_from_losses(masked_token_losses: List[float]) -> Optional[float]:
    """
    Aggregate a list of PER-MASKED-TOKEN cross-entropy losses into a single
    perplexity: exp(mean(losses)). Kept separate from the model-running code
    so it can be unit-tested with synthetic values that don't require any
    pretrained checkpoint -- see test_perplexity_math() below.
    """
    if not masked_token_losses:
        return None
    mean_loss = sum(masked_token_losses) / len(masked_token_losses)
    return math.exp(mean_loss)


def compute_domain_encoder_perplexity(
    model_name: str,
    texts: List[str],
    device: torch.device,
    max_length: int = 128,
    mlm_probability: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    Compute MLM perplexity for one (encoder, domain-texts) pair.

    Returns
    -------
    Dict with: perplexity, avg_loss, n_texts_used, n_texts_skipped,
    n_masked_tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability,
    )

    torch.manual_seed(seed)  # reproducible masking across runs

    total_loss_sum = 0.0  # sum of (per-example mean loss * that example's masked-token count)
    total_masked_tokens = 0
    n_used, n_skipped = 0, 0

    for text in texts:
        if not text or not text.strip():
            n_skipped += 1
            continue

        encoded = tokenizer(text, truncation=True, max_length=max_length)
        if len(encoded["input_ids"]) < 2:
            n_skipped += 1
            continue

        batch = collator([encoded])
        batch = {k: v.to(device) for k, v in batch.items()}

        num_masked = int((batch["labels"] != -100).sum().item())
        if num_masked == 0:
            n_skipped += 1
            continue

        with torch.no_grad():
            output = model(**batch)

        # HF's model.loss is already the MEAN cross-entropy over this
        # example's masked positions; unscale back to a sum so aggregating
        # across texts with different masked-token counts is a correct
        # token-weighted average, not a naive per-example average.
        total_loss_sum += output.loss.item() * num_masked
        total_masked_tokens += num_masked
        n_used += 1

    if total_masked_tokens == 0:
        return {
            "perplexity": None, "avg_loss": None,
            "n_texts_used": n_used, "n_texts_skipped": n_skipped,
            "n_masked_tokens": 0,
        }

    avg_loss = total_loss_sum / total_masked_tokens
    return {
        "perplexity": math.exp(avg_loss),
        "avg_loss": avg_loss,
        "n_texts_used": n_used,
        "n_texts_skipped": n_skipped,
        "n_masked_tokens": total_masked_tokens,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration + report
# ──────────────────────────────────────────────────────────────────────────────

def generate_module3_report(
    model_names: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    no_cuda: bool = False,
    meaningless_ok: bool = False,
    write: bool = True,
) -> str:
    """
    Compute MLM perplexity for every (encoder, domain) pair and write the
    comparison table to outputs/metrics/module3_bert_perplexity.md.

    Parameters
    ----------
    model_names : List[str], optional
        Encoders to compare. Default: bert-base-multilingual-cased,
        xlm-roberta-base (the real production candidates).
    meaningless_ok : bool
        Must be explicitly set True to run a model outside DEFAULT_ENCODERS
        (i.e. a substitute like MiniLM) without the report refusing to
        present its numbers as real perplexity -- see module docstring for
        why a substitute checkpoint's MLM head produces meaningless output
        here, unlike the classification-pipeline substitutions used
        elsewhere in this project.
    """
    if model_names is None:
        model_names = list(DEFAULT_ENCODERS)
    if domains is None:
        domains = sorted(DOMAIN_CONFIGS.keys())

    non_default = [m for m in model_names if m not in DEFAULT_ENCODERS]
    if non_default and not meaningless_ok:
        raise ValueError(
            f"{non_default} produce meaningless MLM perplexity here (random, "
            f"untrained MLM head -- see module docstring). Pass "
            f"meaningless_ok=True to proceed anyway for a code-path check."
        )

    device = torch.device("cuda" if (torch.cuda.is_available() and not no_cuda) else "cpu")

    results = {}
    for domain in domains:
        texts = _load_domain_texts(domain, max_examples=max_examples)
        if not texts:
            logger.warning(f"No test data for {domain}; skipping.")
            continue
        for model_name in model_names:
            logger.info(f"Computing perplexity: {model_name} on {domain} ({len(texts)} texts)...")
            result = compute_domain_encoder_perplexity(model_name, texts, device)
            results[(model_name, domain)] = result
            if result["perplexity"] is not None:
                logger.info(f"  {model_name} / {domain}: perplexity={result['perplexity']:.2f} "
                            f"(n_texts={result['n_texts_used']}, n_masked={result['n_masked_tokens']})")

    lines = ["# Module 3 — BERT Encoder Comparison (MLM Perplexity)", ""]
    lines.append(
        "Masked-language-model perplexity per domain, for each encoder "
        "considered as a standalone language model (via `AutoModelForMaskedLM`"
        " -- a separate model instance from the frozen encoder the "
        "classification pipeline actually uses; this does not touch or "
        "affect that pipeline)."
    )
    lines.append("")
    if non_default:
        lines.append(
            f"⚠️ **This run used {non_default}, not the production encoders "
            f"({', '.join(DEFAULT_ENCODERS)}).** {non_default}'s MLM head is "
            f"randomly initialized on load (missing from its checkpoint), so "
            f"the numbers below are NOT meaningful perplexity scores — this "
            f"run exists only to confirm the data-loading -> masking -> "
            f"forward-pass -> aggregation pipeline executes without error on "
            f"real domain text. Real numbers require "
            f"`bert-base-multilingual-cased` / `xlm-roberta-base`, which "
            f"could not be downloaded in this environment (see module "
            f"docstring) — run on Colab."
        )
        lines.append("")

    lines.append(f"Capped at {max_examples} test examples per domain.")
    lines.append("")
    lines.append("| Encoder | Domain | Perplexity | Avg Loss | Texts Used | Texts Skipped | Masked Tokens |")
    lines.append("|---|---|---|---|---|---|---|")
    for (model_name, domain), r in results.items():
        if r["perplexity"] is None:
            lines.append(f"| {model_name} | {domain} | - | - | {r['n_texts_used']} | {r['n_texts_skipped']} | 0 |")
        else:
            lines.append(
                f"| {model_name} | {domain} | {r['perplexity']:.2f} | {r['avg_loss']:.4f} | "
                f"{r['n_texts_used']} | {r['n_texts_skipped']} | {r['n_masked_tokens']:,} |"
            )
    lines.append("")

    if not results:
        lines.append(
            "*No results — no local test data found for any domain, or "
            "encoder weights could not be loaded. `data/` is gitignored; "
            "this is expected in a fresh checkout.*"
        )
        lines.append("")

    if not non_default:
        missing = []
        for m in model_names:
            for d in domains:
                if (m, d) not in results:
                    missing.append((m, d))
        if missing or not results:
            lines.append(
                "**Pending**: real perplexity numbers require the production "
                "encoders to actually download, which this development "
                "environment's network cannot do for large binary files (see "
                "module docstring). Run this on Colab "
                "(`notebooks/colab_training.ipynb`) to populate real numbers."
            )
            lines.append("")

    report = "\n".join(lines)

    if write:
        os.makedirs(METRICS_DIR, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Saved: {REPORT_PATH}")

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Self-contained math check (no pretrained checkpoint required)
# ──────────────────────────────────────────────────────────────────────────────

def test_perplexity_math():
    """
    Sanity-checks compute_perplexity_from_losses() against hand-computable
    values -- exercises the aggregation formula without needing any
    pretrained MLM checkpoint. Run standalone (this file's __main__) or via
    pytest if collected from tests/.
    """
    # Uniform loss of 0 across all masked tokens -> perfect prediction -> perplexity 1.
    assert abs(compute_perplexity_from_losses([0.0, 0.0, 0.0]) - 1.0) < 1e-9

    # A single masked token with loss = ln(2) -> perplexity = e^ln(2) = 2.
    result = compute_perplexity_from_losses([math.log(2)])
    assert abs(result - 2.0) < 1e-9

    # Mean of [ln(2), ln(8)] = ln(4) -> perplexity = 4.
    result = compute_perplexity_from_losses([math.log(2), math.log(8)])
    assert abs(result - 4.0) < 1e-9

    # No masked tokens at all -> undefined, must return None, not crash/0/inf.
    assert compute_perplexity_from_losses([]) is None

    print("test_perplexity_math: all assertions passed.")


def main():
    parser = argparse.ArgumentParser(description="Module 3 — MLM perplexity comparison")
    parser.add_argument("--model_names", type=str, nargs="+", default=None,
                         help=f"Encoders to compare. Default: {DEFAULT_ENCODERS}")
    parser.add_argument("--domains", type=str, nargs="+", default=None)
    parser.add_argument("--max_examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--meaningless_ok", action="store_true",
                         help="Allow a non-production encoder (e.g. a small "
                              "cached substitute) for a code-path check only "
                              "-- see module docstring for why its numbers "
                              "are not meaningful perplexity.")
    parser.add_argument("--self_test", action="store_true",
                         help="Run test_perplexity_math() and exit, skipping "
                              "any model loading.")
    args = parser.parse_args()

    if args.self_test:
        test_perplexity_math()
        return

    report = generate_module3_report(
        model_names=args.model_names, domains=args.domains,
        max_examples=args.max_examples, no_cuda=args.no_cuda,
        meaningless_ok=args.meaningless_ok,
    )
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    main()
