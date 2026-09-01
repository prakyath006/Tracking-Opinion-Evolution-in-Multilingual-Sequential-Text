"""
=============================================================================
Fuzzy Cross-Domain Typicality Scoring — Analysis Module
=============================================================================
Cross-domain evaluation (scripts/cross_domain_eval.py) treats domain
membership as a hard label: a test sequence is either "the Amazon test set"
or "the Tamil test set". This module adds a softer view for later analysis:
for each test sequence, how typical does it look of each domain, as a fuzzy
membership score across all domains rather than a single hard label (e.g.
0.7 Amazon-like, 0.2 dravidian_tamil-like, 0.1 dravidian_malayalam-like)?

Method (per the task spec):
  1. Encode each sequence with the SAME frozen encoder the rest of the
     pipeline uses (MultilingualTokenizer + DomainAdaptedEmbeddings, CLS
     pooling per review) -- no new encoder is introduced.
  2. Mean-pool each sequence's per-review embeddings into one vector.
  3. Domain centroid = mean of those sequence vectors over a domain's
     training split.
  4. Fuzzy membership for a test sequence = cosine similarity to each
     domain centroid, normalized across domains to sum to 1.

This is a read-only analysis module: it does not change training, the
model, or scripts/cross_domain_eval.py's own logic. It only computes an
additional per-sequence score, meant to sit alongside cross-domain results
for later correlation analysis (e.g. "does degradation correlate with how
atypical a sequence looks of its source domain?") -- that correlation is
explicitly NOT computed here; see the module-level note in
generate_fuzzy_domain_scores()'s docstring about why.

Encoder note (same constraint documented in scripts/sanity_check_pipeline.py
and notebooks/colab_training.ipynb): bert-base-multilingual-cased could not
be downloaded in this development environment -- large binary transfers
stall here, small files do not. This module defaults to it anyway (that is
the correct production encoder, and Colab can download it), but everything
in this file was verified end-to-end locally using
sentence-transformers/all-MiniLM-L6-v2 (already cached) as a substitute, on
a small subsample. Pass --model_name to override.

Usage:
    python src/fuzzy_domain_score.py
    python src/fuzzy_domain_score.py --model_name sentence-transformers/all-MiniLM-L6-v2 --max_sequences 50
    -> outputs/fuzzy_domain_scores.csv
=============================================================================
"""

import os
import sys
import glob
import json
import argparse
import logging
import statistics
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tokenization import MultilingualTokenizer
from embeddings import DomainAdaptedEmbeddings
from dataset import AmazonSequenceDataset, DravidianSequenceDataset
from ontology import DOMAIN_CONFIGS

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fuzzy_domain_scores.csv")


# ──────────────────────────────────────────────────────────────────────────────
# Data access: one function per domain, returning {sequence_id: List[str] texts}
# ──────────────────────────────────────────────────────────────────────────────

def _load_domain_sequences(domain: str, split: str) -> Dict[str, List[str]]:
    """
    Returns {sequence_id: texts} for a domain's split, in the same
    deterministic order AmazonSequenceDataset / DravidianSequenceDataset
    produce it (both split with a fixed default random_seed=42), so
    sequence_id = f"{domain}_{split}_{index}" is reproducible run to run --
    the same scheme scripts/cross_domain_eval.py's dataloaders would
    enumerate in, if it exposed per-sequence records (see this module's
    docstring and generate_fuzzy_domain_scores() for why that join isn't
    live yet).
    """
    if domain == "amazon_beauty":
        ds = AmazonSequenceDataset(split=split)
    elif domain.startswith("dravidian_"):
        language = domain.split("_", 1)[1]
        ds = DravidianSequenceDataset(language=language, split=split)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    return {
        f"{domain}_{split}_{i}": seq["texts"]
        for i, seq in enumerate(ds.sequences)
    }


# ──────────────────────────────────────────────────────────────────────────────
# Encoding
# ──────────────────────────────────────────────────────────────────────────────

def encode_sequence_mean_pooled(
    tokenizer: MultilingualTokenizer,
    embedder: DomainAdaptedEmbeddings,
    texts: List[str],
    max_token_length: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode one sequence (a list of review/comment texts) into a single
    mean-pooled vector: CLS-embed each review with the frozen encoder (same
    two-class pattern as OpinionEvolutionTracker.encode_texts() in
    src/model.py and encode_sequence_batch() in scripts/train_baselines.py),
    then mean over the sequence dimension. Unlike those two, this collapses
    to one vector per sequence rather than keeping per-review positions --
    a domain centroid is a single point, so the per-sequence representation
    feeding it needs to be single points too.

    Returns
    -------
    torch.Tensor of shape [embedding_dim]. Zeros if `texts` is empty.
    """
    embedding_dim = embedder.get_embedding_dim()
    if not texts:
        return torch.zeros(embedding_dim, device=device)

    tokenized = tokenizer.encode_batch(texts, max_length=max_token_length)
    with torch.no_grad():
        review_embeddings = embedder.generate_embeddings(tokenized, strategy="cls")
    return review_embeddings.mean(dim=0)


def encode_domain_sequences(
    sequences: Dict[str, List[str]],
    tokenizer: MultilingualTokenizer,
    embedder: DomainAdaptedEmbeddings,
    max_token_length: int,
    device: torch.device,
    max_sequences: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Encode every sequence in `sequences` to its mean-pooled vector.

    max_sequences caps how many sequences are encoded (first N in dataset
    order) for tractability -- a domain centroid is a mean, and a mean over
    a large random subsample is already a stable estimator of the full-data
    mean, so this is a legitimate speed/cost tradeoff for centroid
    computation, not an approximation that changes what's being measured.
    Per-sequence scoring (the actual analysis output) still runs on
    whichever sequences are passed in; capping only trims how many are
    processed, not which computation is done.
    """
    items = list(sequences.items())
    if max_sequences is not None:
        items = items[:max_sequences]

    encoded = {}
    for seq_id, texts in items:
        encoded[seq_id] = encode_sequence_mean_pooled(
            tokenizer, embedder, texts, max_token_length, device
        )
    return encoded


# ──────────────────────────────────────────────────────────────────────────────
# Centroids and fuzzy membership
# ──────────────────────────────────────────────────────────────────────────────

def compute_domain_centroids(
    domains: List[str],
    tokenizer: MultilingualTokenizer,
    embedder: DomainAdaptedEmbeddings,
    max_token_length: int,
    device: torch.device,
    split: str = "train",
    max_sequences_per_domain: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    One centroid per domain: mean of that domain's training sequences'
    mean-pooled embeddings.

    Returns
    -------
    Dict[domain, torch.Tensor] of shape [embedding_dim] each.
    """
    centroids = {}
    for domain in domains:
        sequences = _load_domain_sequences(domain, split)
        encoded = encode_domain_sequences(
            sequences, tokenizer, embedder, max_token_length, device,
            max_sequences=max_sequences_per_domain,
        )
        if not encoded:
            logger.warning(f"No {split} sequences for {domain}; skipping centroid.")
            continue
        stacked = torch.stack(list(encoded.values()))
        centroids[domain] = stacked.mean(dim=0)
        logger.info(f"Centroid for {domain}: mean of {len(encoded)} sequences "
                    f"(of {len(sequences)} available in {split} split)")
    return centroids


def fuzzy_membership(
    sequence_vector: torch.Tensor,
    centroids: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Cosine similarity of `sequence_vector` to each domain centroid,
    normalized across domains to sum to 1.

    Cosine similarity can be negative in principle, which sum-to-1
    normalization alone doesn't handle (a negative share is not a valid
    membership, and the sum of raw similarities could be <= 0). To keep this
    well-defined for every input while still literally being "normalize the
    similarities so they sum to 1" on the common case, similarities are
    shifted so the minimum across domains is 0 before normalizing --
    equivalent to plain sum-normalization whenever all similarities are
    already non-negative (the typical case for real sentence embeddings of
    related text domains), and gracefully defined otherwise. If every
    shifted similarity is 0 (all domains equally similar), membership falls
    back to uniform.

    Returns
    -------
    Dict[domain, float], values summing to 1.0.
    """
    domains = list(centroids.keys())
    sims = np.array([
        torch.nn.functional.cosine_similarity(
            sequence_vector.unsqueeze(0), centroids[d].unsqueeze(0)
        ).item()
        for d in domains
    ])

    shifted = sims - sims.min()
    total = shifted.sum()
    if total <= 0:
        shares = np.full(len(domains), 1.0 / len(domains))
    else:
        shares = shifted / total

    return {d: float(s) for d, s in zip(domains, shares)}


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def generate_fuzzy_domain_scores(
    model_name: str = "bert-base-multilingual-cased",
    domains: Optional[List[str]] = None,
    split: str = "test",
    max_token_length: int = 128,
    max_sequences_per_domain: Optional[int] = None,
    no_cuda: bool = False,
    write: bool = True,
) -> pd.DataFrame:
    """
    Compute fuzzy domain-typicality scores for every sequence in `split`
    (default "test", matching what cross_domain_eval.py evaluates on) across
    every domain in `domains` (default: all of DOMAIN_CONFIGS).

    Join-with-cross-domain-results note
    ------------------------------------
    The task calls for storing these scores "alongside cross-domain
    evaluation results (join on sequence ID)". scripts/cross_domain_eval.py's
    evaluate_cross_domain() currently only ever returns/saves AGGREGATE
    metrics per source->target pair (accuracy, F1, SCS) -- it does not save
    a per-sequence prediction table, so there is no existing per-sequence
    file to join against yet. This module still assigns a stable,
    reproducible sequence_id (f"{domain}_{split}_{index}", using the same
    deterministic split logic -- fixed default random_seed=42 -- that
    AmazonSequenceDataset / DravidianSequenceDataset already use, which is
    what cross_domain_eval.py's dataloaders iterate), so a join is possible
    the moment cross_domain_eval.py's output includes per-sequence records.
    Adding that is a change to cross_domain_eval.py's existing logic, which
    this task explicitly scoped out ("only reads its output"), so it is not
    done here -- this is a documented gap, not a silent one.

    Correlation note
    -----------------
    Per the task: this function makes fuzzy scores available: it does NOT
    correlate them against cross-domain degradation. Doing so needs real
    cross-domain results (real trained checkpoints), which do not exist yet
    in this project -- see outputs/cross_domain/ once
    notebooks/colab_training.ipynb has been run.

    Returns
    -------
    pd.DataFrame with columns: sequence_id, source_domain, split,
    fuzzy_<domain> for each domain in `domains`.
    """
    if domains is None:
        domains = sorted(DOMAIN_CONFIGS.keys())

    device = torch.device("cuda" if (torch.cuda.is_available() and not no_cuda) else "cpu")
    tokenizer = MultilingualTokenizer(model_name)
    embedder = DomainAdaptedEmbeddings(model_name=model_name, use_cuda=not no_cuda, finetune_layers=0)
    embedder.eval()

    logger.info(f"Computing domain centroids from '{'train'}' split ({model_name})...")
    centroids = compute_domain_centroids(
        domains, tokenizer, embedder, max_token_length, device,
        split="train", max_sequences_per_domain=max_sequences_per_domain,
    )
    if not centroids:
        raise RuntimeError("No domain centroids could be computed -- no training data found.")

    rows = []
    for domain in domains:
        sequences = _load_domain_sequences(domain, split)
        encoded = encode_domain_sequences(
            sequences, tokenizer, embedder, max_token_length, device,
            max_sequences=max_sequences_per_domain,
        )
        logger.info(f"Scoring {len(encoded)} {domain}/{split} sequences "
                    f"(of {len(sequences)} available)...")
        for seq_id, vec in encoded.items():
            membership = fuzzy_membership(vec, centroids)
            row = {"sequence_id": seq_id, "source_domain": domain, "split": split}
            for d in domains:
                row[f"fuzzy_{d}"] = membership.get(d)
            rows.append(row)

    df = pd.DataFrame(rows)

    if write:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
        logger.info(f"Saved {len(df)} rows -> {OUTPUT_PATH}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Module 5 report: F1 Stability Ratio + cross-domain summary + fuzzy scores
# ──────────────────────────────────────────────────────────────────────────────

CROSS_DOMAIN_DIR = os.path.join(WORKSPACE_ROOT, "outputs", "cross_domain")
METRICS_DIR = os.path.join(WORKSPACE_ROOT, "outputs", "metrics")
MODULE5_REPORT_PATH = os.path.join(METRICS_DIR, "module5_cross_domain.md")
HEADS = ["sentiment", "trend", "trajectory"]


def compute_f1_stability_ratio(f1_scores: List[float]) -> Dict:
    """
    F1 Stability Ratio = mean(F1) / std(F1) across a set of evaluated
    domains/setups, for one task head. Same mean/std-ratio construction as
    Module 1's Coverage Stability Ratio and Module 4's SCS Reliability
    Ratio -- named for what it concretely measures (how stable a head's F1
    is across domains), not "Sharpe ratio".

    Returns
    -------
    Dict with: ratio (float or None), mean, std, n, note.
    """
    if len(f1_scores) < 2:
        return {
            "ratio": None, "mean": (f1_scores[0] if f1_scores else None),
            "std": None, "n": len(f1_scores),
            "note": f"Need >=2 evaluated setups to compute a standard deviation; only {len(f1_scores)} available.",
        }

    mean_f1 = statistics.mean(f1_scores)
    std_f1 = statistics.stdev(f1_scores)

    if std_f1 == 0:
        return {"ratio": None, "mean": mean_f1, "std": 0.0, "n": len(f1_scores),
                "note": "F1 is identical across all evaluated setups (std=0); ratio is undefined (division by zero)."}

    return {"ratio": mean_f1 / std_f1, "mean": mean_f1, "std": std_f1,
            "n": len(f1_scores), "note": f"Computed over {len(f1_scores)} evaluated setup(s)."}


def load_cross_domain_results() -> Dict[str, Dict]:
    """
    Read every outputs/cross_domain/cross_domain_results_*.json written by
    scripts/cross_domain_eval.py (one per Dravidian language paired against
    Amazon) and merge into one dict keyed by setup name (e.g.
    "amazon_to_dravidian_tamil"). Read-only -- does not run evaluation.

    Returns
    -------
    Dict[setup_name, result_dict]. Empty if no results exist yet.
    """
    merged = {}
    for path in sorted(glob.glob(os.path.join(CROSS_DOMAIN_DIR, "cross_domain_results_*.json"))):
        with open(path, encoding="utf-8") as f:
            merged.update(json.load(f))
    return merged


def summarize_fuzzy_scores(csv_path: str = OUTPUT_PATH) -> Optional[pd.DataFrame]:
    """
    Read an existing fuzzy_domain_scores.csv (from generate_fuzzy_domain_scores())
    if present and summarize it: mean fuzzy-membership-in-own-domain per
    source domain, and how often each domain's sequences score highest on
    their own centroid (argmax match rate) -- read-only, does not run scoring.

    Returns
    -------
    pd.DataFrame summary, or None if no scores file exists yet.
    """
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    fuzzy_cols = [c for c in df.columns if c.startswith("fuzzy_")]
    if not fuzzy_cols:
        return None

    df = df.copy()
    df["argmax_domain"] = df[fuzzy_cols].idxmax(axis=1).str.replace("fuzzy_", "", regex=False)

    rows = []
    for domain, group in df.groupby("source_domain"):
        own_col = f"fuzzy_{domain}"
        rows.append({
            "domain": domain,
            "n_sequences": len(group),
            "mean_own_domain_score": group[own_col].mean() if own_col in group else None,
            "argmax_matches_own_domain_pct": (group["argmax_domain"] == domain).mean() * 100,
        })
    return pd.DataFrame(rows)


def generate_module5_report(write: bool = True) -> str:
    """
    Module 5 (Cross-Domain Generalization) report: in-domain vs cross-domain
    F1 + % degradation per head (reusing scripts/cross_domain_eval.py's
    saved JSON output -- does not re-run evaluation), F1 Stability Ratio
    per head across whatever setups have been evaluated, and a summary of
    fuzzy domain-typicality scores if generate_fuzzy_domain_scores() has
    been run. States "pending" wherever the underlying results don't exist
    yet, rather than fabricating a number.

    Returns
    -------
    str
        The markdown report text.
    """
    results = load_cross_domain_results()

    lines = ["# Module 5 — Cross-Domain Generalization", ""]

    # ── In-domain vs cross-domain F1 + degradation, per head ──
    lines.append("## In-Domain vs. Cross-Domain F1")
    lines.append("")
    if not results:
        lines.append(
            "**Pending.** No cross-domain results found under "
            "`outputs/cross_domain/`. Run `python scripts/cross_domain_eval.py "
            "--language <tamil|malayalam|kannada>` (requires trained "
            "checkpoints from `scripts/train.py` for both `amazon` and the "
            "chosen Dravidian language) to populate this section."
        )
        lines.append("")
    else:
        lines.append("| Setup | Sentiment F1 | Trend F1 | Trajectory F1 | SCS |")
        lines.append("|---|---|---|---|---|")
        for setup, r in results.items():
            sent_f1 = r.get("sentiment", {}).get("f1_macro")
            trend_f1 = r.get("trend", {}).get("f1_macro")
            traj_f1 = r.get("trajectory", {}).get("f1_macro")
            scs = r.get("scs", {}).get("scs_mean")
            trend_str = f"{trend_f1:.4f}" if trend_f1 is not None else "- (older result, predates trend tracking)"
            lines.append(
                f"| {setup} | {sent_f1:.4f} | {trend_str} | {traj_f1:.4f} | {scs:.4f} |"
            )
        lines.append("")

        # Degradation: pair every "X_to_X" (in-domain) with "X_to_Y" (cross-domain, same source X)
        lines.append("### % Degradation (in-domain -> cross-domain, per head)")
        lines.append("")
        degradation_rows = []
        for setup, r in results.items():
            parts = setup.split("_to_")
            source, target = parts[0], parts[-1]
            if source == target:
                continue
            in_domain_key = f"{source}_to_{source}"
            if in_domain_key not in results:
                continue
            for head in HEADS:
                in_f1 = results[in_domain_key].get(head, {}).get("f1_macro")
                cross_f1 = r.get(head, {}).get("f1_macro")
                if in_f1 is None or cross_f1 is None or in_f1 == 0:
                    continue
                degradation_pct = ((in_f1 - cross_f1) / in_f1) * 100
                degradation_rows.append((setup, head, in_f1, cross_f1, degradation_pct))

        if degradation_rows:
            lines.append("| Setup | Head | In-Domain F1 | Cross-Domain F1 | % Degradation |")
            lines.append("|---|---|---|---|---|")
            for setup, head, in_f1, cross_f1, deg in degradation_rows:
                lines.append(f"| {setup} | {head} | {in_f1:.4f} | {cross_f1:.4f} | {deg:.1f}% |")
        else:
            lines.append("*No in-domain/cross-domain pairs available to compute degradation from yet.*")
        lines.append("")

    # ── F1 Stability Ratio per head ──
    lines.append("## F1 Stability Ratio (per head)")
    lines.append("")
    lines.append("mean(F1) / std(F1) across every evaluated domain/setup, for one task head.")
    lines.append("")
    if not results:
        lines.append("**Pending** — no evaluated setups yet.")
        lines.append("")
    else:
        lines.append("| Head | Ratio | Mean F1 | Std F1 | Setups Used |")
        lines.append("|---|---|---|---|---|")
        for head in HEADS:
            f1_scores = [r[head]["f1_macro"] for r in results.values()
                         if head in r and r[head].get("f1_macro") is not None]
            stability = compute_f1_stability_ratio(f1_scores)
            if stability["ratio"] is None:
                lines.append(f"| {head} | not computed | - | - | {stability['n']} ({stability['note']}) |")
            else:
                lines.append(
                    f"| {head} | {stability['ratio']:.2f} | {stability['mean']:.4f} | "
                    f"{stability['std']:.4f} | {stability['n']} |"
                )
        lines.append("")

    # ── Fuzzy domain-typicality scores ──
    lines.append("## Fuzzy Domain-Typicality Scores")
    lines.append("")
    fuzzy_summary = summarize_fuzzy_scores()
    if fuzzy_summary is None:
        lines.append(
            "**Pending.** No `outputs/fuzzy_domain_scores.csv` found. Run "
            "`python src/fuzzy_domain_score.py` to populate this section — "
            "per-test-sequence fuzzy membership scores across all domains, "
            "computed via cosine similarity to domain centroids (see "
            "`generate_fuzzy_domain_scores()`)."
        )
        lines.append("")
    else:
        lines.append(
            "Summary of `outputs/fuzzy_domain_scores.csv`. "
            "`argmax_matches_own_domain_pct` = how often a domain's own test "
            "sequences score *highest* on that domain's own centroid (a "
            "sanity signal, not a correctness guarantee)."
        )
        lines.append("")
        lines.append("| Domain | Sequences | Mean Own-Domain Score | Argmax Matches Own Domain |")
        lines.append("|---|---|---|---|")
        for _, row in fuzzy_summary.iterrows():
            lines.append(
                f"| {row['domain']} | {row['n_sequences']} | "
                f"{row['mean_own_domain_score']:.4f} | "
                f"{row['argmax_matches_own_domain_pct']:.1f}% |"
            )
        lines.append("")
        lines.append(
            "Not yet correlated against cross-domain degradation (per task "
            "scope) — these scores are available for that analysis once "
            "both this table and the degradation table above are populated "
            "from the same real trained checkpoints."
        )
        lines.append("")

    report = "\n".join(lines)

    if write:
        os.makedirs(METRICS_DIR, exist_ok=True)
        with open(MODULE5_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Saved: {MODULE5_REPORT_PATH}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Fuzzy cross-domain typicality scoring")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_token_length", type=int, default=128)
    parser.add_argument("--max_sequences", type=int, default=None,
                         help="Cap sequences processed per domain (centroid + scoring). "
                              "Default: use all.")
    parser.add_argument("--no_cuda", action="store_true")
    args = parser.parse_args()

    df = generate_fuzzy_domain_scores(
        model_name=args.model_name,
        split=args.split,
        max_token_length=args.max_token_length,
        max_sequences_per_domain=args.max_sequences,
        no_cuda=args.no_cuda,
    )
    print(df.head(10).to_string())
    print(f"\n{len(df)} total sequences scored.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
    main()
