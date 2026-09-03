"""
=============================================================================
Step A5 — WSD Evaluation Metrics
=============================================================================
Evaluates the WSD module against the real Dravidian corpus:
  1. Sense-disambiguation accuracy (on a sampled set, flagged for review)
  2. Coverage: % of ambiguous words resolved
  3. Improvement over Most-Common-Sense (MCS) baseline

Output: outputs/metrics/module2_wsd_results.md
=============================================================================
"""

import os
import sys
import json
import random
import logging
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from wsd import WordSenseDisambiguator, MostCommonSenseBaseline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_corpus_texts(data_dir: str, max_per_lang: int = 5000):
    """Load preprocessed Dravidian texts."""
    texts = {}
    files = {
        "tamil": "tamil_sentiment_train_preprocessed.csv",
        "malayalam": "mal_sentiment_train_preprocessed.csv",
        "kannada": "kannada_sentiment_train_preprocessed.csv",
    }
    for lang, fname in files.items():
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "text" in df.columns:
                all_texts = df["text"].dropna().astype(str).tolist()
                texts[lang] = all_texts[:max_per_lang]
                logger.info(f"Loaded {len(texts[lang])} {lang} texts")
            else:
                logger.warning(f"No 'text' column in {fname}")
        else:
            logger.warning(f"File not found: {path}")
    return texts


def evaluate_wsd(texts_by_lang, wsd, output_dir):
    """Run full WSD evaluation."""
    results = {
        "per_language": {},
        "overall": {},
        "aspect_distribution": {},
        "sample_annotations": [],
    }

    all_texts = []
    for lang, texts in texts_by_lang.items():
        all_texts.extend(texts)

        # Per-language coverage
        stats = wsd.get_coverage_stats(texts)
        results["per_language"][lang] = stats
        logger.info(f"{lang}: coverage={stats['coverage_pct']}%, "
                     f"resolution={stats['resolution_pct']}%")

    # Overall coverage
    results["overall"] = wsd.get_coverage_stats(all_texts)

    # Aspect distribution across corpus
    results["aspect_distribution"] = wsd.get_aspect_distribution(all_texts)

    # Sample 100 annotations for human review (Step A5 requirement)
    sample_size = min(100, len(all_texts))
    sampled_texts = random.sample(all_texts, sample_size)
    for text in sampled_texts:
        annotations = wsd.process(text)
        if annotations:
            results["sample_annotations"].append({
                "text": text[:200],
                "annotations": [
                    {"word": w, "aspect": a, "confidence": c}
                    for w, a, c in annotations
                ],
                "human_verified": False,  # Flagged for review
            })

    # MCS Baseline comparison
    logger.info("Training MCS baseline...")
    mcs = MostCommonSenseBaseline(wsd.lexicon)
    train_texts = all_texts[:len(all_texts) // 2]
    test_texts = all_texts[len(all_texts) // 2:]
    mcs.fit(train_texts)

    # Compare WSD vs MCS on test set
    wsd_correct = 0
    mcs_correct = 0
    total_compared = 0

    for text in test_texts[:1000]:
        wsd_results = wsd.process(text)
        for word, wsd_aspect, wsd_conf in wsd_results:
            mcs_aspect, mcs_conf = mcs.predict(word)
            total_compared += 1
            # Since we don't have gold labels, compare agreement
            # and confidence as proxy
            if wsd_conf > 0.5:
                wsd_correct += 1
            if mcs_conf > 0.5:
                mcs_correct += 1

    results["baseline_comparison"] = {
        "total_compared": total_compared,
        "wsd_high_confidence": wsd_correct,
        "mcs_high_confidence": mcs_correct,
        "wsd_high_conf_pct": round(
            wsd_correct / total_compared * 100, 2
        ) if total_compared > 0 else 0,
        "mcs_high_conf_pct": round(
            mcs_correct / total_compared * 100, 2
        ) if total_compared > 0 else 0,
    }

    return results


def write_report(results, output_path):
    """Write evaluation results as markdown report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "# Module 2 — WSD Evaluation Results\n",
        "## 1. Coverage Statistics\n",
        "### Overall\n",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]

    overall = results["overall"]
    for k, v in overall.items():
        lines.append(f"| {k} | {v} |")

    lines.append("\n### Per Language\n")
    lines.append("| Language | Total Words | Aspect Words | Coverage % | Ambiguous | Resolved | Resolution % |")
    lines.append("|----------|------------|-------------|-----------|-----------|----------|-------------|")

    for lang, stats in results["per_language"].items():
        lines.append(
            f"| {lang} | {stats['total_words']} | {stats['aspect_words']} | "
            f"{stats['coverage_pct']}% | {stats['ambiguous_words']} | "
            f"{stats['resolved_words']} | {stats['resolution_pct']}% |"
        )

    lines.append("\n## 2. Aspect Distribution Across Corpus\n")
    lines.append("| Aspect | Frequency |")
    lines.append("|--------|-----------|")
    for aspect, count in sorted(
        results["aspect_distribution"].items(), key=lambda x: -x[1]
    ):
        lines.append(f"| {aspect} | {count} |")

    lines.append("\n## 3. WSD vs Most-Common-Sense Baseline\n")
    bc = results["baseline_comparison"]
    lines.append(f"| Metric | WSD (Context) | MCS (Baseline) |")
    lines.append(f"|--------|--------------|----------------|")
    lines.append(
        f"| High-confidence predictions | {bc['wsd_high_confidence']} "
        f"({bc['wsd_high_conf_pct']}%) | {bc['mcs_high_confidence']} "
        f"({bc['mcs_high_conf_pct']}%) |"
    )
    lines.append(f"| Total compared | {bc['total_compared']} | {bc['total_compared']} |")

    lines.append(
        f"\n## 4. Sample Annotations (Flagged for Human Review)\n"
    )
    lines.append(
        f"**{len(results['sample_annotations'])} samples** with aspect "
        f"annotations below. `human_verified: false` — requires manual check.\n"
    )
    for i, sample in enumerate(results["sample_annotations"][:20]):
        lines.append(f"### Sample {i+1}")
        lines.append(f"**Text:** {sample['text']}\n")
        for ann in sample["annotations"]:
            lines.append(
                f"- `{ann['word']}` → **{ann['aspect']}** "
                f"(conf: {ann['confidence']:.2f})"
            )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also save raw JSON
    json_path = output_path.replace(".md", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"Report written to {output_path}")
    logger.info(f"Raw data written to {json_path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "preprocessed")
    output_path = os.path.join(
        base_dir, "outputs", "metrics", "module2_wsd_results.md"
    )

    print("=" * 70)
    print("Module 2 — WSD Evaluation")
    print("=" * 70)

    # Load corpus
    texts_by_lang = load_corpus_texts(data_dir)
    if not texts_by_lang:
        print("ERROR: No preprocessed data found. Run preprocessing first.")
        sys.exit(1)

    # Initialize WSD
    wsd = WordSenseDisambiguator()

    # Evaluate
    results = evaluate_wsd(texts_by_lang, wsd, output_path)

    # Write report
    write_report(results, output_path)

    print(f"\nResults saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
