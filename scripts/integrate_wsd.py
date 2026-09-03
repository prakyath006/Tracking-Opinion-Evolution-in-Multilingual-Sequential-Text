"""
=============================================================================
Step A6 — WSD Integration Point
=============================================================================
Wires the WSD module as an optional annotation step between preprocessing
and ontology label mapping. Does NOT modify ontology.py or the existing
sentiment pipeline — WSD produces aspect annotations as a standalone output.

Usage:
    python scripts/integrate_wsd.py --data-dir data/preprocessed --output-dir outputs/wsd
=============================================================================
"""

import os
import sys
import argparse
import json
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from wsd import WordSenseDisambiguator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def annotate_dataset(wsd, df, text_col="text"):
    """
    Annotate a preprocessed DataFrame with WSD aspect predictions.

    Adds columns:
      - wsd_aspects: JSON list of detected aspects per row
      - wsd_primary_aspect: the most confident aspect (or 'none')
      - wsd_aspect_count: number of aspect words detected
    """
    aspects_list = []
    primary_list = []
    count_list = []

    for _, row in df.iterrows():
        text = str(row.get(text_col, ""))
        results = wsd.process(text)

        if results:
            aspects = [
                {"word": w, "aspect": a, "confidence": round(c, 3)}
                for w, a, c in results
            ]
            # Primary = highest confidence
            best = max(results, key=lambda x: x[2])
            primary = best[1]
        else:
            aspects = []
            primary = "none"

        aspects_list.append(json.dumps(aspects, ensure_ascii=False))
        primary_list.append(primary)
        count_list.append(len(results))

    df = df.copy()
    df["wsd_aspects"] = aspects_list
    df["wsd_primary_aspect"] = primary_list
    df["wsd_aspect_count"] = count_list
    return df


def main():
    parser = argparse.ArgumentParser(
        description="WSD Integration — annotate preprocessed data with aspect information"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "data", "preprocessed"
        ),
        help="Directory containing preprocessed CSV files",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "outputs", "wsd"),
        help="Directory to save annotated outputs",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Step A6 — WSD Integration")
    print("=" * 70)

    wsd = WordSenseDisambiguator()

    files = {
        "tamil": "tamil_sentiment_train_preprocessed.csv",
        "malayalam": "mal_sentiment_train_preprocessed.csv",
        "kannada": "kannada_sentiment_train_preprocessed.csv",
    }

    summary = {}

    for lang, fname in files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            logger.warning(f"Not found: {path}")
            continue

        logger.info(f"Processing {lang}...")
        df = pd.read_csv(path)
        annotated = annotate_dataset(wsd, df)

        # Save annotated CSV
        out_path = os.path.join(output_dir, f"{lang}_wsd_annotated.csv")
        annotated.to_csv(out_path, index=False)
        logger.info(f"Saved: {out_path}")

        # Summary stats
        aspect_counts = annotated["wsd_primary_aspect"].value_counts().to_dict()
        total_with_aspect = int((annotated["wsd_aspect_count"] > 0).sum())
        summary[lang] = {
            "total_rows": len(annotated),
            "rows_with_aspects": total_with_aspect,
            "coverage_pct": round(total_with_aspect / len(annotated) * 100, 2),
            "aspect_distribution": aspect_counts,
        }

        print(f"\n{lang.upper()}:")
        print(f"  Total rows: {len(annotated)}")
        print(f"  Rows with aspects: {total_with_aspect} ({summary[lang]['coverage_pct']}%)")
        print(f"  Top aspects: {dict(list(aspect_counts.items())[:5])}")

    # Save summary
    summary_path = os.path.join(output_dir, "wsd_integration_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    print("Done!")


if __name__ == "__main__":
    main()
