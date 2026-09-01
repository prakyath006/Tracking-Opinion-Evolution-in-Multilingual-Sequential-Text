"""
=============================================================================
Export Ontology Output — Module 1a Deliverable
=============================================================================
The ontology (src/ontology.py) is applied in memory every time a Dataset
class loads data (see src/dataset.py) and is never persisted -- there is no
file anywhere that shows "raw label -> ontology state" as a saved artifact.

This script creates that artifact: it runs the real preprocessed data from
both domains through the actual ontology entry points (map_labels_to_ontology,
SentimentState.from_rating, TrajectoryType.compute) and saves the result, so
there is something concrete to show -- not just the taxonomy definition, but
the ontology actually applied to real data.

Output (data/ontology_output/, gitignored like the rest of data/):
  amazon_beauty_ontology_mapped.csv   -- per-review: rating -> SentimentState
  amazon_beauty_trajectories.csv      -- per-user-sequence: -> TrajectoryType
  dravidian_<language>_ontology_mapped.csv -- per-comment: label -> SentimentState
  ontology_summary.json               -- the taxonomy itself (get_ontology_summary())

Usage:
    python scripts/export_ontology_output.py
=============================================================================
"""

import os
import sys
import json
import logging

import pandas as pd

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src"))

from ontology import (
    SentimentState,
    TrajectoryType,
    map_labels_to_ontology,
    get_ontology_summary,
    DOMAIN_CONFIGS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PREPROCESSED_DIR = os.path.join(WORKSPACE_ROOT, "data", "preprocessed")
OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "data", "ontology_output")

LANG_FILE_PREFIX = {"tamil": "tamil", "malayalam": "mal", "kannada": "kannada"}


def export_amazon():
    csv_path = os.path.join(PREPROCESSED_DIR, "amazon_beauty_sequences.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"Skipping Amazon: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path, encoding="utf-8")

    # Per-review: rating -> SentimentState (the ontology's entry point for this domain)
    states = [SentimentState.from_rating(r) for r in df["rating"]]
    df_out = df[["user_id", "sequence_position", "text", "rating"]].copy()
    df_out["ontology_sentiment_state"] = [s.name for s in states]
    df_out["ontology_sentiment_id"] = [s.value for s in states]

    out_path = os.path.join(OUTPUT_DIR, "amazon_beauty_ontology_mapped.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Saved {len(df_out):,} rows -> {out_path}")

    # Per-user sequence: -> TrajectoryType
    rows = []
    for user_id, group in df.groupby("user_id", sort=False):
        group_sorted = group.sort_values("sequence_position")
        seq_states = [SentimentState.from_rating(r) for r in group_sorted["rating"]]
        trajectory = TrajectoryType.compute(seq_states)
        rows.append({
            "user_id": user_id,
            "sequence_length": len(seq_states),
            "sentiment_sequence": " -> ".join(s.name for s in seq_states),
            "ontology_trajectory": trajectory.name,
        })
    traj_df = pd.DataFrame(rows)
    traj_path = os.path.join(OUTPUT_DIR, "amazon_beauty_trajectories.csv")
    traj_df.to_csv(traj_path, index=False, encoding="utf-8")
    logger.info(f"Saved {len(traj_df):,} sequences -> {traj_path}")


def export_dravidian(language: str):
    prefix = LANG_FILE_PREFIX[language]
    csv_path = os.path.join(PREPROCESSED_DIR, f"{prefix}_sentiment_train_preprocessed.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"Skipping {language}: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path, encoding="utf-8")
    domain = f"dravidian_{language}"

    states = map_labels_to_ontology(df["label"].astype(str).tolist(), domain)
    df_out = df[["text", "label"]].copy()
    df_out["ontology_sentiment_state"] = [s.name for s in states]
    df_out["ontology_sentiment_id"] = [s.value for s in states]

    out_path = os.path.join(OUTPUT_DIR, f"dravidian_{language}_ontology_mapped.csv")
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    logger.info(f"Saved {len(df_out):,} rows -> {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # The taxonomy itself, saved as a readable file too.
    summary_path = os.path.join(OUTPUT_DIR, "ontology_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(get_ontology_summary(), f, indent=2)
    logger.info(f"Saved ontology summary -> {summary_path}")

    export_amazon()
    for language in ["tamil", "malayalam", "kannada"]:
        export_dravidian(language)

    logger.info(f"\nAll ontology output saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
