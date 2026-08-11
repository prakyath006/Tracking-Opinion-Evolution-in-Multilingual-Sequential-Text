"""
=============================================================================
Convert Amazon Beauty Sequences JSON -> CSV (reconciled product-category build)
=============================================================================
Flattens data/preprocessed/amazon_beauty_sequences_by_category.json (built by
scripts/build_amazon_sequences.py from the full 701,528-review All_Beauty
pull, grouped by user_id, >=3 reviews per sequence -- the strategy chosen
over exact-product grouping) into the flat CSV AmazonSequenceDataset reads.

This supersedes scripts/convert_amazon_json_to_csv.py, which converted the
older amazon_beauty_sequences.json (a 250,000-review capped pull, 5,324
sequences). That older JSON->CSV pair is left on disk for reference but is
no longer the source AmazonSequenceDataset should read from -- this script
overwrites data/preprocessed/amazon_beauty_sequences.csv, the exact path
AmazonSequenceDataset defaults to (src/dataset.py), so no change to the
Dataset class itself is needed.

Output columns:
  user_id, sequence_position, sequence_length, text, text_original,
  rating, label_encoded, parent_asin, asin, timestamp, date_str

label_encoded is included for parity with the Dravidian preprocessed CSVs
(which keep a label_encoded cache alongside the raw label) and is computed
via SentimentState.from_rating() -- the same ontology entry point
AmazonSequenceDataset itself uses -- so it is guaranteed to already agree
with the ontology; it is not read back by AmazonSequenceDataset (see
src/dataset.py's comment on why label_encoded was dropped from that read
path in Step 3 of the ontology wiring), it is just kept as a human-readable
cross-check column.
"""

import sys
import json
import csv
import os

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src"))

from ontology import SentimentState  # noqa: E402

INPUT_JSON = os.path.join(
    WORKSPACE_ROOT, "data", "preprocessed", "amazon_beauty_sequences_by_category.json"
)
OUTPUT_CSV = os.path.join(
    WORKSPACE_ROOT, "data", "preprocessed", "amazon_beauty_sequences.csv"
)

FIELDNAMES = [
    "user_id", "sequence_position", "sequence_length",
    "text", "text_original", "rating", "label_encoded",
    "parent_asin", "asin", "timestamp", "date_str",
]


def convert():
    print(f"Loading JSON from: {INPUT_JSON}")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        sequences = json.load(f)

    print(f"Total sequences (groups): {len(sequences):,}")

    rows = []
    for seq in sequences:
        # group_key is a list because key_fields=["user_id"] for the
        # product-category strategy; user_id is its only element.
        user_id = seq["group_key"][0]
        seq_len = seq["sequence_length"]
        for pos, review in enumerate(seq["reviews"], start=1):
            rating = review["rating"]
            rows.append({
                "user_id": user_id,
                "sequence_position": pos,
                "sequence_length": seq_len,
                "text": review["text"],
                "text_original": review["text_original"],
                "rating": rating,
                "label_encoded": SentimentState.from_rating(rating).value,
                "parent_asin": review["parent_asin"],
                "asin": review["asin"],
                "timestamp": review["timestamp"],
                "date_str": review["date_str"],
            })

    print(f"Total review rows (flattened): {len(rows):,}")

    if os.path.exists(OUTPUT_CSV):
        old_size = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
        print(f"Overwriting existing {OUTPUT_CSV} ({old_size:.2f} MB)")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    file_size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
    print(f"\n[OK] CSV saved to: {OUTPUT_CSV}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Columns: {FIELDNAMES}")
    print(f"  Total sequences: {len(sequences):,}")
    print(f"  Total rows: {len(rows):,}")


if __name__ == "__main__":
    convert()
