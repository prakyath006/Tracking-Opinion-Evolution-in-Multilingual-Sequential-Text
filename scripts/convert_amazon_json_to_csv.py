"""
=============================================================================
Convert Amazon Beauty Sequences JSON → CSV
=============================================================================
Flattens the nested JSON (user → reviews list) into a flat CSV file
with one row per review, keeping the user_id and sequence position.

Output columns:
  user_id, sequence_position, sequence_length, text, text_original,
  rating, label_encoded, asin, timestamp, date_str
"""

import json
import csv
import os

# Paths
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_JSON = os.path.join(WORKSPACE_ROOT, "data", "preprocessed", "amazon_beauty_sequences.json")
OUTPUT_CSV = os.path.join(WORKSPACE_ROOT, "data", "preprocessed", "amazon_beauty_sequences.csv")

def convert():
    print(f"Loading JSON from: {INPUT_JSON}")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        sequences = json.load(f)

    print(f"Total users (sequences): {len(sequences):,}")

    # Flatten into rows
    rows = []
    for seq in sequences:
        user_id = seq["user_id"]
        seq_len = seq["sequence_length"]
        for pos, review in enumerate(seq["reviews"], start=1):
            rows.append({
                "user_id": user_id,
                "sequence_position": pos,
                "sequence_length": seq_len,
                "text": review["text"],
                "text_original": review["text_original"],
                "rating": review["rating"],
                "label_encoded": review["label_encoded"],
                "asin": review["asin"],
                "timestamp": review["timestamp"],
                "date_str": review["date_str"]
            })

    print(f"Total review rows (flattened): {len(rows):,}")

    # Write CSV
    fieldnames = [
        "user_id", "sequence_position", "sequence_length",
        "text", "text_original", "rating", "label_encoded",
        "asin", "timestamp", "date_str"
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    file_size_mb = os.path.getsize(OUTPUT_CSV) / (1024 * 1024)
    print(f"\n[OK] CSV saved to: {OUTPUT_CSV}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Columns: {fieldnames}")
    print(f"  Total rows: {len(rows):,}")

if __name__ == "__main__":
    convert()
