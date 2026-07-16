"""
=============================================================================
Domain 1 — Amazon Reviews 2023 Sequence Builder (Direct HF JSONL Method)
=============================================================================
This script downloads a lightweight category ('All_Beauty') directly from HuggingFace
as a raw JSONL file, cleans review text, reconstructs chronological user review 
sequences (same user, sorted by date, minimum length of 3), and saves the resulting
sequential dataset.

Bypasses HF Datasets / PyArrow DLL blockers.
"""

import os
import sys
import logging
import json
import urllib.request
import shutil
import pandas as pd
from datetime import datetime

# Add src/ to system path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src"))

from preprocessing import PreprocessingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def download_and_build():
    logger.info("=" * 60)
    logger.info("Domain 1: Amazon Reviews 2023 Direct JSONL Sequence Builder")
    logger.info("=" * 60)
    
    # 1. Ensure directories exist
    raw_dir = os.path.join(WORKSPACE_ROOT, "data", "raw")
    preprocessed_dir = os.path.join(WORKSPACE_ROOT, "data", "preprocessed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # 2. Direct download from HuggingFace LFS
    url = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/All_Beauty.jsonl"
    local_jsonl = os.path.join(raw_dir, "All_Beauty.jsonl")
    
    logger.info(f"Downloading from: {url}")
    logger.info(f"Saving dataset to: {local_jsonl}")
    
    try:
        # Check if already downloaded to save bandwidth
        if os.path.exists(local_jsonl) and os.path.getsize(local_jsonl) > 100000000:
            logger.info("  ✓ File already exists locally. Skipping download.")
        else:
            # Download file using urllib
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, timeout=60) as response, open(local_jsonl, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            logger.info("  ✓ Download complete!")
    except Exception as e:
        logger.error(f"Failed to download dataset from HuggingFace: {e}")
        sys.exit(1)
        
    # 3. Read and parse the .jsonl file
    logger.info("Parsing dataset and applying text preprocessing...")
    
    # Initialize basic preprocessing pipeline (English-only, no code-mix stats needed)
    pipeline = PreprocessingPipeline(filter_mode="keep", include_script_features=False, verbose=False)
    
    # Star rating mapping: 4-5 -> Positive (0), 1-2 -> Negative (1), 3 -> Neutral/Mixed (2)
    def map_stars_to_sentiment(rating):
        if rating >= 4.0:
            return 0  # Positive
        elif rating <= 2.0:
            return 1  # Negative
        else:
            return 2  # Neutral/Mixed
            
    # Read the JSONL lines directly
    user_reviews = {}
    total_reviews = 0
    
    # Limit dataset parsing to avoid extreme memory usage (All_Beauty has ~700K reviews total, we only need a good pool)
    max_records_to_parse = 250000
    
    try:
        with open(local_jsonl, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_records_to_parse:
                    break
                
                review = json.loads(line.strip())
                user_id = review.get("user_id")
                
                # We need user_id, text, and timestamp to build sequences
                if not user_id or "text" not in review or "timestamp" not in review:
                    continue
                    
                total_reviews += 1
                if total_reviews % 50000 == 0:
                    logger.info(f"  Parsed {total_reviews:,} reviews...")
                
                text_raw = review.get("text", "")
                text_clean = pipeline.preprocess_text(text_raw)
                
                review_item = {
                    "text": text_clean,
                    "text_original": text_raw,
                    "rating": float(review.get("rating", 3.0)),
                    "label_encoded": map_stars_to_sentiment(float(review.get("rating", 3.0))),
                    "asin": str(review.get("asin", "unknown")),
                    "timestamp": int(review.get("timestamp", 0))
                }
                
                if user_id not in user_reviews:
                    user_reviews[user_id] = []
                user_reviews[user_id].append(review_item)
                
    except Exception as e:
        logger.error(f"Error parsing jsonl: {e}")
        sys.exit(1)
        
    logger.info(f"Finished parsing. Total reviews loaded: {total_reviews:,}")
    
    # 4. Filter and build sequences
    logger.info("Building sequences for users with 3 or more reviews...")
    
    sequences = []
    min_seq_length = 3
    
    for user_id, reviews in user_reviews.items():
        if len(reviews) < min_seq_length:
            continue
            
        # Sort user reviews chronologically by timestamp
        sorted_reviews = sorted(reviews, key=lambda x: x["timestamp"])
        
        # Format dates for readability
        for r in sorted_reviews:
            r["date_str"] = datetime.fromtimestamp(r["timestamp"]/1000).strftime('%Y-%m-%d %H:%M:%S')
            
        sequences.append({
            "user_id": user_id,
            "sequence_length": len(sorted_reviews),
            "reviews": sorted_reviews
        })
        
    logger.info(f"✓ Sequence building complete!")
    logger.info(f"  Total sequential users (len >= {min_seq_length}): {len(sequences):,}")
    
    # Calculate stats
    lengths = [seq["sequence_length"] for seq in sequences]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    logger.info(f"  Average sequence length: {avg_len:.1f} reviews")
    logger.info(f"  Maximum sequence length: {max_len} reviews")
    
    # 5. Save preprocessed sequences to JSON
    output_file = os.path.join(preprocessed_dir, "amazon_beauty_sequences.json")
    logger.info(f"Saving sequences to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sequences, f, indent=2, ensure_ascii=False)
        
    # Also save a raw CSV split to data/raw for verification
    raw_csv = os.path.join(raw_dir, "amazon_beauty_raw.csv")
    logger.info(f"Saving raw review snapshot to: {raw_csv}")
    
    # Flatten subset for CSV saving
    flat_data = []
    for seq in sequences[:10000]:  # Cap CSV rows at 10,000 for size
        for rev in seq["reviews"]:
            flat_data.append({
                "user_id": seq["user_id"],
                "asin": rev["asin"],
                "text": rev["text"],
                "rating": rev["rating"],
                "timestamp": rev["timestamp"]
            })
    pd.DataFrame(flat_data).to_csv(raw_csv, index=False, encoding="utf-8")
    
    logger.info("\n" + "=" * 60)
    logger.info("DOMAIN 1 DATASET PREPARATION COMPLETE!")
    logger.info("=" * 60)

if __name__ == "__main__":
    download_and_build()
