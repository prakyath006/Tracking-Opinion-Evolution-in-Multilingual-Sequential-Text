"""
=============================================================================
Domain 1 — Amazon Reviews 2023 Sequence Builder (Product-Aware)
=============================================================================
Pulls the All_Beauty category of McAuley-Lab/Amazon-Reviews-2023 and builds
chronological review sequences for opinion-evolution tracking, structured as
staged, testable steps — matching src/preprocessing.py's convention for
Domain 2 (DataLoader -> LanguageFilter -> ... -> LabelEncoder) rather than
the single-function script this replaces
(scripts/download_and_build_amazon_sequences.py, kept for reference).

Pipeline Stages:
  1. Data Acquisition   — pull raw/review_categories/All_Beauty.jsonl
  2. Sequence Grouping  — user_id + parent_asin (exact-product) first,
                          falling back to user_id-only (product-category,
                          since this pull is already scoped to one category)
                          if exact-product grouping does not yield enough
                          sequences
  3. Chronological Sort — order each group by timestamp
  4. Length Filtering   — keep only groups with >= MIN_SEQUENCE_LENGTH reviews
  5. Reporting          — raw review count, groups formed, groups surviving
                          the filter, for every grouping strategy tried

Dataset note
------------
Task called for the "0core_timestamp_w_his_All_Beauty" config, falling back
to "raw_review_All_Beauty" if unavailable. Checked both:
  - 0core_timestamp_w_his_All_Beauty *is* reachable (benchmark/0core/
    timestamp_w_his/All_Beauty.*.csv) but only has
    [user_id, parent_asin, rating, timestamp, history] — no review text. It
    is a pre-built recommendation-benchmark split, not review data, so it
    cannot feed the sentiment task at all.
  - raw_review_All_Beauty (raw/review_categories/All_Beauty.jsonl) has the
    full review text plus user_id, parent_asin, asin, rating, timestamp —
    everything this pipeline needs.
  This script therefore uses raw_review_All_Beauty, per the task's own named
  fallback rather than a silent substitution.

  Separately: `datasets.load_dataset()` cannot load *either* config on this
  installed version (datasets 5.0.1) — McAuley-Lab/Amazon-Reviews-2023 ships
  a custom loading script, and `datasets>=4.0` removed support for dataset
  scripts entirely ("Dataset scripts are no longer supported"). This script
  pulls the same underlying file via huggingface_hub instead, which is how
  the file already on disk (data/raw/All_Beauty.jsonl) was obtained.

Author : Opinion Evolution Tracking Project
Date   : 2026
=============================================================================
"""

import os
import sys
import json
import logging
import argparse
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src"))

from preprocessing import PreprocessingPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(WORKSPACE_ROOT, "data", "raw")
PREPROCESSED_DIR = os.path.join(WORKSPACE_ROOT, "data", "preprocessed")

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"
RAW_REVIEWS_FILE = "raw/review_categories/All_Beauty.jsonl"
LOCAL_JSONL = os.path.join(RAW_DIR, "All_Beauty.jsonl")

MIN_SEQUENCE_LENGTH = 3
# Product-level grouping (user_id + parent_asin) is expected to yield very
# few multi-review groups -- a shopper rarely reviews the exact same product
# 3+ times. This is the count threshold below which the script falls back to
# product-category grouping (user_id only, since the pull is already scoped
# to a single category, All_Beauty).
MIN_VIABLE_SEQUENCES = 200


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA ACQUISITION
# ══════════════════════════════════════════════════════════════════════════════

class AmazonReviewDownloader:
    """Pulls the raw All_Beauty review category from HuggingFace."""

    @staticmethod
    def ensure_local_file(force: bool = False) -> str:
        """
        Ensure the raw JSONL is present locally and complete, downloading it
        via huggingface_hub if missing. `datasets.load_dataset()` is not used
        because this repository ships a custom loading script, which
        datasets>=4.0 refuses to execute ("Dataset scripts are no longer
        supported").

        Returns
        -------
        str
            Path to the local JSONL file.
        """
        os.makedirs(RAW_DIR, exist_ok=True)

        if not force and os.path.exists(LOCAL_JSONL):
            local_size = os.path.getsize(LOCAL_JSONL)
            remote_size = AmazonReviewDownloader._remote_size()
            if remote_size is not None and local_size == remote_size:
                logger.info(
                    f"  {os.path.basename(LOCAL_JSONL)} already present and "
                    f"matches remote size ({local_size:,} bytes). Skipping download."
                )
                return LOCAL_JSONL
            if remote_size is None:
                logger.info(
                    f"  {os.path.basename(LOCAL_JSONL)} already present "
                    f"({local_size:,} bytes); could not verify remote size, "
                    f"reusing local copy."
                )
                return LOCAL_JSONL

        logger.info(f"Downloading {RAW_REVIEWS_FILE} from {REPO_ID} ...")
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=RAW_REVIEWS_FILE,
        )
        # hf_hub_download caches under ~/.cache/huggingface; mirror it into
        # data/raw/ so this repo's data/ directory stays the single place
        # AmazonSequenceDataset and friends look for local data.
        import shutil
        shutil.copyfile(downloaded_path, LOCAL_JSONL)
        logger.info(f"  Saved to {LOCAL_JSONL}")
        return LOCAL_JSONL

    @staticmethod
    def _remote_size() -> Optional[int]:
        try:
            from huggingface_hub import HfApi
            info = HfApi().dataset_info(REPO_ID, files_metadata=True)
            for sibling in info.siblings:
                if sibling.rfilename == RAW_REVIEWS_FILE:
                    return sibling.size
        except Exception as e:
            logger.warning(f"  Could not check remote file size: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. SEQUENCE GROUPING
# ══════════════════════════════════════════════════════════════════════════════

class ReviewLoader:
    """Parses the raw JSONL into per-review records."""

    @staticmethod
    def load(jsonl_path: str, preprocess_text: bool = True) -> List[Dict]:
        """
        Parse every line of the raw JSONL into a review dict.

        Parameters
        ----------
        jsonl_path : str
            Path to the raw All_Beauty.jsonl file.
        preprocess_text : bool
            Whether to run the shared NoiseRemover/TextNormalizer cleaning
            pipeline over each review's text.

        Returns
        -------
        List[Dict]
            One dict per review with the fields needed for grouping.
        """
        pipeline = PreprocessingPipeline(
            filter_mode="keep", include_script_features=False, verbose=False,
        )

        reviews = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                user_id = record.get("user_id")
                parent_asin = record.get("parent_asin")
                if not user_id or not parent_asin or "text" not in record:
                    continue
                if "timestamp" not in record:
                    continue

                text_raw = record.get("text", "") or ""
                reviews.append({
                    "user_id": user_id,
                    "parent_asin": parent_asin,
                    "asin": str(record.get("asin", parent_asin)),
                    "text": pipeline.preprocess_text(text_raw) if preprocess_text else text_raw,
                    "text_original": text_raw,
                    "rating": float(record.get("rating", 3.0)),
                    "timestamp": int(record["timestamp"]),
                })
        return reviews


class SequenceGrouper:
    """Groups reviews into chronological sequences under different strategies."""

    @staticmethod
    def group(
        reviews: List[Dict],
        key_fields: List[str],
        min_length: int = MIN_SEQUENCE_LENGTH,
    ) -> List[Dict]:
        """
        Group reviews by `key_fields`, sort each group chronologically, and
        keep only groups with at least `min_length` reviews.

        Parameters
        ----------
        reviews : List[Dict]
            Flat review records from ReviewLoader.load().
        key_fields : List[str]
            Fields to group by, e.g. ["user_id", "parent_asin"] for
            exact-product grouping or ["user_id"] for product-category
            grouping.
        min_length : int
            Minimum reviews per group to keep it as a sequence.

        Returns
        -------
        List[Dict]
            One dict per surviving sequence: group_key, reviews (sorted),
            sequence_length.
        """
        groups = defaultdict(list)
        for review in reviews:
            key = tuple(review[field] for field in key_fields)
            groups[key].append(review)

        sequences = []
        for key, group_reviews in groups.items():
            if len(group_reviews) < min_length:
                continue
            sorted_reviews = sorted(group_reviews, key=lambda r: r["timestamp"])
            for r in sorted_reviews:
                r["date_str"] = datetime.fromtimestamp(
                    r["timestamp"] / 1000
                ).strftime("%Y-%m-%d %H:%M:%S")
            sequences.append({
                "group_key": key,
                "key_fields": key_fields,
                "sequence_length": len(sorted_reviews),
                "reviews": sorted_reviews,
            })
        return sequences


# ══════════════════════════════════════════════════════════════════════════════
# 3. OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def save_sequences(sequences: List[Dict], name: str) -> str:
    """Save sequences to data/preprocessed/{name}.json."""
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PREPROCESSED_DIR, f"{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sequences, f, indent=2, ensure_ascii=False)
    logger.info(f"  Saved {len(sequences):,} sequences to {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build(force_download: bool = False, save: bool = True) -> Dict:
    logger.info("=" * 60)
    logger.info("Domain 1: Amazon Beauty — Product-Aware Sequence Builder")
    logger.info("=" * 60)

    # ── Stage 1: Data acquisition ──
    jsonl_path = AmazonReviewDownloader.ensure_local_file(force=force_download)

    # ── Stage 2: Parse raw reviews ──
    logger.info("Parsing raw reviews...")
    reviews = ReviewLoader.load(jsonl_path)
    total_reviews = len(reviews)
    logger.info(f"  Total usable raw reviews: {total_reviews:,}")

    # ── Stage 3: Exact-product grouping (user_id + parent_asin) ──
    logger.info("Grouping strategy 1/2: exact-product (user_id + parent_asin)")
    product_groups_all = defaultdict(list)
    for r in reviews:
        product_groups_all[(r["user_id"], r["parent_asin"])].append(r)
    product_group_count = len(product_groups_all)
    product_sequences = SequenceGrouper.group(
        reviews, key_fields=["user_id", "parent_asin"],
    )
    logger.info(f"  Groups formed (user_id + parent_asin): {product_group_count:,}")
    logger.info(
        f"  Groups surviving >= {MIN_SEQUENCE_LENGTH} filter: "
        f"{len(product_sequences):,}"
    )

    # ── Stage 4: Decide strategy ──
    used_fallback = len(product_sequences) < MIN_VIABLE_SEQUENCES
    if used_fallback:
        logger.info(
            f"  Exact-product grouping yielded only {len(product_sequences)} "
            f"sequences (< {MIN_VIABLE_SEQUENCES} viability threshold) -- "
            f"falling back to product-category grouping (user_id only)."
        )
        logger.info("Grouping strategy 2/2: product-category (user_id only)")
        category_groups_all = defaultdict(list)
        for r in reviews:
            category_groups_all[(r["user_id"],)].append(r)
        category_group_count = len(category_groups_all)
        final_sequences = SequenceGrouper.group(reviews, key_fields=["user_id"])
        logger.info(f"  Groups formed (user_id only): {category_group_count:,}")
        logger.info(
            f"  Groups surviving >= {MIN_SEQUENCE_LENGTH} filter: "
            f"{len(final_sequences):,}"
        )
        final_strategy = "product_category (user_id)"
        final_group_count = category_group_count
    else:
        final_sequences = product_sequences
        final_strategy = "exact_product (user_id + parent_asin)"
        final_group_count = product_group_count

    # ── Stage 5: Save + report ──
    if save:
        save_sequences(product_sequences, "amazon_beauty_sequences_by_product")
        if used_fallback:
            save_sequences(final_sequences, "amazon_beauty_sequences_by_category")

    lengths = [s["sequence_length"] for s in final_sequences]
    report = {
        "total_raw_reviews": total_reviews,
        "product_groups_formed": product_group_count,
        "product_sequences_surviving_filter": len(product_sequences),
        "used_fallback": used_fallback,
        "final_strategy": final_strategy,
        "final_groups_formed": final_group_count,
        "final_sequences_surviving_filter": len(final_sequences),
        "avg_sequence_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "max_sequence_length": max(lengths) if lengths else 0,
    }

    logger.info("\n" + "=" * 60)
    logger.info("SEQUENCE BUILD SUMMARY")
    logger.info("=" * 60)
    for k, v in report.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    return {"sequences": final_sequences, "report": report}


def main():
    parser = argparse.ArgumentParser(
        description="Domain 1 — Amazon Beauty product-aware sequence builder"
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-download the raw JSONL even if a local copy already matches.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Build and report sequence counts without writing output files.",
    )
    args = parser.parse_args()
    build(force_download=args.force_download, save=not args.no_save)


if __name__ == "__main__":
    main()
