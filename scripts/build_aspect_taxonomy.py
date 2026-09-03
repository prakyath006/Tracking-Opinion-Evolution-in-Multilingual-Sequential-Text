"""
=============================================================================
Module 2 (WSD) — Step A1: Aspect Taxonomy, derived from real corpus frequency
=============================================================================
Scans the actual preprocessed Dravidian (Tamil/Malayalam/Kannada) sentiment
text for recurring aspect-keyword groups and keeps only the categories that
genuinely occur (>=1% of rows), per the task constraint not to invent
categories the data doesn't support.

This domain is YouTube-comment movie/celebrity sentiment, not a product-
review domain -- so candidate categories are movie-review aspects (story,
acting, music, trailer, ...), not electronics aspects (battery, camera, ...).

Usage:
    python scripts/build_aspect_taxonomy.py
    -> ontology/aspect_taxonomy.json
=============================================================================
"""

import os
import re
import json
import logging

import pandas as pd

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPROCESSED_DIR = os.path.join(WORKSPACE_ROOT, "data", "preprocessed")
OUTPUT_PATH = os.path.join(WORKSPACE_ROOT, "ontology", "aspect_taxonomy.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SENTIMENT_FILES = [
    "tamil_sentiment_train_preprocessed.csv",
    "mal_sentiment_train_preprocessed.csv",
    "kannada_sentiment_train_preprocessed.csv",
]

# Candidate categories with seed keyword variants (English loanwords / common
# transliterations, since the corpus is mostly Latin-script code-mixed).
CANDIDATES = {
    "fan_stardom": {
        "name": "Fan / Stardom",
        "definition": "Commentary about an actor's or star's fan following, fan base, mass appeal, or hype -- not about the film's content itself.",
        "keywords": ["fan", "fans", "mass", "fanbase", "hype"],
    },
    "trailer_teaser": {
        "name": "Trailer / Teaser",
        "definition": "Reactions to a film's promotional trailer, teaser, or promo video, as distinct from reactions to the finished film.",
        "keywords": ["trailer", "teaser", "promo"],
    },
    "music_bgm": {
        "name": "Music / BGM",
        "definition": "Opinions about a film's songs, background score (BGM), or music in general.",
        "keywords": ["music", "bgm", "song", "songs"],
    },
    "box_office_collection": {
        "name": "Box Office / Collection",
        "definition": "Commentary on a film's commercial performance -- collections, hit/flop status, box-office numbers.",
        "keywords": ["box office", "boxoffice", "collection", "hit", "flop", "crore"],
    },
    "dialogue": {
        "name": "Dialogue",
        "definition": "Opinions about specific dialogue lines, dialogue delivery, or 'punch' dialogues.",
        "keywords": ["dialogue", "dialog", "punch dialogue"],
    },
    "acting_performance": {
        "name": "Acting / Performance",
        "definition": "Opinions about an actor's acting or performance quality (nadippu / abhinaya).",
        "keywords": ["acting", "nadippu", "abhinaya", "nadipu"],
    },
    "hero_character": {
        "name": "Hero / Character",
        "definition": "Opinions about a lead hero, heroine, villain, or other named character/casting choice.",
        "keywords": ["hero", "heroine", "villain", "villan"],
    },
    "story_screenplay": {
        "name": "Story / Screenplay",
        "definition": "Opinions about the film's story, plot, or screenplay/script.",
        "keywords": ["story", "kathai", "katha", "kadhai", "script"],
    },
    # Below-threshold candidates, kept here (commented out of the output) so
    # the cutoff is reproducible/auditable rather than silently dropped:
    #   direction (0.58%), visuals/cinematography (0.44%), comedy (0.38%),
    #   review/rating (0.17%)
    "direction": {"name": "Direction", "definition": "Opinions about the film's director/direction.", "keywords": ["director", "direction", "directed"]},
    "visuals_cinematography": {"name": "Visuals / Cinematography", "definition": "Opinions about visuals, cinematography, or VFX.", "keywords": ["visual", "visuals", "cinematography", "vfx", "graphics"]},
    "comedy": {"name": "Comedy", "definition": "Opinions about comedy content or comedian performances.", "keywords": ["comedy", "comedian", "funny"]},
    "review_rating": {"name": "Review / Rating", "definition": "Meta-commentary about reviews or star ratings themselves.", "keywords": ["review", "rating", "stars"]},
}

MIN_FREQUENCY_PCT = 1.0


def compute_category_frequencies() -> dict:
    """% of corpus rows whose text contains at least one keyword variant, per candidate category."""
    texts = []
    for fname in SENTIMENT_FILES:
        path = os.path.join(PREPROCESSED_DIR, fname)
        if not os.path.exists(path):
            logger.warning(f"Missing (data/ is gitignored): {path}")
            continue
        df = pd.read_csv(path)
        texts.append(df["text"].astype(str).str.lower())

    if not texts:
        raise FileNotFoundError(
            "No preprocessed Dravidian sentiment CSVs found under data/preprocessed/. "
            "data/ is gitignored -- run the preprocessing pipeline first."
        )

    all_text = pd.concat(texts, ignore_index=True)
    total = len(all_text)

    freqs = {}
    for cat_id, cat in CANDIDATES.items():
        matched = pd.Series(False, index=all_text.index)
        for kw in cat["keywords"]:
            matched |= all_text.str.contains(re.escape(kw), regex=True, na=False)
        freqs[cat_id] = 100.0 * matched.sum() / total
    return freqs, total


def build_taxonomy() -> dict:
    freqs, total = compute_category_frequencies()

    categories = []
    for cat_id, cat in CANDIDATES.items():
        pct = freqs[cat_id]
        if pct < MIN_FREQUENCY_PCT:
            logger.info(f"Dropping '{cat_id}' — below {MIN_FREQUENCY_PCT}% threshold ({pct:.2f}%)")
            continue
        categories.append({
            "id": cat_id,
            "name": cat["name"],
            "definition": cat["definition"],
            "seed_keywords": cat["keywords"],
            "corpus_frequency_pct": round(pct, 2),
        })

    categories.sort(key=lambda c: c["corpus_frequency_pct"], reverse=True)

    return {
        "_meta": {
            "purpose": "Module 2 (WSD) aspect/sense taxonomy for the Dravidian (Tamil/Malayalam/Kannada) code-mixed domain.",
            "derivation": (
                f"Frequency analysis of candidate English-loanword keyword groups over the "
                f"'text' column of {', '.join(SENTIMENT_FILES)} ({total:,} rows total). This "
                f"domain is YouTube-comment movie/celebrity sentiment, not a product-review "
                f"domain -- categories below were selected because they actually recur in this "
                f"data, not because they are typical WSD/aspect examples elsewhere."
            ),
            "method": (
                f"For each candidate category, counted rows whose lowercased text contains any "
                f"of its keyword variants (regex substring match). Categories below "
                f"{MIN_FREQUENCY_PCT}% of rows were dropped as not genuinely recurring."
            ),
            "not_used": (
                "The task prompt's example categories (Battery, Camera, Display, Price, "
                "Performance, Build Quality) are electronics/product-review aspects. They do "
                "not occur in this project's Dravidian data (movie/entertainment YouTube "
                "comments) and were not used -- this taxonomy is derived from the actual "
                "corpus instead."
            ),
        },
        "categories": categories,
    }


def main():
    taxonomy = build_taxonomy()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(taxonomy['categories'])} categories -> {OUTPUT_PATH}")
    for c in taxonomy["categories"]:
        logger.info(f"  {c['id']:<24} {c['corpus_frequency_pct']:>6.2f}%  {c['name']}")


if __name__ == "__main__":
    main()
