"""
=============================================================================
Module 2 — Word Sense Disambiguation (WSD) for Code-Mixed Text
=============================================================================
Detects aspect-related words in code-mixed text and disambiguates their
sense using a context-window overlap approach with a multilingual lexicon.

Input:  Cleaned text (from preprocessing.py)
Output: List of (word, chosen_aspect, confidence) per message

Design:
  - Uses ontology/wsd_lexicon_full.json as the sense inventory
  - Context-window overlap for disambiguation (not a separate ML model)
  - Standalone and independently testable
  - Does NOT modify ontology.py or the existing sentiment pipeline

Reference: Implementation Plan Part A, Steps A4-A6
Author: Opinion Evolution Tracking Project
Date: 2026
=============================================================================
"""

import json
import os
import re
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Lexicon Loader
# ──────────────────────────────────────────────────────────────────────────────

class WSDLexicon:
    """
    Loads and manages the multilingual WSD lexicon.
    Maps aspect categories to all known surface forms across languages.
    """

    def __init__(self, lexicon_path: Optional[str] = None):
        if lexicon_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            lexicon_path = os.path.join(base, "ontology", "wsd_lexicon_full.json")

        if not os.path.exists(lexicon_path):
            alt_path = lexicon_path.replace("_full.json", ".json")
            if os.path.exists(alt_path):
                lexicon_path = alt_path
            else:
                raise FileNotFoundError(
                    f"WSD lexicon not found at {lexicon_path} or {alt_path}"
                )

        with open(lexicon_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.categories = data.get("categories", {})
        self.meta = data.get("_meta", {})

        # Build reverse index: surface_form -> list of (aspect_id, language)
        self._surface_to_aspect: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._aspect_keywords: Dict[str, set] = {}

        for aspect_id, lang_data in self.categories.items():
            all_forms = set()
            for lang_key, forms in lang_data.items():
                if lang_key.startswith("_"):
                    continue
                if isinstance(forms, list):
                    for form in forms:
                        form_lower = form.lower().strip()
                        if form_lower:
                            self._surface_to_aspect[form_lower].append(
                                (aspect_id, lang_key)
                            )
                            all_forms.add(form_lower)
            self._aspect_keywords[aspect_id] = all_forms

        logger.info(
            f"WSD Lexicon loaded: {len(self.categories)} aspects, "
            f"{len(self._surface_to_aspect)} surface forms"
        )

    def lookup(self, word: str) -> List[Tuple[str, str]]:
        """Look up a word and return list of (aspect_id, language) matches."""
        return self._surface_to_aspect.get(word.lower().strip(), [])

    def is_ambiguous(self, word: str) -> bool:
        """A word is ambiguous if it maps to more than one aspect category."""
        matches = self.lookup(word)
        unique_aspects = set(asp for asp, _ in matches)
        return len(unique_aspects) > 1

    def get_aspect_keywords(self, aspect_id: str) -> set:
        """Get all surface forms associated with an aspect."""
        return self._aspect_keywords.get(aspect_id, set())

    @property
    def aspect_ids(self) -> List[str]:
        return list(self.categories.keys())


# ──────────────────────────────────────────────────────────────────────────────
# 2. Context-Window Disambiguator
# ──────────────────────────────────────────────────────────────────────────────

class ContextWindowDisambiguator:
    """
    Disambiguates word senses using context-window overlap.
    For each ambiguous word, looks at surrounding words in a window and
    counts how many belong to each candidate aspect's keyword set.
    The aspect with the highest overlap wins.
    """

    def __init__(self, lexicon: WSDLexicon, window_size: int = 5):
        self.lexicon = lexicon
        self.window_size = window_size

    def disambiguate_word(
        self,
        word: str,
        context_words: List[str],
        word_position: int,
    ) -> Tuple[str, float]:
        """
        Disambiguate a single word given its context.

        Returns
        -------
        Tuple[str, float]
            (chosen_aspect_id, confidence_score)
        """
        matches = self.lexicon.lookup(word)
        if not matches:
            return ("unknown", 0.0)

        candidate_aspects = set(asp for asp, _ in matches)

        if len(candidate_aspects) == 1:
            return (list(candidate_aspects)[0], 1.0)

        # Get context window
        start = max(0, word_position - self.window_size)
        end = min(len(context_words), word_position + self.window_size + 1)
        window = [
            w.lower()
            for i, w in enumerate(context_words)
            if start <= i < end and i != word_position
        ]

        # Score each candidate aspect by context overlap
        scores: Dict[str, int] = {}
        for aspect_id in candidate_aspects:
            aspect_keywords = self.lexicon.get_aspect_keywords(aspect_id)
            overlap = sum(1 for w in window if w in aspect_keywords)
            scores[aspect_id] = overlap

        total = sum(scores.values())
        if total == 0:
            best = list(candidate_aspects)[0]
            return (best, 0.5)

        best_aspect = max(scores, key=scores.get)
        confidence = scores[best_aspect] / total if total > 0 else 0.0
        return (best_aspect, round(confidence, 4))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Main WSD Module
# ──────────────────────────────────────────────────────────────────────────────

class WordSenseDisambiguator:
    """
    Main WSD module. Given cleaned text, detects aspect-related words
    and disambiguates their sense using context overlap.

    Usage:
        wsd = WordSenseDisambiguator()
        results = wsd.process("Padam vera level bro superb acting bgm romba nalla")
        # Returns: [('acting', 'hero_character', 0.8), ('bgm', 'music_bgm', 1.0)]
    """

    def __init__(
        self,
        lexicon_path: Optional[str] = None,
        window_size: int = 5,
        min_confidence: float = 0.0,
    ):
        self.lexicon = WSDLexicon(lexicon_path)
        self.disambiguator = ContextWindowDisambiguator(
            self.lexicon, window_size
        )
        self.min_confidence = min_confidence

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer for WSD."""
        text = text.lower().strip()
        tokens = re.findall(
            r"[a-zA-Z\u0B80-\u0BFF\u0D00-\u0D7F\u0C80-\u0CFF]+", text
        )
        return tokens

    def process(self, text: str) -> List[Tuple[str, str, float]]:
        """
        Process a single text and return aspect annotations.

        Returns
        -------
        List[Tuple[str, str, float]]
            List of (word, aspect_id, confidence).
        """
        words = self.tokenize(text)
        results = []

        for i, word in enumerate(words):
            matches = self.lexicon.lookup(word)
            if not matches:
                continue
            aspect_id, confidence = self.disambiguator.disambiguate_word(
                word, words, i
            )
            if confidence >= self.min_confidence:
                results.append((word, aspect_id, confidence))

        return results

    def process_batch(
        self, texts: List[str]
    ) -> List[List[Tuple[str, str, float]]]:
        """Process a batch of texts."""
        return [self.process(text) for text in texts]

    def get_aspect_distribution(
        self, texts: List[str]
    ) -> Dict[str, int]:
        """Get frequency of each aspect across a corpus."""
        counts: Dict[str, int] = defaultdict(int)
        for text in texts:
            for _, aspect_id, _ in self.process(text):
                counts[aspect_id] += 1
        return dict(counts)

    def get_coverage_stats(self, texts: List[str]) -> Dict[str, float]:
        """
        Compute WSD coverage statistics.

        Returns dict with total_words, aspect_words, ambiguous_words,
        resolved_words, coverage_pct, resolution_pct.
        """
        total_words = 0
        aspect_words = 0
        ambiguous_words = 0
        resolved_words = 0

        for text in texts:
            words = self.tokenize(text)
            total_words += len(words)
            for i, word in enumerate(words):
                matches = self.lexicon.lookup(word)
                if matches:
                    aspect_words += 1
                    if self.lexicon.is_ambiguous(word):
                        ambiguous_words += 1
                        _, conf = self.disambiguator.disambiguate_word(
                            word, words, i
                        )
                        if conf > 0.5:
                            resolved_words += 1

        return {
            "total_words": total_words,
            "aspect_words": aspect_words,
            "ambiguous_words": ambiguous_words,
            "resolved_words": resolved_words,
            "coverage_pct": round(
                (aspect_words / total_words * 100) if total_words > 0 else 0, 2
            ),
            "resolution_pct": round(
                (resolved_words / ambiguous_words * 100)
                if ambiguous_words > 0
                else 100.0,
                2,
            ),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Most-Common-Sense Baseline
# ──────────────────────────────────────────────────────────────────────────────

class MostCommonSenseBaseline:
    """
    Always-most-common-sense heuristic baseline.
    For each word, always picks the aspect that appeared most frequently.
    """

    def __init__(self, lexicon: WSDLexicon):
        self.lexicon = lexicon
        self.sense_frequencies: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def fit(self, texts: List[str]) -> None:
        """Learn sense frequencies from training data."""
        wsd = WordSenseDisambiguator()
        for text in texts:
            for word, aspect_id, _ in wsd.process(text):
                self.sense_frequencies[word.lower()][aspect_id] += 1

    def predict(self, word: str) -> Tuple[str, float]:
        """Predict the most common sense for a word."""
        word_lower = word.lower()
        if word_lower in self.sense_frequencies:
            freqs = self.sense_frequencies[word_lower]
            best = max(freqs, key=freqs.get)
            total = sum(freqs.values())
            return (best, freqs[best] / total)
        matches = self.lexicon.lookup(word)
        if matches:
            return (matches[0][0], 0.5)
        return ("unknown", 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Standalone test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("Module 2 — Word Sense Disambiguation (WSD) Test")
    print("=" * 70)

    wsd = WordSenseDisambiguator()

    test_texts = [
        "Padam vera level bro superb acting bgm romba nalla",
        "Trailer romba mass fans ku pidikkum hero entry vera level",
        "Story weak but songs super hit movie",
        "Villain acting super dialogue punch romba nalla",
        "Flop movie box office collection romba low",
        "Climax twist nalla irukku screenplay worth watching",
    ]

    print(f"\nLexicon: {len(wsd.lexicon.aspect_ids)} aspects loaded")
    print(f"Surface forms: {len(wsd.lexicon._surface_to_aspect)} entries\n")

    for text in test_texts:
        print(f"Text: {text}")
        results = wsd.process(text)
        if results:
            for word, aspect, conf in results:
                print(f"  -> {word:15s} -> {aspect:25s} (conf: {conf:.2f})")
        else:
            print("  -> No aspect words detected")
        print()

    print("-" * 70)
    print("Coverage Statistics:")
    stats = wsd.get_coverage_stats(test_texts)
    for k, v in stats.items():
        print(f"  {k:20s}: {v}")
