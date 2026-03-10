"""
=============================================================================
DravidianCodeMix-2020 — Preprocessing Pipeline
=============================================================================
A comprehensive preprocessing pipeline for the DravidianCodeMix-2020 dataset
supporting Tamil, Malayalam, and Kannada code-mixed text for:
  • Sentiment Analysis
  • Offensive Language Detection

Pipeline Stages:
  1. Data Loading & Unification
  2. Language Filtering (not-Tamil / not-Malayalam / not-Kannada)
  3. Noise Removal (URLs, mentions, hashtags, emojis, special chars)
  4. Text Normalization (lowercasing, whitespace, punctuation)
  5. Code-Mix Handling (script detection, transliteration-aware tokenization)
  6. Sentence Segmentation
  7. Label Encoding
  8. Statistics & Summary Report

Author : Preprocessing Pipeline for DravidianCodeMix-2020
Date   : 2026
=============================================================================
"""

import os
import re
import csv
import sys
import unicodedata
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "preprocessed")

# Unicode ranges for Dravidian scripts
TAMIL_RANGE      = (0x0B80, 0x0BFF)
MALAYALAM_RANGE  = (0x0D00, 0x0D7F)
KANNADA_RANGE    = (0x0C80, 0x0CFF)
DEVANAGARI_RANGE = (0x0900, 0x097F)
LATIN_RANGE      = (0x0000, 0x007F)

# Emoji pattern (covers most emoji blocks)
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"             # zero width joiner
    "]+",
    flags=re.UNICODE,
)

# Dataset file mapping
DATASET_FILES = {
    # Tamil Sentiment
    "tamil_sentiment": {
        "train": "tamil_sentiment_full_train.csv",
        "dev":   "tamil_sentiment_full_dev.csv",
        "test":  "tamil_sentiment_full_test.csv",
        "full":  "tamil_sentiment_full.csv",
    },
    # Tamil Offensive
    "tamil_offensive": {
        "train": "tamil_offensive_full_train.csv",
        "dev":   "tamil_offensive_full_dev.csv",
        "test":  "tamil_offensive_full_test.csv",
        "full":  "tamil_offensive_full.csv",
    },
    # Malayalam Sentiment
    "mal_sentiment": {
        "train": "mal_full_sentiment_train.csv",
        "dev":   "mal_full_sentiment_dev.csv",
        "test":  "mal_full_sentiment_test.csv",
        "full":  "mal_full_sentiment.tsv",
    },
    # Malayalam Offensive
    "mal_offensive": {
        "train": "mal_full_offensive_train.csv",
        "dev":   "mal_full_offensive_dev.csv",
        "test":  "mal_full_offensive_test.csv",
        "full":  "mal_full_offensive.csv",
    },
    # Kannada Sentiment
    "kannada_sentiment": {
        "train": "kannada_sentiment_train.csv",
        "dev":   "kannada_sentiment_dev.csv",
        "test":  "kannada_sentiment_test.csv",
        "full":  "kannada_sentiment.csv",
    },
    # Kannada Offensive
    "kannada_offensive": {
        "train": "kannada_offensive_train.csv",
        "dev":   "kannada_offensive_dev.csv",
        "test":  "kannada_offensive_test.csv",
        "full":  "kannada_offensive.csv",
    },
}

# Label mappings for encoding
SENTIMENT_LABELS = {
    "Positive":       0,
    "Negative":       1,
    "Mixed_feelings": 2,
    "unknown_state":  3,
    "not-Tamil":      4,
    "not-Malayalam":   4,
    "not-malayalam":  4,
    "not-Kannada":    4,
}

OFFENSIVE_LABELS = {
    "Not_offensive":                       0,
    "Offensive_Targeted_Insult_Individual": 1,
    "Offensive_Targeted_Insult_Group":      2,
    "Offensive_Targeted_Insult_Other":      3,
    "Offensive_Untargetede":               4,   # typo in original dataset
    "Offensive_Untargeted":                4,
    "not-Tamil":                           5,
    "not-Malayalam":                        5,
    "not-malayalam":                        5,
    "not-Kannada":                         5,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & UNIFICATION
# ══════════════════════════════════════════════════════════════════════════════

class DataLoader:
    """
    Flexible data loader that handles the inconsistent delimiters
    and column structures across the DravidianCodeMix-2020 dataset.
    """

    @staticmethod
    def detect_delimiter(filepath: str) -> str:
        """Auto-detect delimiter by reading the first few lines."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(4096)

        # Count potential delimiters
        tab_count   = sample.count("\t")
        semi_count  = sample.count(";")
        comma_count = sample.count(",")

        counts = {"tab": tab_count, "semi": semi_count, "comma": comma_count}
        winner = max(counts, key=counts.get)

        delim_map = {"tab": "\t", "semi": ";", "comma": ","}
        return delim_map[winner]

    @staticmethod
    def _detect_format(filepath: str) -> Dict:
        """
        Detect the file format by inspecting the first few lines.
        Returns dict with 'delimiter', 'text_col', 'label_col' info.
        """
        delimiter = DataLoader.detect_delimiter(filepath)

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = [f.readline() for _ in range(10)]

        # Determine column order
        # Some files have: text<delim>label
        # The TSV file has: label<delim>text
        # Kannada files use semicolons with trailing semicolons: text;label;

        fmt = {
            "delimiter": delimiter,
            "text_first": True,  # default: text is first column
        }

        # Check if labels appear in the first column (like the .tsv file)
        known_labels = set(SENTIMENT_LABELS.keys()) | set(OFFENSIVE_LABELS.keys())
        first_col_label_count = 0
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().rstrip(delimiter).split(delimiter)
            if parts and parts[0].strip() in known_labels:
                first_col_label_count += 1

        if first_col_label_count >= 3:
            fmt["text_first"] = False

        return fmt

    @staticmethod
    def load_file(filepath: str, task: str = "sentiment") -> pd.DataFrame:
        """
        Load a single dataset file into a DataFrame with columns: [text, label].

        Parameters
        ----------
        filepath : str
            Path to the CSV/TSV file.
        task : str
            Either 'sentiment' or 'offensive' (used for label validation).

        Returns
        -------
        pd.DataFrame with columns ['text', 'label']
        """
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame(columns=["text", "label"])

        fmt = DataLoader._detect_format(filepath)
        delimiter = fmt["delimiter"]
        text_first = fmt["text_first"]

        rows = []
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # Strip trailing delimiters (Kannada files have trailing ;)
                line = line.rstrip(delimiter)

                parts = line.split(delimiter)

                if len(parts) < 2:
                    # Single column — skip or treat as text-only
                    continue

                if text_first:
                    text = delimiter.join(parts[:-1]).strip()
                    label = parts[-1].strip()
                else:
                    label = parts[0].strip()
                    text = delimiter.join(parts[1:]).strip()

                # Validate label
                known = (
                    set(SENTIMENT_LABELS.keys())
                    if task == "sentiment"
                    else set(OFFENSIVE_LABELS.keys())
                )
                if label not in known:
                    # If text and label are swapped, try the other order
                    text_alt = parts[-1].strip()
                    label_alt = delimiter.join(parts[:-1]).strip() if text_first else parts[0].strip()
                    if text_alt in known:
                        text, label = label, text_alt
                    elif label_alt in known:
                        text = delimiter.join(parts[1:]).strip() if not text_first else delimiter.join(parts[:-1]).strip()
                        label = label_alt

                rows.append({"text": text, "label": label})

        df = pd.DataFrame(rows)
        logger.info(
            f"Loaded {len(df):,} rows from {os.path.basename(filepath)} "
            f"(delim={repr(delimiter)}, text_first={text_first})"
        )
        return df

    @staticmethod
    def load_dataset(
        dataset_key: str,
        split: str = "train",
        task: str = None,
    ) -> pd.DataFrame:
        """
        Load a specific dataset split.

        Parameters
        ----------
        dataset_key : str
            Key from DATASET_FILES, e.g. 'tamil_sentiment'.
        split : str
            One of 'train', 'dev', 'test', 'full'.
        task : str or None
            'sentiment' or 'offensive'. Auto-detected if None.

        Returns
        -------
        pd.DataFrame
        """
        if dataset_key not in DATASET_FILES:
            raise ValueError(
                f"Unknown dataset: {dataset_key}. "
                f"Available: {list(DATASET_FILES.keys())}"
            )

        if split not in DATASET_FILES[dataset_key]:
            raise ValueError(
                f"Unknown split: {split}. "
                f"Available: {list(DATASET_FILES[dataset_key].keys())}"
            )

        if task is None:
            task = "offensive" if "offensive" in dataset_key else "sentiment"

        filename = DATASET_FILES[dataset_key][split]
        filepath = os.path.join(BASE_DIR, filename)
        return DataLoader.load_file(filepath, task=task)

    @staticmethod
    def load_all(split: str = "train") -> Dict[str, pd.DataFrame]:
        """Load all datasets for a given split."""
        result = {}
        for key in DATASET_FILES:
            task = "offensive" if "offensive" in key else "sentiment"
            try:
                df = DataLoader.load_dataset(key, split=split, task=task)
                if len(df) > 0:
                    df["dataset"] = key
                    df["language"] = key.split("_")[0]  # tamil, mal, kannada
                    result[key] = df
            except Exception as e:
                logger.error(f"Error loading {key}/{split}: {e}")
        return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. LANGUAGE FILTERING
# ══════════════════════════════════════════════════════════════════════════════

class LanguageFilter:
    """
    Filter or tag samples based on the 'not-<Language>' labels.
    Provides options to:
      - Remove non-target-language samples
      - Keep them with a special label
      - Separate them into a different split
    """

    NOT_LANG_LABELS = {"not-Tamil", "not-Malayalam", "not-malayalam", "not-Kannada"}

    @staticmethod
    def is_non_target_language(label: str) -> bool:
        return label in LanguageFilter.NOT_LANG_LABELS

    @staticmethod
    def filter_non_target(
        df: pd.DataFrame,
        mode: str = "remove",
    ) -> pd.DataFrame:
        """
        Handle non-target-language samples.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a 'label' column.
        mode : str
            'remove' — drop non-target-language rows
            'keep'   — keep them as-is
            'tag'    — relabel them as 'other_language'

        Returns
        -------
        pd.DataFrame
        """
        mask = df["label"].apply(LanguageFilter.is_non_target_language)
        count = mask.sum()

        if mode == "remove":
            df_out = df[~mask].reset_index(drop=True)
            logger.info(
                f"LanguageFilter: Removed {count:,} non-target-language samples "
                f"({count / len(df) * 100:.1f}%)"
            )
        elif mode == "tag":
            df_out = df.copy()
            df_out.loc[mask, "label"] = "other_language"
            logger.info(
                f"LanguageFilter: Tagged {count:,} non-target-language samples"
            )
        else:  # keep
            df_out = df.copy()
            logger.info(
                f"LanguageFilter: Keeping {count:,} non-target-language samples"
            )
        return df_out

    @staticmethod
    def split_by_language(
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split into (target_language_df, other_language_df).
        """
        mask = df["label"].apply(LanguageFilter.is_non_target_language)
        return (
            df[~mask].reset_index(drop=True),
            df[mask].reset_index(drop=True),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. NOISE REMOVAL
# ══════════════════════════════════════════════════════════════════════════════

class NoiseRemover:
    """
    Remove social-media noise from code-mixed text.
    """

    # Compiled regex patterns for performance
    URL_PATTERN      = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    MENTION_PATTERN  = re.compile(r"@[\w]+")
    HASHTAG_PATTERN  = re.compile(r"#([\w]+)")
    HTML_PATTERN     = re.compile(r"<[^>]+>")
    NEWLINE_PATTERN  = re.compile(r"\\n|\\r|\\t")
    LITERAL_N        = re.compile(r"\bn\b")  # stray 'n' used as newline in data
    REPEATED_CHARS   = re.compile(r"(.)\1{3,}")  # 4+ repeated chars
    MULTIPLE_SPACES  = re.compile(r"\s{2,}")
    DIGITS_ONLY      = re.compile(r"^\d+$")

    @staticmethod
    def remove_urls(text: str) -> str:
        return NoiseRemover.URL_PATTERN.sub(" ", text)

    @staticmethod
    def remove_mentions(text: str) -> str:
        return NoiseRemover.MENTION_PATTERN.sub(" ", text)

    @staticmethod
    def extract_hashtags(text: str) -> str:
        """Replace #hashtag with the hashtag text (remove # symbol)."""
        return NoiseRemover.HASHTAG_PATTERN.sub(r"\1", text)

    @staticmethod
    def remove_html(text: str) -> str:
        return NoiseRemover.HTML_PATTERN.sub(" ", text)

    @staticmethod
    def remove_emojis(text: str) -> str:
        return EMOJI_PATTERN.sub(" ", text)

    @staticmethod
    def remove_stray_newlines(text: str) -> str:
        """Remove literal \\n that appear in many comments."""
        text = NoiseRemover.NEWLINE_PATTERN.sub(" ", text)
        # In the data, standalone 'n' characters are often newlines
        # Only remove if they appear as word boundaries
        return text

    @staticmethod
    def normalize_repeated_chars(text: str) -> str:
        """Reduce 4+ repeated characters to 2 (e.g. 'suuuuper' -> 'suuper')."""
        return NoiseRemover.REPEATED_CHARS.sub(r"\1\1", text)

    @staticmethod
    def remove_special_characters(text: str) -> str:
        """
        Remove special characters while preserving:
        - Dravidian script characters (Tamil, Malayalam, Kannada)
        - Latin letters and digits
        - Basic punctuation (. , ! ? ')
        - Spaces
        """
        cleaned = []
        for char in text:
            cp = ord(char)
            # Keep Dravidian scripts
            if (TAMIL_RANGE[0] <= cp <= TAMIL_RANGE[1] or
                MALAYALAM_RANGE[0] <= cp <= MALAYALAM_RANGE[1] or
                KANNADA_RANGE[0] <= cp <= KANNADA_RANGE[1] or
                DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]):
                cleaned.append(char)
            # Keep Latin alphanumeric and basic punctuation
            elif char.isalnum() and cp < 0x0300:
                cleaned.append(char)
            elif char in " .,!?'-":
                cleaned.append(char)
            else:
                cleaned.append(" ")
        return "".join(cleaned)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse multiple spaces, strip leading/trailing."""
        return NoiseRemover.MULTIPLE_SPACES.sub(" ", text).strip()

    @staticmethod
    def clean(
        text: str,
        remove_emoji: bool = True,
        remove_url: bool = True,
        remove_mention: bool = True,
        keep_hashtag_text: bool = True,
        normalize_chars: bool = True,
        remove_special: bool = True,
    ) -> str:
        """
        Apply the full noise-removal pipeline to a single text.
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        if remove_url:
            text = NoiseRemover.remove_urls(text)
        if remove_mention:
            text = NoiseRemover.remove_mentions(text)
        if keep_hashtag_text:
            text = NoiseRemover.extract_hashtags(text)
        text = NoiseRemover.remove_html(text)
        if remove_emoji:
            text = NoiseRemover.remove_emojis(text)
        text = NoiseRemover.remove_stray_newlines(text)
        if normalize_chars:
            text = NoiseRemover.normalize_repeated_chars(text)
        if remove_special:
            text = NoiseRemover.remove_special_characters(text)
        text = NoiseRemover.normalize_whitespace(text)
        return text


# ══════════════════════════════════════════════════════════════════════════════
# 4. TEXT NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

class TextNormalizer:
    """
    Normalize text with language-aware strategies.
    """

    @staticmethod
    def lowercase_latin(text: str) -> str:
        """
        Lowercase only the Latin-script portions.
        Dravidian scripts don't have case, so this is safe.
        """
        result = []
        for char in text:
            if ord(char) < 0x0300:  # Basic Latin + Latin Extended
                result.append(char.lower())
            else:
                result.append(char)
        return "".join(result)

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Apply NFC normalization for consistent Unicode representation."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def remove_extra_punctuation(text: str) -> str:
        """Remove excessive punctuation (3+ repeated punctuation marks)."""
        return re.sub(r"([.!?,])\1{2,}", r"\1", text)

    @staticmethod
    def normalize(text: str) -> str:
        """Apply all normalization steps."""
        text = TextNormalizer.normalize_unicode(text)
        text = TextNormalizer.lowercase_latin(text)
        text = TextNormalizer.remove_extra_punctuation(text)
        return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# 5. CODE-MIX HANDLING
# ══════════════════════════════════════════════════════════════════════════════

class CodeMixHandler:
    """
    Analyze and handle code-mixed text (Dravidian + English/Latin).
    """

    @staticmethod
    def detect_scripts(text: str) -> Dict[str, int]:
        """
        Count characters belonging to different scripts.
        Returns dict: script_name -> character_count.
        """
        scripts = {
            "tamil":      0,
            "malayalam":  0,
            "kannada":    0,
            "devanagari": 0,
            "latin":      0,
            "digit":      0,
            "other":      0,
        }

        for char in text:
            if char.isspace():
                continue
            cp = ord(char)
            if TAMIL_RANGE[0] <= cp <= TAMIL_RANGE[1]:
                scripts["tamil"] += 1
            elif MALAYALAM_RANGE[0] <= cp <= MALAYALAM_RANGE[1]:
                scripts["malayalam"] += 1
            elif KANNADA_RANGE[0] <= cp <= KANNADA_RANGE[1]:
                scripts["kannada"] += 1
            elif DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]:
                scripts["devanagari"] += 1
            elif char.isdigit():
                scripts["digit"] += 1
            elif char.isascii() and char.isalpha():
                scripts["latin"] += 1
            else:
                scripts["other"] += 1

        return scripts

    @staticmethod
    def get_dominant_script(text: str) -> str:
        """Return the dominant script in the text."""
        scripts = CodeMixHandler.detect_scripts(text)
        # Exclude digits and other
        script_counts = {
            k: v for k, v in scripts.items()
            if k not in ("digit", "other") and v > 0
        }
        if not script_counts:
            return "unknown"
        return max(script_counts, key=script_counts.get)

    @staticmethod
    def compute_code_mixing_index(text: str) -> float:
        """
        Compute the Code-Mixing Index (CMI).
        CMI = (N - max(w_i)) / N * 100
        where N = total word-level language tags, w_i = count of language i.

        Returns a value between 0 (monolingual) and 100 (highly mixed).
        """
        words = text.split()
        if len(words) <= 1:
            return 0.0

        lang_counts = Counter()
        for word in words:
            scripts = CodeMixHandler.detect_scripts(word)
            # Determine word language
            native = scripts["tamil"] + scripts["malayalam"] + scripts["kannada"]
            latin = scripts["latin"]

            if native > 0 and latin == 0:
                lang_counts["native"] += 1
            elif latin > 0 and native == 0:
                lang_counts["latin"] += 1
            else:
                lang_counts["mixed"] += 1  # word-internal mixing

        total = sum(lang_counts.values())
        if total == 0:
            return 0.0
        max_lang = max(lang_counts.values())
        return (total - max_lang) / total * 100

    @staticmethod
    def segment_by_script(text: str) -> List[Dict[str, str]]:
        """
        Segment text into contiguous script spans.
        Returns list of dicts: [{'script': 'tamil', 'text': '...'}, ...]
        """
        if not text:
            return []

        segments = []
        current_script = None
        current_text = []

        for char in text:
            if char.isspace():
                current_text.append(char)
                continue

            cp = ord(char)
            if TAMIL_RANGE[0] <= cp <= TAMIL_RANGE[1]:
                script = "tamil"
            elif MALAYALAM_RANGE[0] <= cp <= MALAYALAM_RANGE[1]:
                script = "malayalam"
            elif KANNADA_RANGE[0] <= cp <= KANNADA_RANGE[1]:
                script = "kannada"
            elif char.isascii() and char.isalpha():
                script = "latin"
            else:
                script = current_script or "other"

            if script != current_script and current_script is not None:
                text_str = "".join(current_text).strip()
                if text_str:
                    segments.append({"script": current_script, "text": text_str})
                current_text = [char]
            else:
                current_text.append(char)
            current_script = script

        # Don't forget the last segment
        if current_text:
            text_str = "".join(current_text).strip()
            if text_str:
                segments.append({"script": current_script, "text": text_str})

        return segments

    @staticmethod
    def add_script_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add script-related features to the DataFrame."""
        df = df.copy()

        script_data = df["text"].apply(CodeMixHandler.detect_scripts)
        df["script_tamil"]     = script_data.apply(lambda x: x.get("tamil", 0))
        df["script_malayalam"] = script_data.apply(lambda x: x.get("malayalam", 0))
        df["script_kannada"]   = script_data.apply(lambda x: x.get("kannada", 0))
        df["script_latin"]     = script_data.apply(lambda x: x.get("latin", 0))
        df["dominant_script"]  = df["text"].apply(CodeMixHandler.get_dominant_script)
        df["code_mix_index"]   = df["text"].apply(
            CodeMixHandler.compute_code_mixing_index
        )

        return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. SENTENCE SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

class SentenceSegmenter:
    """
    Segment code-mixed social media text into sentences.
    Handles the lack of standard sentence boundaries in informal text.
    """

    # Sentence boundary characters (including Dravidian sentence-enders)
    BOUNDARIES = re.compile(
        r"(?<=[.!?।॥])\s+"  # After standard punctuation + Devanagari danda
        r"|(?:\.\.\.\s+)"    # Ellipsis followed by space
    )

    @staticmethod
    def segment(text: str) -> List[str]:
        """
        Split text into sentences.
        For social media text, each comment is often a single sentence,
        but we handle multi-sentence comments too.
        """
        if not text or not text.strip():
            return []

        # Split on sentence boundaries
        sentences = SentenceSegmenter.BOUNDARIES.split(text)
        # Filter empty
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences if sentences else [text.strip()]

    @staticmethod
    def get_sentence_count(text: str) -> int:
        return len(SentenceSegmenter.segment(text))


# ══════════════════════════════════════════════════════════════════════════════
# 7. LABEL ENCODING
# ══════════════════════════════════════════════════════════════════════════════

class LabelEncoder:
    """
    Encode string labels into integers for model training.
    """

    @staticmethod
    def encode_sentiment(
        df: pd.DataFrame,
        include_non_target: bool = False,
    ) -> pd.DataFrame:
        """
        Encode sentiment labels.

        Labels:
          0 = Positive
          1 = Negative
          2 = Mixed_feelings
          3 = unknown_state
          4 = not-<Language> (if include_non_target)
        """
        df = df.copy()
        df["label_encoded"] = df["label"].map(SENTIMENT_LABELS)

        if not include_non_target:
            df = df[df["label_encoded"] != 4].reset_index(drop=True)

        unmapped = df["label_encoded"].isna().sum()
        if unmapped > 0:
            logger.warning(
                f"LabelEncoder: {unmapped} labels could not be mapped. "
                f"Unique: {df.loc[df['label_encoded'].isna(), 'label'].unique()}"
            )
        return df

    @staticmethod
    def encode_offensive(
        df: pd.DataFrame,
        include_non_target: bool = False,
    ) -> pd.DataFrame:
        """
        Encode offensive language labels.

        Labels:
          0 = Not_offensive
          1 = Offensive_Targeted_Insult_Individual
          2 = Offensive_Targeted_Insult_Group
          3 = Offensive_Targeted_Insult_Other
          4 = Offensive_Untargetede
          5 = not-<Language> (if include_non_target)
        """
        df = df.copy()
        df["label_encoded"] = df["label"].map(OFFENSIVE_LABELS)

        if not include_non_target:
            df = df[df["label_encoded"] != 5].reset_index(drop=True)

        unmapped = df["label_encoded"].isna().sum()
        if unmapped > 0:
            logger.warning(
                f"LabelEncoder: {unmapped} labels could not be mapped. "
                f"Unique: {df.loc[df['label_encoded'].isna(), 'label'].unique()}"
            )
        return df


# ══════════════════════════════════════════════════════════════════════════════
# 8. STATISTICS & REPORTING
# ══════════════════════════════════════════════════════════════════════════════

class DatasetAnalyzer:
    """
    Generate statistics and reports about the dataset.
    """

    @staticmethod
    def basic_stats(df: pd.DataFrame, name: str = "Dataset") -> Dict:
        """Compute basic dataset statistics."""
        stats = {
            "name":              name,
            "total_samples":     len(df),
            "label_distribution": dict(df["label"].value_counts()),
            "avg_text_length":   df["text"].str.len().mean(),
            "median_text_length": df["text"].str.len().median(),
            "max_text_length":   df["text"].str.len().max(),
            "min_text_length":   df["text"].str.len().min(),
            "avg_word_count":    df["text"].str.split().str.len().mean(),
            "empty_texts":       (df["text"].str.strip() == "").sum(),
        }
        return stats

    @staticmethod
    def print_stats(stats: Dict) -> None:
        """Pretty-print dataset statistics."""
        print(f"\n{'='*60}")
        print(f"  Dataset: {stats['name']}")
        print(f"{'='*60}")
        print(f"  Total samples    : {stats['total_samples']:,}")
        print(f"  Avg text length  : {stats['avg_text_length']:.1f} chars")
        print(f"  Median text len  : {stats['median_text_length']:.0f} chars")
        print(f"  Max text length  : {stats['max_text_length']:.0f} chars")
        print(f"  Min text length  : {stats['min_text_length']:.0f} chars")
        print(f"  Avg word count   : {stats['avg_word_count']:.1f} words")
        print(f"  Empty texts      : {stats['empty_texts']}")
        print(f"\n  Label Distribution:")
        for label, count in sorted(
            stats["label_distribution"].items(), key=lambda x: -x[1]
        ):
            pct = count / stats["total_samples"] * 100
            bar = "█" * int(pct / 2)
            print(f"    {label:45s} {count:6,} ({pct:5.1f}%) {bar}")
        print()

    @staticmethod
    def code_mix_report(df: pd.DataFrame, name: str = "Dataset") -> Dict:
        """Generate code-mixing analysis report."""
        if "code_mix_index" not in df.columns:
            df = CodeMixHandler.add_script_features(df)

        report = {
            "name":              name,
            "avg_cmi":           df["code_mix_index"].mean(),
            "median_cmi":        df["code_mix_index"].median(),
            "monolingual_pct":   (df["code_mix_index"] == 0).mean() * 100,
            "high_cm_pct":       (df["code_mix_index"] > 50).mean() * 100,
            "dominant_scripts":  dict(df["dominant_script"].value_counts()),
        }
        return report

    @staticmethod
    def print_code_mix_report(report: Dict) -> None:
        """Pretty-print code-mixing report."""
        print(f"\n{'─'*60}")
        print(f"  Code-Mixing Report: {report['name']}")
        print(f"{'─'*60}")
        print(f"  Avg CMI              : {report['avg_cmi']:.1f}%")
        print(f"  Median CMI           : {report['median_cmi']:.1f}%")
        print(f"  Monolingual samples  : {report['monolingual_pct']:.1f}%")
        print(f"  Highly mixed (>50%)  : {report['high_cm_pct']:.1f}%")
        print(f"\n  Dominant Script Distribution:")
        for script, count in sorted(
            report["dominant_scripts"].items(), key=lambda x: -x[1]
        ):
            print(f"    {script:20s} {count:6,}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class PreprocessingPipeline:
    """
    Main pipeline that orchestrates all preprocessing steps.
    """

    def __init__(
        self,
        filter_mode: str = "remove",
        include_script_features: bool = True,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        filter_mode : str
            How to handle non-target-language samples.
            'remove' | 'keep' | 'tag'
        include_script_features : bool
            Whether to add code-mixing features.
        verbose : bool
            Print progress and reports.
        """
        self.filter_mode = filter_mode
        self.include_script_features = include_script_features
        self.verbose = verbose

    def preprocess_text(self, text: str) -> str:
        """Apply all text-level preprocessing."""
        text = NoiseRemover.clean(text)
        text = TextNormalizer.normalize(text)
        return text

    def process_dataframe(
        self,
        df: pd.DataFrame,
        task: str = "sentiment",
        dataset_name: str = "dataset",
    ) -> pd.DataFrame:
        """
        Run the full preprocessing pipeline on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: ['text', 'label']
        task : str
            'sentiment' or 'offensive'
        dataset_name : str
            Name for logging/reporting

        Returns
        -------
        pd.DataFrame with preprocessed text and encoded labels
        """
        if self.verbose:
            logger.info(f"{'='*50}")
            logger.info(f"Processing: {dataset_name} ({len(df):,} samples)")
            logger.info(f"{'='*50}")

        # ── Step 1: Pre-stats ──
        if self.verbose:
            pre_stats = DatasetAnalyzer.basic_stats(df, f"{dataset_name} (RAW)")
            DatasetAnalyzer.print_stats(pre_stats)

        # ── Step 2: Language Filtering ──
        df = LanguageFilter.filter_non_target(df, mode=self.filter_mode)

        # ── Step 3: Text Preprocessing ──
        if self.verbose:
            logger.info("Applying noise removal & text normalization...")
        df = df.copy()
        df["text_original"] = df["text"]
        df["text"] = df["text"].apply(self.preprocess_text)

        # Remove empty texts after cleaning
        empty_mask = df["text"].str.strip() == ""
        empty_count = empty_mask.sum()
        if empty_count > 0:
            df = df[~empty_mask].reset_index(drop=True)
            if self.verbose:
                logger.info(f"Removed {empty_count} empty texts after cleaning")

        # ── Step 4: Sentence Segmentation Stats ──
        df["sentence_count"] = df["text"].apply(SentenceSegmenter.get_sentence_count)
        df["word_count"]     = df["text"].apply(lambda x: len(x.split()))
        df["char_count"]     = df["text"].apply(len)

        # ── Step 5: Code-Mix Features ──
        if self.include_script_features:
            if self.verbose:
                logger.info("Computing code-mixing features...")
            df = CodeMixHandler.add_script_features(df)

        # ── Step 6: Label Encoding ──
        if task == "sentiment":
            df = LabelEncoder.encode_sentiment(df, include_non_target=False)
        else:
            df = LabelEncoder.encode_offensive(df, include_non_target=False)

        # ── Step 7: Post-stats ──
        if self.verbose:
            post_stats = DatasetAnalyzer.basic_stats(
                df, f"{dataset_name} (PREPROCESSED)"
            )
            DatasetAnalyzer.print_stats(post_stats)

            if self.include_script_features:
                cm_report = DatasetAnalyzer.code_mix_report(df, dataset_name)
                DatasetAnalyzer.print_code_mix_report(cm_report)

        return df

    def process_all(
        self,
        split: str = "train",
        save: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Process all datasets for a given split.

        Parameters
        ----------
        split : str
            'train', 'dev', 'test', or 'full'
        save : bool
            Whether to save preprocessed files.

        Returns
        -------
        Dict[str, pd.DataFrame]
        """
        logger.info(f"\n{'#'*60}")
        logger.info(f"  DRAVIDIANCODEMIX-2020 PREPROCESSING PIPELINE")
        logger.info(f"  Split: {split.upper()}")
        logger.info(f"{'#'*60}\n")

        # Load all datasets
        datasets = DataLoader.load_all(split=split)
        results = {}

        for key, df in datasets.items():
            task = "offensive" if "offensive" in key else "sentiment"
            processed = self.process_dataframe(
                df, task=task, dataset_name=f"{key}_{split}"
            )
            results[key] = processed

            # Save
            if save:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                out_path = os.path.join(OUTPUT_DIR, f"{key}_{split}_preprocessed.csv")
                processed.to_csv(out_path, index=False, encoding="utf-8")
                logger.info(f"Saved: {out_path}")

        # ── Summary ──
        if self.verbose:
            self._print_summary(results, split)

        return results

    def _print_summary(
        self, results: Dict[str, pd.DataFrame], split: str
    ) -> None:
        """Print final summary of all processed datasets."""
        print(f"\n{'#'*60}")
        print(f"  PREPROCESSING COMPLETE — {split.upper()} SPLIT")
        print(f"{'#'*60}")
        print(f"\n  {'Dataset':<30s} {'Samples':>10s} {'Avg Words':>10s} {'Avg CMI':>10s}")
        print(f"  {'─'*60}")

        total = 0
        for key, df in results.items():
            total += len(df)
            avg_words = df["word_count"].mean() if "word_count" in df else 0
            avg_cmi = (
                df["code_mix_index"].mean()
                if "code_mix_index" in df
                else 0
            )
            print(
                f"  {key:<30s} {len(df):>10,} {avg_words:>10.1f} {avg_cmi:>9.1f}%"
            )
        print(f"  {'─'*60}")
        print(f"  {'TOTAL':<30s} {total:>10,}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI / ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Run the preprocessing pipeline from the command line.

    Usage:
        python preprocessing_pipeline.py [--split SPLIT] [--filter-mode MODE]
                                         [--no-script-features] [--quiet]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="DravidianCodeMix-2020 Preprocessing Pipeline"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "dev", "test", "full"],
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--filter-mode",
        type=str,
        default="remove",
        choices=["remove", "keep", "tag"],
        help="How to handle non-target-language samples (default: remove)",
    )
    parser.add_argument(
        "--no-script-features",
        action="store_true",
        help="Disable code-mixing feature extraction",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Process a single dataset (e.g., 'tamil_sentiment')",
    )

    args = parser.parse_args()

    pipeline = PreprocessingPipeline(
        filter_mode=args.filter_mode,
        include_script_features=not args.no_script_features,
        verbose=not args.quiet,
    )

    if args.dataset:
        # Process single dataset
        task = "offensive" if "offensive" in args.dataset else "sentiment"
        df = DataLoader.load_dataset(args.dataset, split=args.split, task=task)
        processed = pipeline.process_dataframe(
            df, task=task, dataset_name=f"{args.dataset}_{args.split}"
        )
        # Save
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(
            OUTPUT_DIR, f"{args.dataset}_{args.split}_preprocessed.csv"
        )
        processed.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"Saved: {out_path}")
    else:
        # Process all datasets
        pipeline.process_all(split=args.split, save=True)


if __name__ == "__main__":
    main()
