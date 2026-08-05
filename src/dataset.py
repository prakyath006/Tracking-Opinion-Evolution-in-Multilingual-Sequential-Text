"""
=============================================================================
Dataset Module — PyTorch Dataset Classes for Both Domains
=============================================================================
Provides Dataset and DataLoader utilities for:
  • Domain 1 (Amazon Beauty): User-level review sequences for opinion tracking
  • Domain 2 (DravidianCodeMix): Individual comment-level sentiment samples

Handles variable-length sequences with padding/collation for batched training.

Author : Opinion Evolution Tracking Project
Date   : 2026
=============================================================================
"""

import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from ontology import (
    SentimentState,
    TransitionType,
    TrajectoryType,
    map_labels_to_ontology,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE_DIR)
PREPROCESSED_DIR = os.path.join(WORKSPACE_ROOT, "data", "preprocessed")

# Trajectory label name -> encoded id, derived from the ontology so this module
# can never drift from TrajectoryType. Kept as a module constant because
# scripts/demo_full_project.py imports it for display purposes.
TRAJECTORY_LABELS = {t.name: t.value for t in TrajectoryType}


# ──────────────────────────────────────────────────────────────────────────────
# Ontology-backed label construction
# ──────────────────────────────────────────────────────────────────────────────
# The linear-regression trajectory logic that used to live here
# (compute_trajectory_label / compute_trend_label) has been removed.
# TrajectoryType.compute() and TransitionType.compute() in src/ontology.py are
# now the single canonical implementation for every domain — see the DESIGN
# DECISION block at the top of ontology.py for the rationale.

def build_trend_labels(states: List[SentimentState]) -> List[int]:
    """
    Build per-position pairwise trend labels for a sequence of states.

    Position 0 has no predecessor and is labelled STABLE, matching the
    convention used by the trend head (padded positions use -1 instead).

    Parameters
    ----------
    states : List[SentimentState]
        Sentiment states in temporal order.

    Returns
    -------
    List[int]
        Encoded TransitionType values, one per position.
    """
    trends = [TransitionType.STABLE.value]
    for i in range(1, len(states)):
        trends.append(TransitionType.compute(states[i - 1], states[i]).value)
    return trends


def dravidian_domain_key(language: str) -> str:
    """Ontology DOMAIN_CONFIGS key for a DravidianCodeMix language."""
    return f"dravidian_{language.lower().strip()}"


def read_sentiment_states(df: pd.DataFrame, domain: str) -> List[SentimentState]:
    """
    Read a preprocessed DataFrame's sentiment column as ontology states.

    The raw string 'label' column is the source of truth and is mapped through
    map_labels_to_ontology(), so the domain's label_mapping in ontology.py is
    the only place label spellings are interpreted. The numeric
    'label_encoded' column written by preprocessing.py is treated as a cache
    of that mapping and is only cross-checked, never used as the input.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe with a 'label' (and usually 'label_encoded')
        column.
    domain : str
        Domain key in ontology.DOMAIN_CONFIGS.

    Returns
    -------
    List[SentimentState]
    """
    if "label" not in df.columns:
        raise KeyError(
            f"Expected a raw 'label' column to map through the ontology for "
            f"domain '{domain}'. Columns present: {list(df.columns)}. "
            f"Re-run src/preprocessing.py to regenerate this CSV."
        )

    states = map_labels_to_ontology(df["label"].astype(str).tolist(), domain)

    # Drift check: preprocessing.py's numeric encoding must agree with the
    # ontology. If it does not, the CSV predates an ontology change.
    if "label_encoded" in df.columns:
        encoded = df["label_encoded"].astype(int).tolist()
        mismatches = sum(1 for s, e in zip(states, encoded) if s.value != e)
        if mismatches:
            logger.warning(
                "%s: %d/%d rows where label_encoded disagrees with the "
                "ontology mapping of 'label'. Using the ontology; regenerate "
                "this CSV with src/preprocessing.py to clear this warning.",
                domain, mismatches, len(encoded),
            )

    return states


# ──────────────────────────────────────────────────────────────────────────────
# Domain 1: Amazon Beauty Sequence Dataset
# ──────────────────────────────────────────────────────────────────────────────

class AmazonSequenceDataset(Dataset):
    """
    PyTorch Dataset for Amazon Beauty review sequences.
    
    Each sample is a user's full review sequence (chronologically ordered).
    Returns the text list, per-review sentiment labels, and sequence-level
    trajectory label.
    """
    
    def __init__(
        self,
        csv_path: Optional[str] = None,
        max_seq_len: int = 20,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        """
        Parameters
        ----------
        csv_path : str or None
            Path to amazon_beauty_sequences.csv. Auto-detected if None.
        max_seq_len : int
            Maximum number of reviews per sequence (truncates longer ones).
        split : str
            One of 'train', 'val', 'test'.
        train_ratio : float
            Fraction of users for training.
        val_ratio : float
            Fraction of users for validation.
        random_seed : int
            Random seed for reproducible splits.
        """
        if csv_path is None:
            csv_path = os.path.join(PREPROCESSED_DIR, "amazon_beauty_sequences.csv")
            
        logger.info(f"Loading Amazon sequences from: {csv_path}")
        df = pd.read_csv(csv_path, encoding="utf-8")
        
        # Group reviews by user_id, keeping chronological order
        self.sequences = []
        grouped = df.groupby("user_id", sort=False)
        
        for user_id, group in grouped:
            group_sorted = group.sort_values("sequence_position")
            
            texts = group_sorted["text"].fillna("").astype(str).tolist()
            ratings = group_sorted["rating"].astype(float).tolist()

            # Truncate long sequences
            if len(texts) > max_seq_len:
                texts = texts[:max_seq_len]
                ratings = ratings[:max_seq_len]

            # The Amazon CSV carries star ratings rather than string labels, so
            # the ontology entry point here is SentimentState.from_rating()
            # instead of map_labels_to_ontology(). label_encoded is no longer
            # read: it is a cache of exactly this mapping (see
            # scripts/download_and_build_amazon_sequences.py) and is verified
            # against the ontology in tests/test_ontology_consistency.py.
            states = [SentimentState.from_rating(r) for r in ratings]
            sentiments = [s.value for s in states]
            trajectory = TrajectoryType.compute(states).value
            trends = build_trend_labels(states)

            self.sequences.append({
                "user_id": user_id,
                "texts": texts,
                "ratings": ratings,
                "sentiments": sentiments,
                "trends": trends,
                "trajectory": trajectory,
                "seq_len": len(texts),
            })
        
        # Split by user
        np.random.seed(random_seed)
        n_total = len(self.sequences)
        indices = np.random.permutation(n_total)
        
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        if split == "train":
            selected = indices[:n_train]
        elif split == "val":
            selected = indices[n_train:n_train + n_val]
        elif split == "test":
            selected = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.sequences = [self.sequences[i] for i in selected]
        
        logger.info(
            f"Amazon {split} set: {len(self.sequences)} users, "
            f"avg seq len: {np.mean([s['seq_len'] for s in self.sequences]):.1f}"
        )
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        seq = self.sequences[idx]
        return {
            "texts": seq["texts"],                                    # List[str]
            "sentiments": torch.tensor(seq["sentiments"], dtype=torch.long),  # [seq_len]
            "trends": torch.tensor(seq["trends"], dtype=torch.long),          # [seq_len]
            "trajectory": torch.tensor(seq["trajectory"], dtype=torch.long),  # scalar
            "seq_len": seq["seq_len"],                                        # int
        }


# ──────────────────────────────────────────────────────────────────────────────
# Domain 2: DravidianCodeMix Dataset
# ──────────────────────────────────────────────────────────────────────────────

class DravidianDataset(Dataset):
    """
    PyTorch Dataset for DravidianCodeMix sentiment/offensive language detection.
    
    Each sample is a single comment with its sentiment label.
    """
    
    def __init__(
        self,
        language: str = "tamil",
        task: str = "sentiment",
        split: str = "train",
        csv_path: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        """
        Parameters
        ----------
        language : str
            One of 'tamil', 'malayalam', 'kannada'.
        task : str
            One of 'sentiment', 'offensive'.
        split : str
            One of 'train', 'val', 'test'.
        csv_path : str or None
            Path to preprocessed CSV. Auto-detected if None.
        train_ratio : float
            Fraction for training split.
        val_ratio : float
            Fraction for validation split.
        random_seed : int
            Random seed for reproducible splits.
        """
        if csv_path is None:
            # Map language names to file prefixes
            lang_prefix = {
                "tamil": "tamil",
                "malayalam": "mal",
                "kannada": "kannada",
            }
            prefix = lang_prefix.get(language, language)
            csv_path = os.path.join(
                PREPROCESSED_DIR, 
                f"{prefix}_{task}_train_preprocessed.csv"
            )
        
        logger.info(f"Loading Dravidian dataset from: {csv_path}")
        df = pd.read_csv(csv_path, encoding="utf-8")
        
        # Store text and label
        self.texts = df["text"].fillna("").tolist()
        if task == "sentiment":
            # Sentiment labels are owned by the ontology.
            states = read_sentiment_states(df, dravidian_domain_key(language))
            self.labels = [s.value for s in states]
        else:
            # The ontology models sentiment only — it has no offensive-language
            # taxonomy — so the offensive task keeps preprocessing.py's own
            # OFFENSIVE_LABELS encoding rather than being forced through
            # SentimentState. See Step 7 note in ontology.py's DOMAIN_CONFIGS.
            self.labels = df["label_encoded"].astype(int).tolist()
        self.language = language
        self.task = task

        # Count unique labels
        self.num_classes = len(set(self.labels))
        
        # Split data
        np.random.seed(random_seed)
        n_total = len(self.texts)
        indices = np.random.permutation(n_total)
        
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        if split == "train":
            selected = indices[:n_train]
        elif split == "val":
            selected = indices[n_train:n_train + n_val]
        elif split == "test":
            selected = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.texts = [self.texts[i] for i in selected]
        self.labels = [self.labels[i] for i in selected]
        
        logger.info(
            f"Dravidian {language}/{task} {split}: {len(self.texts)} samples, "
            f"{self.num_classes} classes"
        )
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict:
        return {
            "text": self.texts[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Domain 2: DravidianCodeMix SEQUENCE Dataset (Sliding Window)
# ──────────────────────────────────────────────────────────────────────────────

class DravidianSequenceDataset(Dataset):
    """
    PyTorch Dataset that creates SEQUENCES from DravidianCodeMix comments
    using a sliding-window approach.
    
    Since the raw DravidianCodeMix data has no thread/video IDs or timestamps,
    consecutive comments are grouped into pseudo-threads of a fixed window
    size (e.g., 5 comments). Within each pseudo-thread, the sequence of
    sentiment labels is used to compute trajectory labels.
    
    This is a standard technique in sequential NLP when explicit thread
    structure is unavailable (see: Bamman et al., 2014; Card et al., 2016).
    """
    
    def __init__(
        self,
        language: str = "tamil",
        task: str = "sentiment",
        split: str = "train",
        csv_path: Optional[str] = None,
        window_size: int = 5,
        stride: int = 3,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: int = 42,
    ):
        """
        Parameters
        ----------
        language : str
            One of 'tamil', 'malayalam', 'kannada'.
        task : str
            One of 'sentiment', 'offensive'.
        split : str
            One of 'train', 'val', 'test'.
        csv_path : str or None
            Path to preprocessed CSV. Auto-detected if None.
        window_size : int
            Number of consecutive comments per pseudo-sequence.
        stride : int
            Step size for sliding window (stride < window = overlapping windows).
        train_ratio : float
            Fraction for training split.
        val_ratio : float
            Fraction for validation split.
        random_seed : int
            Random seed for reproducible splits.
        """
        if csv_path is None:
            lang_prefix = {"tamil": "tamil", "malayalam": "mal", "kannada": "kannada"}
            prefix = lang_prefix.get(language, language)
            csv_path = os.path.join(
                PREPROCESSED_DIR, f"{prefix}_{task}_train_preprocessed.csv"
            )
        
        logger.info(f"Loading Dravidian sequences from: {csv_path}")
        df = pd.read_csv(csv_path, encoding="utf-8")
        
        if task != "sentiment":
            raise ValueError(
                f"DravidianSequenceDataset builds trajectory labels from the "
                f"sentiment ontology and cannot be used with task='{task}'. "
                f"Use DravidianDataset for non-sentiment tasks."
            )

        texts_all = df["text"].fillna("").tolist()
        states_all = read_sentiment_states(df, dravidian_domain_key(language))
        labels_all = [s.value for s in states_all]

        # Build sliding-window sequences
        self.sequences = []
        for start in range(0, len(texts_all) - window_size + 1, stride):
            end = start + window_size
            texts = texts_all[start:end]
            sentiments = labels_all[start:end]
            
            # Derive trajectory and trend labels straight from the ontology
            # states — no pseudo-rating detour is needed any more.
            states = states_all[start:end]
            trajectory = TrajectoryType.compute(states).value
            trends = build_trend_labels(states)

            self.sequences.append({
                "texts": texts,
                "sentiments": sentiments,
                "trends": trends,
                "trajectory": trajectory,
                "seq_len": len(texts),
            })
        
        # Split sequences
        np.random.seed(random_seed)
        n_total = len(self.sequences)
        indices = np.random.permutation(n_total)
        
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        if split == "train":
            selected = indices[:n_train]
        elif split == "val":
            selected = indices[n_train:n_train + n_val]
        elif split == "test":
            selected = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        self.sequences = [self.sequences[i] for i in selected]
        self.language = language
        self.task = task
        
        logger.info(
            f"Dravidian {language} sequences ({split}): {len(self.sequences)} "
            f"pseudo-threads (window={window_size}, stride={stride})"
        )
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        seq = self.sequences[idx]
        return {
            "texts": seq["texts"],
            "sentiments": torch.tensor(seq["sentiments"], dtype=torch.long),
            "trends": torch.tensor(seq["trends"], dtype=torch.long),
            "trajectory": torch.tensor(seq["trajectory"], dtype=torch.long),
            "seq_len": seq["seq_len"],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Collate Functions for DataLoader
# ──────────────────────────────────────────────────────────────────────────────

def sequence_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for AmazonSequenceDataset.
    
    Handles variable-length sequences by padding sentiment, trend tensors
    and collecting text lists.
    
    Parameters
    ----------
    batch : List[Dict]
        List of samples from AmazonSequenceDataset.__getitem__.
        
    Returns
    -------
    Dict with padded tensors and metadata.
    """
    texts_batch = [sample["texts"] for sample in batch]          # List[List[str]]
    seq_lens = [sample["seq_len"] for sample in batch]           # List[int]
    
    # Pad sentiment and trend tensors
    sentiments = pad_sequence(
        [sample["sentiments"] for sample in batch], 
        batch_first=True, 
        padding_value=-1   # -1 = ignore index for CrossEntropyLoss
    )
    trends = pad_sequence(
        [sample["trends"] for sample in batch], 
        batch_first=True, 
        padding_value=-1
    )
    trajectories = torch.stack([sample["trajectory"] for sample in batch])
    
    # Create a padding mask: True where real reviews exist, False for padding
    max_len = max(seq_lens)
    padding_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, length in enumerate(seq_lens):
        padding_mask[i, :length] = True
    
    return {
        "texts": texts_batch,              # List[List[str]]
        "sentiments": sentiments,          # [batch, max_seq_len]
        "trends": trends,                  # [batch, max_seq_len]
        "trajectories": trajectories,      # [batch]
        "seq_lens": seq_lens,              # List[int]
        "padding_mask": padding_mask,      # [batch, max_seq_len]
    }


def simple_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DravidianDataset (single text + label).
    
    Parameters
    ----------
    batch : List[Dict]
        List of samples from DravidianDataset.__getitem__.
        
    Returns
    -------
    Dict with text list and label tensor.
    """
    texts = [sample["text"] for sample in batch]
    labels = torch.stack([sample["label"] for sample in batch])
    
    return {
        "texts": texts,    # List[str]
        "labels": labels,  # [batch]
    }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: Create DataLoaders
# ──────────────────────────────────────────────────────────────────────────────

def get_amazon_dataloader(
    split: str = "train",
    batch_size: int = 16,
    max_seq_len: int = 20,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Creates a DataLoader for Amazon Beauty sequences."""
    dataset = AmazonSequenceDataset(split=split, max_seq_len=max_seq_len, **kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=sequence_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_dravidian_dataloader(
    language: str = "tamil",
    task: str = "sentiment",
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Creates a DataLoader for DravidianCodeMix data (single samples)."""
    dataset = DravidianDataset(
        language=language, task=task, split=split, **kwargs
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=simple_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_dravidian_sequence_dataloader(
    language: str = "tamil",
    task: str = "sentiment",
    split: str = "train",
    batch_size: int = 16,
    window_size: int = 5,
    stride: int = 3,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """Creates a DataLoader for DravidianCodeMix pseudo-sequences."""
    dataset = DravidianSequenceDataset(
        language=language, task=task, split=split,
        window_size=window_size, stride=stride, **kwargs
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=sequence_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

