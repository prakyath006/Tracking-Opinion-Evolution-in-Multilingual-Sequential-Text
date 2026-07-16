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

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BASE_DIR)
PREPROCESSED_DIR = os.path.join(WORKSPACE_ROOT, "data", "preprocessed")

# Trajectory labels for sequence-level classification
TRAJECTORY_LABELS = {
    "IMPROVING": 0,   # Sentiment going up over time
    "DECLINING": 1,   # Sentiment going down over time
    "STABLE": 2,      # Sentiment stays roughly the same
    "VOLATILE": 3,    # Sentiment fluctuates unpredictably
}


def compute_trajectory_label(ratings: List[float]) -> int:
    """
    Compute the trajectory label from a sequence of ratings.
    
    Uses linear regression slope to determine trend direction,
    and variance to detect volatility.
    
    Parameters
    ----------
    ratings : List[float]
        Chronological list of star ratings (1-5 scale).
        
    Returns
    -------
    int
        Trajectory label: 0=IMPROVING, 1=DECLINING, 2=STABLE, 3=VOLATILE
    """
    if len(ratings) < 2:
        return TRAJECTORY_LABELS["STABLE"]
    
    ratings_arr = np.array(ratings, dtype=np.float32)
    n = len(ratings_arr)
    
    # Linear regression: fit slope
    x = np.arange(n, dtype=np.float32)
    x_mean = x.mean()
    y_mean = ratings_arr.mean()
    slope = np.sum((x - x_mean) * (ratings_arr - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-9)
    
    # Variance for volatility detection
    variance = np.var(ratings_arr)
    
    # Thresholds (tuned for 1-5 star scale)
    slope_threshold = 0.15    # Minimum slope magnitude for a clear trend
    volatility_threshold = 1.5  # Variance threshold for volatile sequences
    
    if variance > volatility_threshold:
        return TRAJECTORY_LABELS["VOLATILE"]
    elif slope > slope_threshold:
        return TRAJECTORY_LABELS["IMPROVING"]
    elif slope < -slope_threshold:
        return TRAJECTORY_LABELS["DECLINING"]
    else:
        return TRAJECTORY_LABELS["STABLE"]


def compute_trend_label(prev_rating: float, curr_rating: float) -> int:
    """
    Compute pairwise trend between consecutive reviews.
    
    Parameters
    ----------
    prev_rating : float
        Previous review's rating.
    curr_rating : float
        Current review's rating.
        
    Returns
    -------
    int
        0=IMPROVING, 1=DECLINING, 2=STABLE
    """
    diff = curr_rating - prev_rating
    if diff > 0.5:
        return 0  # IMPROVING
    elif diff < -0.5:
        return 1  # DECLINING
    else:
        return 2  # STABLE


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
            
            texts = group_sorted["text"].tolist()
            ratings = group_sorted["rating"].astype(float).tolist()
            sentiments = group_sorted["label_encoded"].astype(int).tolist()
            
            # Truncate long sequences
            if len(texts) > max_seq_len:
                texts = texts[:max_seq_len]
                ratings = ratings[:max_seq_len]
                sentiments = sentiments[:max_seq_len]
            
            # Compute trajectory label from ratings
            trajectory = compute_trajectory_label(ratings)
            
            # Compute pairwise trend labels
            trends = [2]  # First review has no previous → STABLE
            for i in range(1, len(ratings)):
                trends.append(compute_trend_label(ratings[i-1], ratings[i]))
            
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

# Sentiment label mapping for DravidianCodeMix (from preprocessing.py)
DRAVIDIAN_SENTIMENT_TO_RATING = {
    0: 5.0,   # Positive  -> high rating
    1: 1.0,   # Negative  -> low rating
    2: 3.0,   # Mixed     -> mid rating
    3: 3.0,   # Unknown   -> mid rating
}


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
        
        texts_all = df["text"].fillna("").tolist()
        labels_all = df["label_encoded"].astype(int).tolist()
        
        # Build sliding-window sequences
        self.sequences = []
        for start in range(0, len(texts_all) - window_size + 1, stride):
            end = start + window_size
            texts = texts_all[start:end]
            sentiments = labels_all[start:end]
            
            # Convert sentiment labels to pseudo-ratings for trajectory computation
            ratings = [DRAVIDIAN_SENTIMENT_TO_RATING.get(s, 3.0) for s in sentiments]
            
            # Compute trajectory label
            trajectory = compute_trajectory_label(ratings)
            
            # Compute pairwise trend labels
            trends = [2]  # First position has no predecessor
            for i in range(1, len(ratings)):
                trends.append(compute_trend_label(ratings[i-1], ratings[i]))
            
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

