"""
=============================================================================
Module 6 — Multi-Task Classification Heads
=============================================================================
Three classification heads that use the attended sequence representation
from Module 5 to make predictions at different granularities:

  1. Aspect Sentiment Head — per-review sentiment (Positive/Negative/Neutral/Mixed)
  2. Trend Classification Head — pairwise trend (IMPROVING/DECLINING/STABLE)
  3. Trajectory Label Head — sequence-level trajectory (IMPROVING/DECLINING/STABLE/VOLATILE)

Multi-task learning allows these heads to share the underlying representation,
which acts as a regularizer and improves generalization.

Author : Opinion Evolution Tracking Project
Date   : 2026
=============================================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from ontology import SentimentState, TransitionType, TrajectoryType

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """
    A small feed-forward classification head.
    Architecture: Linear -> ReLU -> Dropout -> Linear -> Output
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Input feature dimension.
        hidden_dim : int
            Hidden layer dimension.
        num_classes : int
            Number of output classes.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features.
            
        Returns
        -------
        torch.Tensor
            Logits of shape [..., num_classes].
        """
        return self.classifier(x)


class MultiTaskClassifier(nn.Module):
    """
    Multi-task classification module with three heads:
    
    1. Sentiment Head: Predicts per-review sentiment from Bi-LSTM hidden states
       - Input:  [batch, seq_len, hidden_dim]  (per-review states)
       - Output: [batch, seq_len, num_sentiment_classes]
       
    2. Trend Head: Predicts pairwise opinion trend from Bi-LSTM hidden states
       - Input:  [batch, seq_len, hidden_dim]  (per-review states)
       - Output: [batch, seq_len, num_trend_classes]  (TransitionType)

    3. Trajectory Head: Predicts overall sequence trajectory from context vector
       - Input:  [batch, hidden_dim]  (attended context)
       - Output: [batch, num_trajectory_classes]  (TrajectoryType)

    All three class counts default to the ontology's num_classes(), so the
    heads cannot drift from the taxonomies in src/ontology.py.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_sentiment_classes: int = SentimentState.num_classes(),
        num_trend_classes: int = TransitionType.num_classes(),
        num_trajectory_classes: int = TrajectoryType.num_classes(),
        dropout: float = 0.3,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of Bi-LSTM output (hidden * 2).
        hidden_dim : int
            Hidden dimension for each classification head.
        num_sentiment_classes : int
            Number of sentiment categories. Defaults to
            SentimentState.num_classes().
        num_trend_classes : int
            Number of trend categories. Defaults to
            TransitionType.num_classes().
        num_trajectory_classes : int
            Number of trajectory categories. Defaults to
            TrajectoryType.num_classes().
        dropout : float
            Dropout probability.
        """
        super().__init__()
        
        # Head 1: Per-review sentiment prediction
        self.sentiment_head = ClassificationHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_sentiment_classes,
            dropout=dropout,
        )
        
        # Head 2: Pairwise trend prediction
        self.trend_head = ClassificationHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_trend_classes,
            dropout=dropout,
        )
        
        # Head 3: Sequence-level trajectory prediction
        self.trajectory_head = ClassificationHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_trajectory_classes,
            dropout=dropout,
        )
        
        self.num_sentiment_classes = num_sentiment_classes
        self.num_trend_classes = num_trend_classes
        self.num_trajectory_classes = num_trajectory_classes
        
        logger.info(
            f"MultiTaskClassifier: sentiment={num_sentiment_classes}, "
            f"trend={num_trend_classes}, trajectory={num_trajectory_classes}"
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        context_vector: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all three classification heads.
        
        Parameters
        ----------
        hidden_states : torch.Tensor
            Per-review hidden states from Bi-LSTM. 
            Shape: [batch, seq_len, input_dim]
        context_vector : torch.Tensor
            Attended context vector from Attention layer.
            Shape: [batch, input_dim]
            
        Returns
        -------
        Dict[str, torch.Tensor]
            - 'sentiment_logits': [batch, seq_len, num_sentiment_classes]
            - 'trend_logits': [batch, seq_len, num_trend_classes]
            - 'trajectory_logits': [batch, num_trajectory_classes]
        """
        # Sentiment: predict for each review position
        sentiment_logits = self.sentiment_head(hidden_states)
        
        # Trend: predict for each review position
        trend_logits = self.trend_head(hidden_states)
        
        # Trajectory: predict from the whole-sequence context vector
        trajectory_logits = self.trajectory_head(context_vector)
        
        return {
            "sentiment_logits": sentiment_logits,
            "trend_logits": trend_logits,
            "trajectory_logits": trajectory_logits,
        }


def compute_class_weights(
    labels: List[int],
    num_classes: int,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a label list.

    weight_c = N / (num_classes * count_c), for classes that appear in
    `labels`; classes absent from `labels` get weight 0.0 (there is no
    frequency to invert, and CrossEntropyLoss's `weight` tensor must still
    have one entry per class index).

    This normalization keeps the weights centered around 1.0 (a perfectly
    balanced label set gets all-1.0 weights), so `trend_weight` /
    `trajectory_weight` / `sentiment_weight` task-level scaling in
    MultiTaskLoss keeps meaning what it did before per-class weights existed.

    Parameters
    ----------
    labels : List[int]
        Encoded class ids from the TRAINING split only. Computing this from
        val/test would leak split information into what the model is
        optimized for.
    num_classes : int
        Total classes in the taxonomy (e.g. SentimentState.num_classes()),
        so absent classes still get a weight entry.

    Returns
    -------
    torch.Tensor
        Float tensor of shape [num_classes].
    """
    counts = torch.zeros(num_classes, dtype=torch.float)
    for label in labels:
        if 0 <= label < num_classes:
            counts[label] += 1

    total = counts.sum().item()
    weights = torch.zeros(num_classes, dtype=torch.float)
    if total > 0:
        present = counts > 0
        weights[present] = total / (num_classes * counts[present])
    return weights


class MultiTaskLoss(nn.Module):
    """
    Computes weighted multi-task loss combining all three heads.

    Total Loss = w1 * SentimentLoss + w2 * TrendLoss + w3 * TrajectoryLoss

    Uses CrossEntropyLoss with ignore_index=-1 to handle padded positions
    in variable-length sequences.

    Each head can optionally take a per-class weight tensor (see
    compute_class_weights()) to counteract label imbalance -- trajectory and
    trend labels in particular are heavily skewed toward STABLE across every
    domain (see tests/test_ontology_consistency.py's real-data distributions
    and scripts/train.py's --auto_class_weights flag).
    """

    def __init__(
        self,
        sentiment_weight: float = 1.0,
        trend_weight: float = 0.5,
        trajectory_weight: float = 1.0,
        sentiment_class_weights: Optional[torch.Tensor] = None,
        trend_class_weights: Optional[torch.Tensor] = None,
        trajectory_class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        sentiment_weight : float
            Task-level weight for the sentiment classification loss.
        trend_weight : float
            Task-level weight for the trend classification loss.
        trajectory_weight : float
            Task-level weight for the trajectory classification loss.
        sentiment_class_weights : torch.Tensor, optional
            Per-class weight tensor of shape [num_sentiment_classes], e.g.
            from compute_class_weights(). None means unweighted (uniform).
        trend_class_weights : torch.Tensor, optional
            Per-class weight tensor of shape [num_trend_classes].
        trajectory_class_weights : torch.Tensor, optional
            Per-class weight tensor of shape [num_trajectory_classes].
        """
        super().__init__()

        self.sentiment_weight = sentiment_weight
        self.trend_weight = trend_weight
        self.trajectory_weight = trajectory_weight

        # ignore_index=-1 means padded positions (label=-1) are excluded.
        # `weight` is registered as a buffer by nn.CrossEntropyLoss, so it
        # moves with the rest of this module on MultiTaskLoss(...).to(device).
        self.sentiment_loss_fn = nn.CrossEntropyLoss(
            ignore_index=-1, weight=sentiment_class_weights,
        )
        self.trend_loss_fn = nn.CrossEntropyLoss(
            ignore_index=-1, weight=trend_class_weights,
        )
        self.trajectory_loss_fn = nn.CrossEntropyLoss(
            weight=trajectory_class_weights,
        )
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Parameters
        ----------
        predictions : Dict[str, torch.Tensor]
            Output from MultiTaskClassifier.forward().
        targets : Dict[str, torch.Tensor]
            - 'sentiments': [batch, seq_len] with values 0..num_classes-1 or -1
            - 'trends': [batch, seq_len] with values 0..2 or -1
            - 'trajectories': [batch] with values 0..3
            
        Returns
        -------
        Dict[str, torch.Tensor]
            - 'total_loss': weighted sum of all losses
            - 'sentiment_loss': individual sentiment loss
            - 'trend_loss': individual trend loss
            - 'trajectory_loss': individual trajectory loss
        """
        # Sentiment loss: reshape [batch, seq_len, C] -> [batch*seq_len, C]
        sent_logits = predictions["sentiment_logits"]
        sent_targets = targets["sentiments"]
        batch_size, seq_len, n_classes = sent_logits.shape
        
        sentiment_loss = self.sentiment_loss_fn(
            sent_logits.reshape(-1, n_classes),
            sent_targets.reshape(-1),
        )
        
        # Trend loss: same reshaping
        trend_logits = predictions["trend_logits"]
        trend_targets = targets["trends"]
        n_trend_classes = trend_logits.shape[-1]
        
        trend_loss = self.trend_loss_fn(
            trend_logits.reshape(-1, n_trend_classes),
            trend_targets.reshape(-1),
        )
        
        # Trajectory loss: already [batch, C] vs [batch]
        trajectory_loss = self.trajectory_loss_fn(
            predictions["trajectory_logits"],
            targets["trajectories"],
        )
        
        # Weighted total
        total_loss = (
            self.sentiment_weight * sentiment_loss
            + self.trend_weight * trend_loss
            + self.trajectory_weight * trajectory_loss
        )
        
        return {
            "total_loss": total_loss,
            "sentiment_loss": sentiment_loss,
            "trend_loss": trend_loss,
            "trajectory_loss": trajectory_loss,
        }
