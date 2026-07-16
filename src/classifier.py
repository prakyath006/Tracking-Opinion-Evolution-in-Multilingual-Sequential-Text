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
from typing import Dict, Optional

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
       - Output: [batch, seq_len, 3]  (IMPROVING/DECLINING/STABLE)
       
    3. Trajectory Head: Predicts overall sequence trajectory from context vector
       - Input:  [batch, hidden_dim]  (attended context)
       - Output: [batch, 4]  (IMPROVING/DECLINING/STABLE/VOLATILE)
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_sentiment_classes: int = 4,
        num_trend_classes: int = 3,
        num_trajectory_classes: int = 4,
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
            Number of sentiment categories (e.g., 4 for Pos/Neg/Neutral/Mixed).
        num_trend_classes : int
            Number of trend categories (3: IMPROVING/DECLINING/STABLE).
        num_trajectory_classes : int
            Number of trajectory categories (4: IMPROVING/DECLINING/STABLE/VOLATILE).
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


class MultiTaskLoss(nn.Module):
    """
    Computes weighted multi-task loss combining all three heads.
    
    Total Loss = w1 * SentimentLoss + w2 * TrendLoss + w3 * TrajectoryLoss
    
    Uses CrossEntropyLoss with ignore_index=-1 to handle padded positions
    in variable-length sequences.
    """
    
    def __init__(
        self,
        sentiment_weight: float = 1.0,
        trend_weight: float = 0.5,
        trajectory_weight: float = 1.0,
    ):
        """
        Parameters
        ----------
        sentiment_weight : float
            Weight for the sentiment classification loss.
        trend_weight : float
            Weight for the trend classification loss.
        trajectory_weight : float
            Weight for the trajectory classification loss.
        """
        super().__init__()
        
        self.sentiment_weight = sentiment_weight
        self.trend_weight = trend_weight
        self.trajectory_weight = trajectory_weight
        
        # ignore_index=-1 means padded positions (label=-1) are excluded
        self.sentiment_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.trend_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.trajectory_loss_fn = nn.CrossEntropyLoss()
    
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
