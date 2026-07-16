"""
=============================================================================
Module 5 — Attention Layer for Semantic Transition Detection
=============================================================================
Self-attention mechanism applied over the Bi-LSTM hidden states to identify
which reviews in a user's sequence are most important for understanding the
overall opinion trajectory.

Key Insight: Not all reviews matter equally. A user who writes 10 neutral
reviews and then 1 strongly negative review has a clear opinion shift.
Attention learns to weight that negative review more heavily.

Architecture:
    Input:  [batch, seq_len, hidden*2]  (from Bi-LSTM)
    Output: [batch, hidden*2]            (attended context vector)
            + attention weights [batch, seq_len] (for interpretability)

Author : Opinion Evolution Tracking Project
Date   : 2026
=============================================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class SelfAttention(nn.Module):
    """
    Additive (Bahdanau-style) self-attention over sequence hidden states.
    
    Learns to assign importance weights to each review position in the
    sequence, producing a weighted context vector that captures the most
    informative parts of the opinion evolution.
    
    The attention weights are also returned for interpretability —
    you can visualize which reviews the model considers most important
    for its predictions.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        attention_dim: int = 128,
        dropout: float = 0.1,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of input hidden states (hidden*2 from Bi-LSTM).
        attention_dim : int
            Dimensionality of the internal attention space.
        dropout : float
            Dropout on attention weights to prevent overfitting.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        
        # Learnable attention parameters:
        # W: projects hidden states into attention space
        # u: context vector that measures alignment
        self.W = nn.Linear(input_dim, attention_dim, bias=True)
        self.u = nn.Linear(attention_dim, 1, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        
        logger.info(
            f"SelfAttention initialized: input_dim={input_dim}, "
            f"attention_dim={attention_dim}"
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted context vector.
        
        Parameters
        ----------
        hidden_states : torch.Tensor
            Bi-LSTM outputs of shape [batch, seq_len, input_dim].
        mask : torch.Tensor, optional
            Boolean mask of shape [batch, seq_len]. True = valid, False = padded.
            Padded positions get -inf attention score (zero weight after softmax).
            
        Returns
        -------
        context : torch.Tensor
            Weighted context vector of shape [batch, input_dim].
        attention_weights : torch.Tensor
            Normalized attention weights of shape [batch, seq_len].
            Sums to 1 across the sequence dimension for each sample.
        """
        # Step 1: Project hidden states to attention space
        # [batch, seq_len, input_dim] -> [batch, seq_len, attention_dim]
        energy = self.tanh(self.W(hidden_states))
        
        # Step 2: Compute alignment scores
        # [batch, seq_len, attention_dim] -> [batch, seq_len, 1] -> [batch, seq_len]
        scores = self.u(energy).squeeze(-1)
        
        # Step 3: Mask padded positions with -inf (so softmax gives them 0 weight)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        
        # Step 4: Normalize scores to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Step 5: Weighted sum of hidden states
        # [batch, 1, seq_len] @ [batch, seq_len, input_dim] -> [batch, 1, input_dim]
        context = torch.bmm(attention_weights.unsqueeze(1), hidden_states)
        context = context.squeeze(1)  # [batch, input_dim]
        
        return context, attention_weights


class MultiHeadSequenceAttention(nn.Module):
    """
    Multi-head attention variant for richer representation.
    
    Uses multiple attention heads to capture different aspects of
    opinion evolution (e.g., one head for sentiment shifts, another
    for topic changes).
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        num_heads: int = 4,
        attention_dim: int = 128,
        dropout: float = 0.1,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Input hidden dimension.
        num_heads : int
            Number of attention heads.
        attention_dim : int
            Attention space dimension per head.
        dropout : float
            Dropout probability.
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.input_dim = input_dim
        
        # Multiple attention heads
        self.heads = nn.ModuleList([
            SelfAttention(input_dim, attention_dim, dropout)
            for _ in range(num_heads)
        ])
        
        # Projection to combine multi-head outputs back to input_dim
        self.output_projection = nn.Sequential(
            nn.Linear(input_dim * num_heads, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        logger.info(
            f"MultiHeadSequenceAttention: {num_heads} heads, "
            f"input_dim={input_dim}, attention_dim={attention_dim}"
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all attention heads.
        
        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape [batch, seq_len, input_dim].
        mask : torch.Tensor, optional
            Shape [batch, seq_len], True = valid.
            
        Returns
        -------
        context : torch.Tensor
            Combined context vector, shape [batch, input_dim].
        attention_weights : torch.Tensor
            Averaged attention weights across heads, shape [batch, seq_len].
        """
        contexts = []
        all_weights = []
        
        for head in self.heads:
            ctx, weights = head(hidden_states, mask)
            contexts.append(ctx)
            all_weights.append(weights)
        
        # Concatenate all head outputs
        combined = torch.cat(contexts, dim=-1)  # [batch, input_dim * num_heads]
        
        # Project back to input_dim
        context = self.output_projection(combined)  # [batch, input_dim]
        
        # Average attention weights across heads for interpretability
        attention_weights = torch.stack(all_weights, dim=0).mean(dim=0)  # [batch, seq_len]
        
        return context, attention_weights
