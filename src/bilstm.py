"""
=============================================================================
Module 4 — Bi-LSTM Sequential Encoder
=============================================================================
Takes 768-dimensional contextual embeddings from Module 3 (one per review in
a user's timeline) and models the sequential dependencies across the review
sequence using a Bidirectional LSTM.

Addresses Gap 1: "Transformers fail to capture sequential opinion evolution"
— Standard BERT processes each review independently. The Bi-LSTM explicitly
models how a user's opinion CHANGES over time across multiple reviews.

Architecture:
    Input:  [batch, seq_len, 768]    (embeddings per review)
    Output: [batch, seq_len, hidden*2]  (bidirectional hidden states)

Author : Opinion Evolution Tracking Project
Date   : 2026
=============================================================================
"""

import logging
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for modeling sequential opinion evolution.
    
    Reads a sequence of review embeddings and captures both forward
    (past → future) and backward (future → past) temporal dependencies.
    
    The forward direction captures how past opinions influence current ones.
    The backward direction captures how future opinions contextualize earlier ones.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of input embeddings (768 for mBERT/XLM-R).
        hidden_dim : int
            Number of hidden units per direction per layer.
        num_layers : int
            Number of stacked LSTM layers (depth).
        dropout : float
            Dropout between LSTM layers (applied if num_layers > 1).
        bidirectional : bool
            If True, use bidirectional LSTM (doubles output dimension).
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = hidden_dim * self.num_directions
        
        # Input projection: optional linear layer to adapt embedding dim
        # to LSTM input dim if they differ
        self.input_projection = None
        if input_dim != hidden_dim:
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            lstm_input_dim = hidden_dim
        else:
            lstm_input_dim = input_dim
        
        # Core Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        # Layer normalization on output for training stability
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
        # Output dropout
        self.dropout = nn.Dropout(dropout)
        
        logger.info(
            f"BiLSTM initialized: input={input_dim}, hidden={hidden_dim}, "
            f"layers={num_layers}, bidirectional={bidirectional}, "
            f"output_dim={self.output_dim}"
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,
        seq_lens: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the Bi-LSTM.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Review embeddings of shape [batch, seq_len, input_dim].
            Each position represents one review's embedding in the sequence.
        seq_lens : list of int, optional
            Actual lengths of each sequence in the batch (before padding).
            If provided, uses PackedSequence for efficiency and correctness.
            
        Returns
        -------
        hidden_states : torch.Tensor
            Shape [batch, seq_len, hidden*2]. The bidirectional hidden state
            at each review position in the sequence.
        (h_n, c_n) : Tuple[torch.Tensor, torch.Tensor]
            Final hidden and cell states of the LSTM.
        """
        batch_size = embeddings.size(0)
        
        # Optional input projection
        if self.input_projection is not None:
            embeddings = self.input_projection(embeddings)
        
        # Pack sequences for efficient processing (ignores padding)
        if seq_lens is not None:
            # Clamp seq_lens to actual tensor dimension to avoid errors
            max_len = embeddings.size(1)
            seq_lens_clamped = [min(s, max_len) for s in seq_lens]
            
            packed = pack_padded_sequence(
                embeddings,
                lengths=seq_lens_clamped,
                batch_first=True,
                enforce_sorted=False,  # Allows unsorted sequences
            )
            packed_output, (h_n, c_n) = self.lstm(packed)
            hidden_states, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=max_len
            )
        else:
            hidden_states, (h_n, c_n) = self.lstm(embeddings)
        
        # Apply layer normalization and dropout
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states, (h_n, c_n)
    
    def get_output_dim(self) -> int:
        """Returns the output dimensionality (hidden * num_directions)."""
        return self.output_dim
    
    def get_final_hidden(
        self, h_n: torch.Tensor
    ) -> torch.Tensor:
        """
        Extracts the final hidden state from both directions and concatenates.
        
        Parameters
        ----------
        h_n : torch.Tensor
            Final hidden state from LSTM, shape [num_layers*num_dir, batch, hidden].
            
        Returns
        -------
        torch.Tensor
            Concatenated final hidden state, shape [batch, hidden*2].
        """
        if self.bidirectional:
            # h_n has shape [num_layers * 2, batch, hidden]
            # Take the last layer's forward and backward hidden states
            forward_final = h_n[-2]   # Last layer, forward direction
            backward_final = h_n[-1]  # Last layer, backward direction
            return torch.cat([forward_final, backward_final], dim=-1)
        else:
            return h_n[-1]  # Last layer's hidden state
