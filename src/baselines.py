"""
=============================================================================
Baseline Models for Comparison
=============================================================================
Implements 4 baseline models to compare against the full Opinion Evolution
Tracker (mBERT + Bi-LSTM + Attention + Multi-task):

  Baseline 1: mBERT Sentence-Level     — Fine-tuned mBERT, each review independently
  Baseline 2: XLM-R Sentence-Level     — Fine-tuned XLM-R, each review independently
  Baseline 3: LSTM-only (no attention)  — mBERT + LSTM without attention mechanism
  Baseline 4: TextCNN                   — CNN-based text classifier with pretrained embeddings

These baselines demonstrate the value of each component:
  - Baselines 1,2 show why sequential modeling (LSTM) matters
  - Baseline 3 shows why attention matters
  - Baseline 4 shows why pretrained transformers matter

Author : Opinion Evolution Tracking Project
Date   : 2026
=============================================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from transformers import AutoModel

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 1 & 2: Sentence-Level Transformer Classifier
# ──────────────────────────────────────────────────────────────────────────────

class SentenceLevelTransformer(nn.Module):
    """
    Baselines 1 & 2: Fine-tuned mBERT or XLM-R for sentence-level
    classification (no sequential modeling).
    
    Each review is classified independently — there is no LSTM or attention
    over the sequence. This demonstrates what you lose by ignoring sequential
    opinion evolution.
    
    Architecture:
        Text -> Tokenize -> mBERT/XLM-R -> CLS token -> Linear -> Prediction
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        num_classes: int = 4,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
        use_cuda: bool = True,
    ):
        super().__init__()
        
        self.device = torch.device(
            "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        )
        
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        
        self.to(self.device)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"SentenceLevelTransformer ({model_name}): "
            f"{trainable:,} trainable params"
        )
    
    def forward(self, input_ids, attention_mask, **kwargs):
        """
        Forward pass for a batch of individual texts.
        
        Parameters
        ----------
        input_ids : torch.Tensor
            Shape [batch, seq_len].
        attention_mask : torch.Tensor
            Shape [batch, seq_len].
            
        Returns
        -------
        torch.Tensor
            Logits of shape [batch, num_classes].
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(cls_output)
        
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 3: LSTM-only (No Attention)
# ──────────────────────────────────────────────────────────────────────────────

class LSTMOnlyModel(nn.Module):
    """
    Baseline 3: mBERT embeddings + LSTM, but NO attention mechanism.
    
    Uses the final hidden state of the LSTM directly for classification
    instead of the attention-weighted context vector. This shows the
    value of the attention layer in isolating important reviews.
    
    Architecture:
        Text Sequence -> mBERT (frozen) -> Bi-LSTM -> Final Hidden -> Classification
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 4,
        dropout: float = 0.3,
        use_cuda: bool = True,
    ):
        super().__init__()
        
        self.device = torch.device(
            "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        
        output_dim = hidden_dim * 2  # bidirectional
        
        # Per-review sentiment head
        self.sentiment_head = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        
        # Trajectory head (uses final hidden state)
        self.trajectory_head = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )
        
        self.to(self.device)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"LSTMOnlyModel: {trainable:,} trainable params")
    
    def forward(
        self, 
        embeddings: torch.Tensor,
        seq_lens: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Parameters
        ----------
        embeddings : torch.Tensor
            Shape [batch, seq_len, 768].
        seq_lens : List[int], optional
            Actual sequence lengths.
            
        Returns
        -------
        Dict with sentiment_logits and trajectory_logits.
        """
        embeddings = embeddings.to(self.device)
        
        if seq_lens is not None:
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            max_len = embeddings.size(1)
            seq_lens_clamped = [min(s, max_len) for s in seq_lens]
            packed = pack_padded_sequence(
                embeddings, seq_lens_clamped, batch_first=True, enforce_sorted=False
            )
            output, (h_n, _) = self.lstm(packed)
            hidden_states, _ = pad_packed_sequence(output, batch_first=True, total_length=max_len)
        else:
            hidden_states, (h_n, _) = self.lstm(embeddings)
        
        # Per-review sentiment
        sentiment_logits = self.sentiment_head(hidden_states)
        
        # Trajectory from final hidden state (concat forward + backward)
        forward_final = h_n[-2]
        backward_final = h_n[-1]
        final_hidden = torch.cat([forward_final, backward_final], dim=-1)
        trajectory_logits = self.trajectory_head(final_hidden)
        
        return {
            "sentiment_logits": sentiment_logits,
            "trajectory_logits": trajectory_logits,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Baseline 4: TextCNN
# ──────────────────────────────────────────────────────────────────────────────

class TextCNN(nn.Module):
    """
    Baseline 4: CNN-based text classifier.
    
    Uses 1D convolutions with multiple kernel sizes to capture n-gram
    features. This is a classical baseline that doesn't use pretrained
    transformers or sequential modeling.
    
    Architecture:
        Text -> Embedding -> Conv1D (multiple kernels) -> MaxPool -> FC -> Prediction
    """
    
    def __init__(
        self,
        vocab_size: int = 120000,
        embedding_dim: int = 300,
        num_classes: int = 4,
        num_filters: int = 128,
        kernel_sizes: list = None,
        dropout: float = 0.5,
        use_cuda: bool = True,
    ):
        super().__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4, 5]
        
        self.device = torch.device(
            "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        )
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2,
            )
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        
        self.to(self.device)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"TextCNN: {trainable:,} trainable params")
    
    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape [batch, seq_len].
            
        Returns
        -------
        torch.Tensor
            Logits of shape [batch, num_classes].
        """
        input_ids = input_ids.to(self.device)
        
        # Embedding: [batch, seq_len, embedding_dim]
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # Transpose for Conv1d: [batch, embedding_dim, seq_len]
        embedded = embedded.transpose(1, 2)
        
        # Apply each conv filter + max-pool
        conv_outputs = []
        for conv in self.convs:
            x = F.relu(conv(embedded))       # [batch, num_filters, seq_len]
            x = F.max_pool1d(x, x.size(2))  # [batch, num_filters, 1]
            x = x.squeeze(2)                 # [batch, num_filters]
            conv_outputs.append(x)
        
        # Concatenate filter outputs
        combined = torch.cat(conv_outputs, dim=1)  # [batch, num_filters * len(kernels)]
        combined = self.dropout(combined)
        
        logits = self.classifier(combined)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# Ablation: Attention-Only Model (No Bi-LSTM)
# ──────────────────────────────────────────────────────────────────────────────

class AttentionOnlyModel(nn.Module):
    """
    Ablation variant: Attention directly over embeddings (no Bi-LSTM).
    
    Demonstrates the value of the Bi-LSTM component by removing it.
    Attention is applied directly to the 768-dim embeddings without
    any sequential encoding.
    
    Architecture:
        Embeddings -> Attention -> Classification (no LSTM in between)
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        attention_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.3,
        use_cuda: bool = True,
    ):
        super().__init__()
        
        self.device = torch.device(
            "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        )
        
        # Attention directly on embeddings (no LSTM)
        self.W = nn.Linear(embedding_dim, attention_dim)
        self.u = nn.Linear(attention_dim, 1, bias=False)
        self.tanh = nn.Tanh()
        
        # Per-review sentiment head
        self.sentiment_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        
        # Trajectory head (from attended context)
        self.trajectory_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )
        
        self.dropout = nn.Dropout(dropout)
        self.to(self.device)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"AttentionOnlyModel: {trainable:,} trainable params")
    
    def forward(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        embeddings = embeddings.to(self.device)
        
        # Attention directly on embeddings
        energy = self.tanh(self.W(embeddings))
        scores = self.u(energy).squeeze(-1)
        
        if mask is not None:
            mask = mask.to(self.device)
            scores = scores.masked_fill(~mask, float("-inf"))
        
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        context = torch.bmm(weights.unsqueeze(1), embeddings).squeeze(1)
        
        # Predictions
        sentiment_logits = self.sentiment_head(embeddings)
        trajectory_logits = self.trajectory_head(context)
        
        return {
            "sentiment_logits": sentiment_logits,
            "trajectory_logits": trajectory_logits,
            "attention_weights": weights,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Baseline Comparison Summary
# ──────────────────────────────────────────────────────────────────────────────

BASELINE_REGISTRY = {
    "mbert_sentence": {
        "class": SentenceLevelTransformer,
        "default_args": {"model_name": "bert-base-multilingual-cased"},
        "description": "mBERT fine-tuned per-sentence (no sequential modeling)",
    },
    "xlmr_sentence": {
        "class": SentenceLevelTransformer,
        "default_args": {"model_name": "xlm-roberta-base"},
        "description": "XLM-R fine-tuned per-sentence (no sequential modeling)",
    },
    "lstm_only": {
        "class": LSTMOnlyModel,
        "default_args": {},
        "description": "mBERT + Bi-LSTM (no attention mechanism)",
    },
    "attention_only": {
        "class": AttentionOnlyModel,
        "default_args": {},
        "description": "Attention over embeddings (no Bi-LSTM)",
    },
    "textcnn": {
        "class": TextCNN,
        "default_args": {},
        "description": "TextCNN with word embeddings (no transformers, no LSTM)",
    },
}


def get_baseline_model(name: str, **kwargs) -> nn.Module:
    """
    Factory function to instantiate a baseline model by name.
    
    Parameters
    ----------
    name : str
        One of: 'mbert_sentence', 'xlmr_sentence', 'lstm_only',
                'attention_only', 'textcnn'
    **kwargs
        Override default arguments.
    
    Returns
    -------
    nn.Module
        The instantiated baseline model.
    """
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline: {name}. "
            f"Available: {list(BASELINE_REGISTRY.keys())}"
        )
    
    entry = BASELINE_REGISTRY[name]
    args = {**entry["default_args"], **kwargs}
    
    logger.info(f"Creating baseline: {name} -- {entry['description']}")
    return entry["class"](**args)

