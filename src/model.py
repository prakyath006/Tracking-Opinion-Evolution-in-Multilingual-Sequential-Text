"""
=============================================================================
Full Model — Opinion Evolution Tracker (End-to-End)
=============================================================================
Assembles all modules into a single PyTorch model:

    Input Text Sequence
        -> Module 2 (Tokenization via MultilingualTokenizer)
        -> Module 3 (mBERT/XLM-R Contextual Embeddings)
        -> Module 4 (Bi-LSTM Sequential Encoder)
        -> Module 5 (Attention Layer)
        -> Module 6 (Multi-Task Classification Heads)

This is the complete "Our Model" from the paper:
    mBERT + Bi-LSTM + Attention + Multi-task

Author : Opinion Evolution Tracking Project
Date   : 2026
=============================================================================
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from tokenization import MultilingualTokenizer
from embeddings import DomainAdaptedEmbeddings
from bilstm import BiLSTMEncoder
from attention import SelfAttention, MultiHeadSequenceAttention
from classifier import MultiTaskClassifier

logger = logging.getLogger(__name__)


class OpinionEvolutionTracker(nn.Module):
    """
    End-to-end model for tracking opinion evolution in multilingual
    sequential text.
    
    Architecture:
    
        Texts (list of review strings per user)
            |
            v
        [mBERT/XLM-R Tokenizer] --> token IDs, attention masks
            |
            v
        [mBERT/XLM-R Encoder] --> 768-dim embeddings per review (CLS pooling)
            |
            v
        [Bi-LSTM] --> sequential hidden states (captures temporal evolution)
            |
            v
        [Attention] --> weighted context vector (highlights important reviews)
            |
            v
        [Multi-Task Heads]
            |-- Sentiment Head (per-review)
            |-- Trend Head (per-review pairwise)
            |-- Trajectory Head (sequence-level)
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        max_token_length: int = 128,
        embedding_strategy: str = "cls",
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
        attention_type: str = "single",
        attention_dim: int = 128,
        attention_num_heads: int = 4,
        attention_dropout: float = 0.1,
        classifier_hidden_dim: int = 128,
        classifier_dropout: float = 0.3,
        num_sentiment_classes: int = 4,
        num_trend_classes: int = 3,
        num_trajectory_classes: int = 4,
        freeze_encoder: bool = True,
        use_cuda: bool = True,
    ):
        """
        Parameters
        ----------
        model_name : str
            HuggingFace model name ('bert-base-multilingual-cased' or 'xlm-roberta-base').
        max_token_length : int
            Maximum subword token length per review.
        embedding_strategy : str
            Pooling strategy for embeddings ('cls', 'mean', 'none').
        lstm_hidden_dim : int
            Hidden dimension for Bi-LSTM.
        lstm_num_layers : int
            Number of LSTM layers.
        lstm_dropout : float
            Dropout for LSTM.
        attention_type : str
            'single' for SelfAttention, 'multi' for MultiHeadSequenceAttention.
        attention_dim : int
            Attention internal dimension.
        attention_num_heads : int
            Number of attention heads (only for 'multi' type).
        attention_dropout : float
            Attention dropout.
        classifier_hidden_dim : int
            Hidden dim for classification heads.
        classifier_dropout : float
            Dropout for classification heads.
        num_sentiment_classes : int
            Number of sentiment categories.
        num_trend_classes : int
            Number of trend categories.
        num_trajectory_classes : int
            Number of trajectory categories.
        freeze_encoder : bool
            If True, freeze the pre-trained encoder weights (recommended for
            smaller datasets to prevent overfitting).
        use_cuda : bool
            Whether to use GPU if available.
        """
        super().__init__()
        
        self.max_token_length = max_token_length
        self.embedding_strategy = embedding_strategy
        self.freeze_encoder = freeze_encoder
        
        # Device setup
        self.device = torch.device(
            "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
        )
        
        # ── Module 2: Tokenizer (not an nn.Module, just a utility) ──
        self.tokenizer = MultilingualTokenizer(model_name)
        
        # ── Module 3: Domain-Adapted Embeddings (Own Embeddings) ──
        self.embedding_generator = DomainAdaptedEmbeddings(
            model_name=model_name, use_cuda=use_cuda,
            finetune_layers=0 if freeze_encoder else 3,
        )
        embedding_dim = self.embedding_generator.get_embedding_dim()  # 768
        
        # ── Module 4: Bi-LSTM ──
        self.bilstm = BiLSTMEncoder(
            input_dim=embedding_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        lstm_output_dim = self.bilstm.get_output_dim()  # hidden * 2
        
        # ── Module 5: Attention ──
        if attention_type == "multi":
            self.attention = MultiHeadSequenceAttention(
                input_dim=lstm_output_dim,
                num_heads=attention_num_heads,
                attention_dim=attention_dim,
                dropout=attention_dropout,
            )
        else:
            self.attention = SelfAttention(
                input_dim=lstm_output_dim,
                attention_dim=attention_dim,
                dropout=attention_dropout,
            )
        
        # ── Module 6: Multi-Task Classifier ──
        self.classifier = MultiTaskClassifier(
            input_dim=lstm_output_dim,
            hidden_dim=classifier_hidden_dim,
            num_sentiment_classes=num_sentiment_classes,
            num_trend_classes=num_trend_classes,
            num_trajectory_classes=num_trajectory_classes,
            dropout=classifier_dropout,
        )
        
        # Move trainable components to device
        self.bilstm.to(self.device)
        self.attention.to(self.device)
        self.classifier.to(self.device)
        
        # Log model summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"OpinionEvolutionTracker initialized on {self.device}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    def encode_texts(
        self, texts_batch: List[List[str]]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Encode a batch of text sequences into embeddings.
        
        Parameters
        ----------
        texts_batch : List[List[str]]
            Batch of text sequences. Each element is a list of review strings
            for one user. E.g., [["review1", "review2"], ["review1", "review2", "review3"]]
            
        Returns
        -------
        embeddings : torch.Tensor
            Shape [batch, max_seq_len, 768]. Padded embeddings.
        seq_lens : List[int]
            Actual sequence lengths.
        """
        batch_size = len(texts_batch)
        seq_lens = [len(texts) for texts in texts_batch]
        max_seq_len = max(seq_lens)
        
        embedding_dim = self.embedding_generator.get_embedding_dim()
        
        # Pre-allocate tensor
        all_embeddings = torch.zeros(
            batch_size, max_seq_len, embedding_dim, device=self.device
        )
        
        # Process each review through tokenizer + encoder
        for i, texts in enumerate(texts_batch):
            if len(texts) == 0:
                continue
            
            # Tokenize all reviews in this sequence at once
            tokenized = self.tokenizer.encode_batch(
                texts, max_length=self.max_token_length
            )
            
            # Generate embeddings (CLS pooling by default)
            with torch.no_grad():
                embeddings = self.embedding_generator.generate_embeddings(
                    tokenized, strategy=self.embedding_strategy
                )
            
            # Store embeddings for valid positions
            all_embeddings[i, :len(texts), :] = embeddings
        
        return all_embeddings, seq_lens
    
    def forward(
        self,
        texts_batch: List[List[str]],
        seq_lens: Optional[List[int]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through all modules.
        
        Parameters
        ----------
        texts_batch : List[List[str]]
            Batch of review sequences. Each element is a list of review
            strings for one user.
        seq_lens : List[int], optional
            Actual sequence lengths. Computed automatically if not provided.
        padding_mask : torch.Tensor, optional
            Boolean mask [batch, max_seq_len], True = valid review position.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            - 'sentiment_logits': [batch, seq_len, num_sentiment_classes]
            - 'trend_logits': [batch, seq_len, num_trend_classes]
            - 'trajectory_logits': [batch, num_trajectory_classes]
            - 'attention_weights': [batch, seq_len]
        """
        # Step 1-2: Tokenize + Generate embeddings
        embeddings, computed_seq_lens = self.encode_texts(texts_batch)
        
        if seq_lens is None:
            seq_lens = computed_seq_lens
        
        # Step 3: Bi-LSTM
        hidden_states, (h_n, c_n) = self.bilstm(embeddings, seq_lens)
        
        # Step 4: Attention
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device)
        else:
            # Create mask from seq_lens
            max_len = hidden_states.size(1)
            padding_mask = torch.zeros(
                len(seq_lens), max_len, dtype=torch.bool, device=self.device
            )
            for i, length in enumerate(seq_lens):
                padding_mask[i, :length] = True
        
        context_vector, attention_weights = self.attention(hidden_states, padding_mask)
        
        # Step 5: Multi-task classification
        predictions = self.classifier(hidden_states, context_vector)
        predictions["attention_weights"] = attention_weights
        
        return predictions
    
    def predict(
        self,
        texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Predict for a single user's review sequence.
        
        Parameters
        ----------
        texts : List[str]
            A single user's chronologically ordered reviews.
            
        Returns
        -------
        Dict with predicted labels and attention weights.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward([texts])
            
            sentiment_preds = outputs["sentiment_logits"][0].argmax(dim=-1)
            trend_preds = outputs["trend_logits"][0].argmax(dim=-1)
            trajectory_pred = outputs["trajectory_logits"][0].argmax(dim=-1)
            attention = outputs["attention_weights"][0]
        
        return {
            "sentiments": sentiment_preds.cpu(),
            "trends": trend_preds.cpu(),
            "trajectory": trajectory_pred.cpu().item(),
            "attention_weights": attention.cpu(),
        }
    
    def get_trainable_params(self) -> list:
        """Returns only the trainable parameters (excludes frozen encoder)."""
        return [p for p in self.parameters() if p.requires_grad]
