"""
=============================================================================
Module 3 — Contextual Embedding Generation Module
=============================================================================
This module loads the pre-trained transformer model (mBERT or XLM-RoBERTa),
accepts inputs from the Tokenization Module (Module 2), passes them through
the model's encoder layers in evaluation/inference mode, and extracts the
768-dimensional contextual embeddings.

Aligns with:
- Gap 1 (Capturing Sequential Opinion Evolution) by producing inputs for Bi-LSTM.
- Gap 2 (Limited Semantic Representation) by obtaining contextual embeddings
  from pre-trained encoders.
"""

import logging
from typing import Dict, Union
import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)

class ContextualEmbeddingGenerator(nn.Module):
    """
    Contextual Embedding Generator that wraps pre-trained HuggingFace models
    (such as mBERT or XLM-R), manages GPU placement, and extracts different
    pooling strategies (CLS, Mean, or raw word-level token states).
    """
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased", use_cuda: bool = True):
        """
        Initializes the contextual embedding generator.
        
        Parameters
        ----------
        model_name : str
            The HuggingFace model name (e.g. 'bert-base-multilingual-cased', 'xlm-roberta-base').
        use_cuda : bool
            Whether to attempt moving the model to GPU (CUDA) for acceleration.
        """
        super().__init__()
        self.model_name = model_name
        
        logger.info(f"Loading pre-trained transformer model: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Set device dynamically based on GPU availability and parameter
        self.device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        self.encoder.to(self.device)
        logger.info(f"Model placed on device: {self.device}")
        
        # Put the model in evaluation mode to turn off dropout/batchnorm updates
        self.encoder.eval()
        
    def forward(self, tokenized_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to get the full raw hidden states from the encoder.
        
        Parameters
        ----------
        tokenized_inputs : Dict[str, torch.Tensor]
            Dictionary containing token IDs, attention masks, etc., from the tokenizer.
            
        Returns
        -------
        torch.Tensor
            The last hidden state tensor from the encoder (shape: [batch_size, seq_len, hidden_size]).
        """
        # Move all input tensors to the same device as the model
        inputs_on_device = {key: val.to(self.device) for key, val in tokenized_inputs.items()}
        
        # Run forward pass. We use torch.no_grad() because this module only handles
        # embedding extraction (inference), freezing the encoder weights to save memory.
        with torch.no_grad():
            outputs = self.encoder(**inputs_on_device)
            
        # The first output is always the last hidden state of all tokens.
        # Shape: [batch_size, sequence_length, hidden_dimension (768)]
        return outputs.last_hidden_state
        
    def generate_embeddings(
        self, 
        tokenized_inputs: Dict[str, torch.Tensor], 
        strategy: str = "cls"
    ) -> torch.Tensor:
        """
        Generates contextual embeddings using the specified pooling strategy.
        
        Parameters
        ----------
        tokenized_inputs : Dict[str, torch.Tensor]
            Tensors returned by the Tokenization Module.
        strategy : str
            Strategy to select representation:
            - 'cls': Extract the CLS token representation (shape: [batch_size, 768]).
            - 'mean': Average across all token states, ignoring padding tokens (shape: [batch_size, 768]).
            - 'none': Return the raw hidden states for all tokens (shape: [batch_size, seq_len, 768]).
            
        Returns
        -------
        torch.Tensor
            Contextual embeddings matrix.
        """
        # Get raw token hidden states from the encoder
        last_hidden_state = self.forward(tokenized_inputs)
        
        if strategy == "none":
            # Return token-level embeddings (Option B from implementation plan)
            return last_hidden_state
            
        elif strategy == "cls":
            # Extract CLS token representation (Option A from implementation plan)
            # The CLS token is always at index 0 of the sequence dimension.
            # Shape: [batch_size, 768]
            return last_hidden_state[:, 0, :]
            
        elif strategy == "mean":
            # Mean pooling strategy (average embeddings of non-pad tokens)
            attention_mask = tokenized_inputs["attention_mask"].to(self.device)
            
            # Expand attention mask to match hidden state dimensions: [batch, seq_len, 1]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # Sum up embeddings where attention mask is 1
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            
            # Avoid division by zero
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            
            # Return averaged embeddings
            return sum_embeddings / sum_mask
            
        else:
            raise ValueError(f"Unknown embedding pooling strategy: {strategy}. Choose 'cls', 'mean', or 'none'.")
            
    def get_embedding_dim(self) -> int:
        """Returns the dimensions of the generated embeddings (typically 768)."""
        return self.encoder.config.hidden_size
