"""
=============================================================================
Module 2 — Tokenization Module
=============================================================================
This module handles loading pre-trained tokenizers (mBERT or XLM-RoBERTa),
subword tokenization, adding special tokens, padding, truncation, generating
attention masks, and preparing tensor inputs compatible with the BERT/XLM-R
contextual encoders.

Aligns with Gap 2 (Limited Semantic Representation Across Multilingual and Code-Mixed Languages)
by leveraging subword tokenizers trained on multilingual and transliterated text.
"""

import logging
from typing import Dict, List, Union
import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class MultilingualTokenizer:
    """
    Tokenizer wrapper that selects and loads the appropriate pre-trained tokenizer
    (mBERT or XLM-R) from HuggingFace, performs subword tokenization, handles
    special tokens, and formats batches as PyTorch tensors ready for the encoder.
    """
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        """
        Initializes the tokenizer.
        
        Parameters
        ----------
        model_name : str
            The name of the HuggingFace pre-trained model.
            Common choices:
            - 'bert-base-multilingual-cased' (mBERT)
            - 'xlm-roberta-base' (XLM-R)
        """
        self.model_name = model_name
        logger.info(f"Loading pre-trained tokenizer: {model_name}")
        
        # Load the tokenizer from HuggingFace cache/server.
        # This will auto-detect the correct subclass (BertTokenizerFast or XLMRobertaTokenizerFast).
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def tokenize_text(self, text: str) -> List[str]:
        """
        Performs raw subword tokenization on a single text string.
        Helps inspect how code-mixed tokens are segmented.
        
        Parameters
        ----------
        text : str
            The input sentence (e.g. code-mixed Dravidian text).
            
        Returns
        -------
        List[str]
            A list of subwords/tokens.
        """
        # We perform tokenization without special tokens to see how words are split.
        return self.tokenizer.tokenize(text)
        
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Maps a list of string tokens to their corresponding vocabulary IDs.
        
        Parameters
        ----------
        tokens : List[str]
            List of subwords.
            
        Returns
        -------
        List[int]
            List of unique vocabulary integer IDs.
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)
        
    def encode_sequence(self, text: str, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        Encodes a single sentence, adding special tokens ([CLS]/<s> and [SEP]/</s>),
        performing truncation, and converting it to PyTorch tensors.
        
        Parameters
        ----------
        text : str
            The input sentence.
        max_length : int
            Maximum length threshold for truncation.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - 'input_ids': ID values including special tokens
            - 'attention_mask': 1s for real tokens, 0s for padding
            - 'token_type_ids': Segment IDs (present for mBERT only)
        """
        # encode_plus handles tokenization, special tokens, mapping to IDs,
        # truncation, attention masks, and returns PyTorch tensors.
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,      # Adds [CLS] at beginning and [SEP] at end
            max_length=max_length,         # Caps sequence length to avoid out of memory
            truncation=True,               # Truncates if text exceeds max_length
            padding="max_length",          # Pad up to max_length for uniform shapes
            return_attention_mask=True,    # Generate attention mask to ignore pad tokens
            return_tensors="pt"            # Return PyTorch tensors directly
        )
        
        # Squeeze batch dimension for a single text representation
        return {key: val.squeeze(0) for key, val in encoding.items()}
        
    def encode_batch(self, texts: List[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        Encodes a list/batch of sentences, performing uniform padding and truncation,
        making them compatible with batch inference in the BERT/XLM-R encoder.
        
        Parameters
        ----------
        texts : List[str]
            List of sentences to process.
        max_length : int
            Maximum sequence length for truncation and padding.
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing tokenized batch tensors:
            - 'input_ids' (shape: [batch_size, max_length])
            - 'attention_mask' (shape: [batch_size, max_length])
            - 'token_type_ids' (optional, shape: [batch_size, max_length])
        """
        # Empty input handling
        if not texts:
            return {}
            
        # batch_encode_plus encodes all sentences together and aligns them via padding.
        encoding = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",          # Standardizes tensor dimensions for parallel computation
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        return dict(encoding)
        
    def get_vocab_size(self) -> int:
        """Gets the total size of the vocabulary."""
        return len(self.tokenizer)
        
    def get_special_tokens(self) -> Dict[str, Union[int, str]]:
        """Gets mapping of special tokens to their names/IDs."""
        return {
            "cls_token": self.tokenizer.cls_token,
            "cls_token_id": self.tokenizer.cls_token_id,
            "sep_token": self.tokenizer.sep_token,
            "sep_token_id": self.tokenizer.sep_token_id,
            "pad_token": self.tokenizer.pad_token,
            "pad_token_id": self.tokenizer.pad_token_id,
            "unk_token": self.tokenizer.unk_token,
            "unk_token_id": self.tokenizer.unk_token_id
        }
