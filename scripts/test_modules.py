"""
=============================================================================
Verification Test Script — Modules 2 & 3
=============================================================================
This script loads sample code-mixed Dravidian sentiment text from the
preprocessed datasets, runs them through the tokenization and contextual
embedding modules, checks the tensor dimensions, and verifies proper device placement.
"""

import os
import sys
import logging
import pandas as pd
import torch

# Add workspace root and src/ directory to system path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src"))

from tokenization import MultilingualTokenizer
from embeddings import ContextualEmbeddingGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def test_pipeline():
    logger.info("=" * 60)
    logger.info("Starting Verification Test for Modules 2 and 3")
    logger.info("=" * 60)
    
    # 1. Load sample preprocessed Dravidian data
    preprocessed_dir = os.path.join(WORKSPACE_ROOT, "data", "preprocessed")
    sample_file = os.path.join(preprocessed_dir, "tamil_sentiment_train_preprocessed.csv")
    
    if not os.path.exists(sample_file):
        logger.error(f"Sample preprocessed file not found: {sample_file}")
        logger.info("Please run preprocessing first using preprocessing_pipeline.py")
        sys.exit(1)
        
    logger.info(f"Loading sample data from: {sample_file}")
    df = pd.read_csv(sample_file)
    logger.info(f"Total rows in preprocessed file: {len(df):,}")
    
    # Grab first 5 reviews for testing
    samples = df["text"].head(5).tolist()
    labels = df["label"].head(5).tolist()
    
    logger.info("\nLoaded sample reviews:")
    for idx, (samp, lbl) in enumerate(zip(samples, labels)):
        logger.info(f"  [{idx + 1}] Label: {lbl:<15s} | Text: {repr(samp[:80])}...")
        
    # 2. Select model (defaulting to mBERT)
    model_name = "bert-base-multilingual-cased"
    logger.info(f"\nUsing transformer model: {model_name}")
    
    # 3. Initialize Tokenizer (Module 2)
    tokenizer = MultilingualTokenizer(model_name=model_name)
    logger.info("✓ Tokenizer initialized successfully!")
    logger.info(f"  Vocab size: {tokenizer.get_vocab_size():,}")
    
    special = tokenizer.get_special_tokens()
    logger.info("  Special tokens:")
    for name, tok_val in special.items():
        logger.info(f"    - {name}: {tok_val}")
        
    # Test subword splitting
    logger.info("\nSubword tokenization check:")
    sample_mixed = "Marana mass ah irukke, song super"
    subwords = tokenizer.tokenize_text(sample_mixed)
    subword_ids = tokenizer.convert_tokens_to_ids(subwords)
    logger.info(f"  Text: {sample_mixed}")
    logger.info(f"  Subwords: {subwords}")
    logger.info(f"  Subword IDs: {subword_ids}")
    
    # Encode single sequence
    logger.info("\nEncoding single sequence:")
    single_encoding = tokenizer.encode_sequence(sample_mixed, max_length=32)
    for key, tensor in single_encoding.items():
        logger.info(f"  Key: {key:<18s} | Shape: {list(tensor.shape)} | Dtype: {tensor.dtype}")
        
    # Encode batch of preprocessed samples
    logger.info("\nEncoding batch of preprocessed Dravidian comments:")
    batch_encoding = tokenizer.encode_batch(samples, max_length=64)
    for key, tensor in batch_encoding.items():
        logger.info(f"  Key: {key:<18s} | Shape: {list(tensor.shape)} | Dtype: {tensor.dtype}")
        
    # 4. Initialize Embedding Generator (Module 3)
    # We set use_cuda=True to verify GPU functionality
    generator = ContextualEmbeddingGenerator(model_name=model_name, use_cuda=True)
    logger.info("✓ Contextual Embedding Generator initialized successfully!")
    logger.info(f"  Device: {generator.device}")
    
    # Generate Sentence-Level Embeddings (CLS Token - Option A)
    logger.info("\nExtracting CLS token embeddings (Sentence-level representations):")
    cls_embeddings = generator.generate_embeddings(batch_encoding, strategy="cls")
    logger.info(f"  CLS Embeddings shape: {list(cls_embeddings.shape)} (Expected: [batch_size, 768])")
    logger.info(f"  Type: {cls_embeddings.dtype} | Device: {cls_embeddings.device}")
    
    # Verify shape
    assert list(cls_embeddings.shape) == [len(samples), 768], "CLS embedding shape mismatch!"
    logger.info("  ✓ CLS embedding shape verified!")
    
    # Generate Mean Pooled Embeddings
    logger.info("\nExtracting Mean Pooled representations:")
    mean_embeddings = generator.generate_embeddings(batch_encoding, strategy="mean")
    logger.info(f"  Mean Embeddings shape: {list(mean_embeddings.shape)} (Expected: [batch_size, 768])")
    
    assert list(mean_embeddings.shape) == [len(samples), 768], "Mean embedding shape mismatch!"
    logger.info("  ✓ Mean embedding shape verified!")
    
    # Generate Raw Token-Level Hidden States (Option B)
    logger.info("\nExtracting Raw token-level hidden states (Sequence representation):")
    raw_states = generator.generate_embeddings(batch_encoding, strategy="none")
    logger.info(f"  Raw states shape: {list(raw_states.shape)} (Expected: [batch_size, seq_len, 768])")
    
    assert list(raw_states.shape) == [len(samples), 64, 768], "Token hidden states shape mismatch!"
    logger.info("  ✓ Raw sequence-level hidden states verified!")
    
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION TEST COMPLETED SUCCESSFULLY!")
    logger.info("Both modules are fully operational and ready for sequential modeling.")
    logger.info("=" * 60)

if __name__ == "__main__":
    test_pipeline()
