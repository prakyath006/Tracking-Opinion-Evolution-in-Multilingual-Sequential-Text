"""
=============================================================================
PROJECT DEMO — Run this to show all completed work
=============================================================================
Produces a clear, structured output covering:
  1. Project overview
  2. Data statistics (both domains)
  3. Preprocessing results (sample before/after)
  4. Module tests (tensor shapes, architecture)
  5. Model architecture summary
  6. SCS metric demonstration
  7. Baseline comparison summary

Run: python scripts/demo_full_project.py
=============================================================================
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import torch

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src"))

PREPROCESSED_DIR = os.path.join(WORKSPACE_ROOT, "data", "preprocessed")


def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def subsection(title):
    print(f"\n  --- {title} ---")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Project Overview
# ══════════════════════════════════════════════════════════════════════════════

def show_project_overview():
    section("1. PROJECT OVERVIEW")
    print("""
  Title:  Tracking Opinion Evolution in Multilingual Sequential Text
  
  Goal:   Build a model that tracks how user opinions CHANGE over time
          across sequential text (reviews, comments) in multiple languages.
  
  Architecture:
    Input Text --> mBERT/XLM-R --> Bi-LSTM --> Attention --> Multi-Task Heads
                  (Module 2-3)   (Module 4)  (Module 5)    (Module 6)
  
  Domains:
    Domain 1: Amazon Beauty Reviews (English, E-commerce)
    Domain 2: DravidianCodeMix (Tamil/Malayalam/Kannada, Social Media)
  
  Novel Contribution:
    - Sequence Consistency Score (SCS) metric
    - Multi-task opinion evolution tracking
    - Cross-domain multilingual evaluation
  """)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Data Statistics
# ══════════════════════════════════════════════════════════════════════════════

def show_data_stats():
    section("2. DATASET STATISTICS")
    
    subsection("Domain 1: Amazon Beauty (E-commerce)")
    amazon_path = os.path.join(PREPROCESSED_DIR, "amazon_beauty_sequences.csv")
    if os.path.exists(amazon_path):
        df = pd.read_csv(amazon_path)
        n_users = df["user_id"].nunique()
        n_reviews = len(df)
        avg_seq = df.groupby("user_id").size().mean()
        max_seq = df.groupby("user_id").size().max()
        
        print(f"  Total reviews:     {n_reviews:,}")
        print(f"  Total users:       {n_users:,}")
        print(f"  Avg reviews/user:  {avg_seq:.1f}")
        print(f"  Max reviews/user:  {max_seq}")
        print(f"  Rating distribution:")
        for rating, count in df["rating"].value_counts().sort_index().items():
            pct = count / len(df) * 100
            bar = "#" * int(pct / 2)
            print(f"    {rating:.0f} star: {count:6,} ({pct:5.1f}%) {bar}")
        
        print(f"\n  Sentiment label distribution:")
        label_map = {0: "Positive (4-5 stars)", 1: "Negative (1-2 stars)", 2: "Neutral (3 stars)"}
        for label, count in df["label_encoded"].value_counts().sort_index().items():
            pct = count / len(df) * 100
            name = label_map.get(label, f"Label {label}")
            print(f"    {label} ({name:25s}): {count:6,} ({pct:.1f}%)")
    
    subsection("Domain 2: DravidianCodeMix (Social Media)")
    dravidian_files = {
        "Tamil Sentiment": "tamil_sentiment_train_preprocessed.csv",
        "Tamil Offensive": "tamil_offensive_train_preprocessed.csv",
        "Malayalam Sentiment": "mal_sentiment_train_preprocessed.csv",
        "Malayalam Offensive": "mal_offensive_train_preprocessed.csv",
        "Kannada Sentiment": "kannada_sentiment_train_preprocessed.csv",
        "Kannada Offensive": "kannada_offensive_train_preprocessed.csv",
    }
    
    print(f"  {'Dataset':<25} {'Rows':>8} {'Labels':>8} {'Avg Words':>10}")
    print(f"  {'-'*55}")
    
    for name, filename in dravidian_files.items():
        path = os.path.join(PREPROCESSED_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            n_labels = df["label_encoded"].nunique()
            avg_words = df["word_count"].mean() if "word_count" in df.columns else 0
            print(f"  {name:<25} {len(df):>8,} {n_labels:>8} {avg_words:>10.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Preprocessing Demo
# ══════════════════════════════════════════════════════════════════════════════

def show_preprocessing_demo():
    section("3. PREPROCESSING RESULTS (Before vs After)")
    
    subsection("Amazon Beauty - Sample Reviews")
    amazon_path = os.path.join(PREPROCESSED_DIR, "amazon_beauty_sequences.csv")
    if os.path.exists(amazon_path):
        df = pd.read_csv(amazon_path)
        for i in range(3):
            row = df.iloc[i]
            print(f"\n  Sample {i+1} (Rating: {row['rating']:.0f}, Label: {row['label_encoded']}):")
            orig = str(row.get("text_original", ""))[:120]
            clean = str(row["text"])[:120]
            print(f"    ORIGINAL: {orig}...")
            print(f"    CLEANED:  {clean}...")
    
    subsection("DravidianCodeMix Tamil - Sample Comments")
    tamil_path = os.path.join(PREPROCESSED_DIR, "tamil_sentiment_train_preprocessed.csv")
    if os.path.exists(tamil_path):
        df = pd.read_csv(tamil_path)
        for i in range(3):
            row = df.iloc[i]
            print(f"\n  Sample {i+1} (Label: {row['label']}):")
            orig = str(row.get("text_original", ""))[:120]
            clean = str(row["text"])[:120]
            print(f"    ORIGINAL: {orig}...")
            print(f"    CLEANED:  {clean}...")
            if "dominant_script" in df.columns:
                print(f"    Script: {row['dominant_script']}, CMI: {row.get('code_mix_index', 0):.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Module Architecture Test
# ══════════════════════════════════════════════════════════════════════════════

def show_module_tests():
    section("4. MODULE ARCHITECTURE VERIFICATION")
    
    print("\n  Testing tensor flow through all modules...\n")
    
    # Module 4: Bi-LSTM
    from bilstm import BiLSTMEncoder
    bilstm = BiLSTMEncoder(input_dim=768, hidden_dim=256, num_layers=2)
    x = torch.randn(4, 5, 768)
    h, (h_n, c_n) = bilstm(x, seq_lens=[5, 3, 4, 2])
    print(f"  Module 4 (Bi-LSTM):")
    print(f"    Input:  [4, 5, 768]  (4 users, 5 reviews, 768-dim embeddings)")
    print(f"    Output: {list(h.shape)}  (bidirectional hidden states)")
    print(f"    Status: PASS")
    
    # Module 5: Attention
    from attention import SelfAttention
    attn = SelfAttention(input_dim=512, attention_dim=128)
    mask = torch.tensor([[1,1,1,1,1],[1,1,1,0,0],[1,1,1,1,0],[1,1,0,0,0]], dtype=torch.bool)
    ctx, weights = attn(h, mask)
    print(f"\n  Module 5 (Attention):")
    print(f"    Input:  {list(h.shape)}  (Bi-LSTM hidden states)")
    print(f"    Output: {list(ctx.shape)}  (attended context vector)")
    print(f"    Attention weights: {list(weights.shape)}")
    print(f"    Status: PASS")
    
    # Module 6: Multi-Task Classifier
    from classifier import MultiTaskClassifier
    clf = MultiTaskClassifier(input_dim=512, hidden_dim=128)
    outputs = clf(h, ctx)
    print(f"\n  Module 6 (Multi-Task Classifier):")
    print(f"    Sentiment logits:  {list(outputs['sentiment_logits'].shape)}  (per-review, 4 classes)")
    print(f"    Trend logits:      {list(outputs['trend_logits'].shape)}  (per-review, 3 classes)")
    print(f"    Trajectory logits: {list(outputs['trajectory_logits'].shape)}  (per-sequence, 4 classes)")
    print(f"    Status: PASS")
    
    # Module 7: Evaluation
    from evaluation import compute_classification_metrics, sequence_consistency_score
    y_true = [0, 1, 2, 0, 1, 0, 2, 1, 0, 0]
    y_pred = [0, 1, 2, 0, 2, 0, 2, 1, 1, 0]
    metrics = compute_classification_metrics(y_true, y_pred)
    print(f"\n  Module 7 (Evaluation):")
    print(f"    Test accuracy: {metrics['accuracy']:.4f}")
    print(f"    Test F1 macro: {metrics['f1_macro']:.4f}")
    print(f"    Status: PASS")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: SCS Metric Demo
# ══════════════════════════════════════════════════════════════════════════════

def show_scs_demo():
    section("5. NOVEL METRIC: Sequence Consistency Score (SCS)")
    
    from evaluation import sequence_consistency_score
    
    print("""
  SCS measures how consistent a model's predictions are across a sequence.
  Formula: SCS = 1 - (label_flips / total_transitions)
  
  Higher SCS = more coherent predictions (fewer random oscillations)
  """)
    
    examples = [
        ("Consistent (good)",  [0, 0, 0, 0, 0]),
        ("Smooth shift",       [0, 0, 1, 1, 1]),
        ("Gradual decline",    [0, 0, 1, 1, 2]),
        ("Volatile (bad)",     [0, 1, 0, 1, 0]),
        ("Very volatile",      [0, 2, 1, 0, 2]),
    ]
    
    print(f"  {'Pattern':<25} {'Sequence':<25} {'SCS':>8}")
    print(f"  {'-'*60}")
    
    for name, seq in examples:
        scs = sequence_consistency_score([seq])
        labels = " -> ".join(["Pos" if s==0 else "Neg" if s==1 else "Neu" for s in seq])
        print(f"  {name:<25} {labels:<25} {scs['scs_mean']:>8.4f}")
    
    print(f"\n  Interpretation:")
    print(f"    SCS = 1.00: Perfectly consistent (no changes)")
    print(f"    SCS = 0.75: One opinion shift in the sequence")
    print(f"    SCS = 0.00: Maximum volatility (every step changes)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Model Summary
# ══════════════════════════════════════════════════════════════════════════════

def show_model_summary():
    section("6. MODEL ARCHITECTURE SUMMARY")
    
    from bilstm import BiLSTMEncoder
    from attention import SelfAttention, MultiHeadSequenceAttention
    from classifier import MultiTaskClassifier
    
    bilstm = BiLSTMEncoder(input_dim=768, hidden_dim=256, num_layers=2)
    attn = SelfAttention(input_dim=512, attention_dim=128)
    multi_attn = MultiHeadSequenceAttention(input_dim=512, num_heads=4, attention_dim=128)
    clf = MultiTaskClassifier(input_dim=512, hidden_dim=128)
    
    components = [
        ("mBERT Encoder (frozen)", 178_000_000, False),
        ("Bi-LSTM (2 layers)", sum(p.numel() for p in bilstm.parameters()), True),
        ("Self-Attention", sum(p.numel() for p in attn.parameters()), True),
        ("Multi-Task Classifier", sum(p.numel() for p in clf.parameters()), True),
    ]
    
    print(f"\n  {'Component':<30} {'Parameters':>15} {'Trainable':>12}")
    print(f"  {'-'*60}")
    
    total = 0
    trainable = 0
    for name, params, is_trainable in components:
        total += params
        if is_trainable:
            trainable += params
        status = "Yes" if is_trainable else "No (frozen)"
        print(f"  {name:<30} {params:>15,} {status:>12}")
    
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<30} {total:>15,}")
    print(f"  {'TRAINABLE':<30} {trainable:>15,}")
    
    print(f"\n  Baselines for comparison:")
    from baselines import TextCNN, LSTMOnlyModel
    tcnn = TextCNN(vocab_size=1000, num_classes=4)
    lstm = LSTMOnlyModel(num_classes=4)
    
    baselines = [
        ("1. mBERT Sentence-Level", "Fine-tuned mBERT, no LSTM"),
        ("2. XLM-R Sentence-Level", "Fine-tuned XLM-R, no LSTM"),
        ("3. LSTM-Only (no attention)", f"{sum(p.numel() for p in lstm.parameters()):,} params"),
        ("4. TextCNN", f"{sum(p.numel() for p in tcnn.parameters()):,} params"),
        ("5. Our Model (full)", "mBERT + Bi-LSTM + Attention + Multi-task"),
    ]
    
    for name, desc in baselines:
        print(f"    {name}: {desc}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Dataset Loader Demo
# ══════════════════════════════════════════════════════════════════════════════

def show_dataloader_demo():
    section("7. DATASET & DATALOADER DEMO")
    
    from dataset import AmazonSequenceDataset, DravidianDataset, TRAJECTORY_LABELS
    
    subsection("Amazon Beauty Sequences")
    for split in ["train", "val", "test"]:
        ds = AmazonSequenceDataset(split=split)
        print(f"  {split:>5}: {len(ds):,} user sequences")
    
    # Show trajectory distribution
    ds = AmazonSequenceDataset(split="train")
    traj_counts = {}
    for i in range(len(ds)):
        t = ds[i]["trajectory"].item()
        traj_counts[t] = traj_counts.get(t, 0) + 1
    
    inv_labels = {v: k for k, v in TRAJECTORY_LABELS.items()}
    print(f"\n  Trajectory label distribution (train):")
    for label_id in sorted(traj_counts.keys()):
        count = traj_counts[label_id]
        pct = count / len(ds) * 100
        name = inv_labels.get(label_id, f"Unknown({label_id})")
        print(f"    {name:<12}: {count:5,} ({pct:.1f}%)")
    
    subsection("DravidianCodeMix (Single Samples)")
    for lang in ["tamil", "malayalam", "kannada"]:
        try:
            ds = DravidianDataset(language=lang, task="sentiment", split="train")
            print(f"  {lang:>10} sentiment train: {len(ds):,} samples, {ds.num_classes} classes")
        except Exception:
            print(f"  {lang:>10}: file not found")
    
    subsection("DravidianCodeMix SEQUENCES (Sliding Window)")
    from dataset import DravidianSequenceDataset
    for lang in ["tamil", "malayalam", "kannada"]:
        try:
            for split in ["train", "val", "test"]:
                ds = DravidianSequenceDataset(language=lang, task="sentiment", split=split)
                print(f"  {lang:>10} {split:>5}: {len(ds):,} pseudo-thread sequences")
        except Exception:
            print(f"  {lang:>10}: file not found")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: File Structure
# ══════════════════════════════════════════════════════════════════════════════

def show_file_structure():
    section("8. PROJECT FILE STRUCTURE")
    
    src_files = [
        ("preprocessing.py", "Module 1: Text preprocessing pipeline"),
        ("tokenization.py",  "Module 2: mBERT/XLM-R tokenization"),
        ("embeddings.py",    "Module 3: Contextual embedding generation"),
        ("bilstm.py",        "Module 4: Bi-LSTM sequential encoder"),
        ("attention.py",     "Module 5: Attention layer"),
        ("classifier.py",    "Module 6: Multi-task classification heads"),
        ("evaluation.py",    "Module 7: Evaluation metrics + SCS"),
        ("model.py",         "Full end-to-end model"),
        ("dataset.py",       "PyTorch Dataset & DataLoader"),
        ("baselines.py",     "4 baseline models"),
    ]
    
    print(f"\n  src/")
    for fname, desc in src_files:
        path = os.path.join(WORKSPACE_ROOT, "src", fname)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        lines = sum(1 for _ in open(path, encoding="utf-8")) if os.path.exists(path) else 0
        print(f"    {fname:<25} {lines:>5} lines  | {desc}")
    
    print(f"\n  Total source code: {sum(sum(1 for _ in open(os.path.join(WORKSPACE_ROOT, 'src', f), encoding='utf-8')) for f, _ in src_files if os.path.exists(os.path.join(WORKSPACE_ROOT, 'src', f))):,} lines")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    start = time.time()
    
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   TRACKING OPINION EVOLUTION IN MULTILINGUAL SEQUENTIAL TEXT    #")
    print("#   ------------------------------------------------------------ #")
    print("#   Complete Project Demonstration                                #")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    show_project_overview()
    show_data_stats()
    show_preprocessing_demo()
    show_module_tests()
    show_scs_demo()
    show_model_summary()
    show_dataloader_demo()
    show_file_structure()
    
    elapsed = time.time() - start
    
    section("DEMO COMPLETE")
    print(f"\n  All modules verified. Total demo time: {elapsed:.1f}s")
    print(f"  Ready for training: python scripts/train.py --domain amazon --epochs 20")
    print()
