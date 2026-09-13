# Steps B2 & B3 — Encoder Backbone and Sequential Architecture Comparison

## Step B2: BERT Encoder & MLM Perplexity Comparison

| Model | Encoder Architecture | Vocabulary | MLM Perplexity Applicability |
| :--- | :--- | :--- | :--- |
| **Full Model (OET)** | `bert-base-multilingual-cased` (frozen -- 0 trainable encoder layers; only Bi-LSTM + Attention + heads train, 3,093,003 params) | 119,547 (WordPiece) | **Applicable** (mBERT backbone) |
| **mBERT Sentence** | `bert-base-multilingual-cased` (re-run at matched 0-layer capacity, 2026-09-05 — see `docs/capacity_matched_comparison.md`) | 119,547 (WordPiece) | **Applicable** (direct baseline) |
| **XLM-R Sentence** | `xlm-roberta-base` (re-run at matched 0-layer capacity, 2026-09-05 — see `docs/capacity_matched_comparison.md`) | 250,002 (BPE) | **Applicable** (high-capacity benchmark) |
| **LSTM-Only** | `bert-base-multilingual-cased` | 119,547 (WordPiece) | **Applicable** (identical encoder) |
| **Attention-Only** | `bert-base-multilingual-cased` | 119,547 (WordPiece) | **Applicable** (identical encoder) |
| **TextCNN** | Randomly-initialised `nn.Embedding` (300d), learned from scratch -- no pretrained vectors | <=20,000 whitespace tokens (corpus-derived) | **N/A** (no masked language model) |

---

## Step B3: Sequential Opinion Modeling, Calibration & SCS

| Model | Sequential Encoder | Heads Supported | Sequence Consistency Score (SCS) | Calibration (ECE) & Entropy |
| :--- | :--- | :--- | :--- | :--- |
| **Full Model (OET)** | **Bi-LSTM + Self-Attention** | **Sentiment, Trend, Trajectory** | **Supported** | **Monitored across all 3 heads** |
| **mBERT Sentence** | None (Independent reviews) | Sentiment only | N/A (single sentence) | Sentiment only |
| **XLM-R Sentence** | None (Independent reviews) | Sentiment only | N/A (single sentence) | Sentiment only |
| **LSTM-Only** | Bi-LSTM (Final hidden state) | Sentiment, Trajectory | Supported | Sentiment and Trajectory |
| **Attention-Only** | Multi-Head Self-Attention | Sentiment, Trajectory | Supported | Sentiment and Trajectory |
| **TextCNN** | 1D CNN over words | Sentiment only | N/A (single sentence) | Sentiment only |

---

## Methodological Contributions
- **Sequence Consistency Score (SCS)** evaluates sequential temporal prediction reliability (`mean(SCS)` and `std(SCS)`). Only models with trajectory heads can compute it.
- **Expected Calibration Error (ECE)** monitors probability reliability, ensuring that prediction confidence aligns with empirical accuracy.
