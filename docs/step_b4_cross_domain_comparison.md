# Step B4 — Cross-Domain Transfer & Fuzzy Typicality Comparison

## Overview
Cross-domain generalization tests the model's ability to maintain opinion tracking performance when transferred across disjoint domains:
- **Domain 1**: Amazon E-Commerce Beauty Reviews (Structured ratings, formal reviews)
- **Domain 2**: Dravidian YouTube Social Media Comments (Tamil, Malayalam, Kannada with heavy code-mixing and informal colloquialisms)

---

## Cross-Domain Transfer Profiles

| Model | Evaluated Directions | Transfer Representation | Fuzzy Typicality Score | Domain Shift Resilience |
| :--- | :--- | :--- | :--- | :--- |
| **Full Model (OET)** | **Both (Amazon ↔ Dravidian)** | **Unified Ontology + Bi-LSTM Sequence States** | **Supported (Sequence Centroids)** | **High (Attention buffers lexical drift)** |
| **mBERT Sentence** | Both directions (Sentiment only) | Token embeddings | Supported (Sentence level) | Low (Steep drop due to vocabulary mismatch) |
| **XLM-R Sentence** | Both directions (Sentiment only) | Multilingual token embeddings | Supported (Sentence level) | Moderate (Higher vocabulary reduces unknown tokens) |
| **LSTM-Only** | Both directions (Sentiment + Trajectory) | Unweighted recurrent hidden states | Supported (Sequence level) | Moderate (Lacks attention to highlight invariant tokens) |
| **Attention-Only** | Both directions (Sentiment + Trajectory) | Multi-head self-attention pooling | Supported (Sequence level) | Moderate (Highlights invariant tokens, but loses temporal order) |
| **TextCNN** | Both directions (Sentiment only) | Static word vectors | N/A (Static space) | Poor (Severe OOV rates in code-mixed text) |

---

## Fuzzy Typicality Scoring (`src/fuzzy_domain_score.py`)
- For each test sequence $s$, fuzzy membership vector $\mu(s) = [\mu_{\text{Amazon}}, \mu_{\text{Tamil}}, \mu_{\text{Malayalam}}, \mu_{\text{Kannada}}]$ is computed via cosine similarity to source domain centroids.
- Sequences with lower source typicality exhibit higher degradation, confirming that domain distance directly predicts transfer difficulty.
