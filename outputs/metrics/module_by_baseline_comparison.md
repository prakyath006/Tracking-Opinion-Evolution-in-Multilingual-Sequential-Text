# Module-by-Baseline Comparison Matrix

> **Deliverable for Part B (Steps B1 – B5)**: Comprehensive module-specific architectural and metric capability comparison between the proposed **OpinionEvolutionTracker** and all **5 baseline models**.

---

## Summary Overview of Models

| Model Identifier | Architecture Description | Scope / Level | Training Strategy |
| :--- | :--- | :--- | :--- |
| **Full Model (OET)** | mBERT + Fine-Tuned Adaptor + Bi-LSTM + Self-Attention + Multi-Task Heads | Sequence (Multi-Task) | End-to-end multi-task with inverse-frequency loss weighting |
| **mBERT Sentence** | Fine-Tuned bert-base-multilingual-cased + Linear Classifier | Single Sentence | Flattened review-level sentiment classification |
| **XLM-R Sentence** | Fine-Tuned xlm-roberta-base + Linear Classifier | Single Sentence | Flattened review-level sentiment classification |
| **LSTM-Only** | mBERT + Bi-LSTM (Final hidden state, no attention) | Sequence (Ablation) | Sequence sentiment + trajectory (omits trend head) |
| **Attention-Only** | mBERT + Multi-Head Attention (No Bi-LSTM recurrence) | Sequence (Ablation) | Sequence sentiment + trajectory (omits trend head) |
| **TextCNN** | Pretrained Embeddings + 1D Convolutions (3, 4, 5) + Max-Pool | Single Sentence | Non-transformer n-gram baseline |

---

## Comprehensive Module × Baseline Comparison Matrix

| Module Dimension | Full Model (OET) | mBERT Sentence | XLM-R Sentence | LSTM-Only (Ablation) | Attention-Only (Ablation) | TextCNN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Module 1: Ontology Awareness** | Fully Aware (3 Levels) | Partially Aware (Level 1 only) | Partially Aware (Level 1 only) | Partially Aware (Levels 1 & 3) | Partially Aware (Levels 1 & 3) | Partially Aware (Level 1 only) |
| — *SentimentState (Level 1)* | Yes (SentimentState: 4 classes) | Yes (SentimentState: 4 classes) | Yes (SentimentState: 4 classes) | Yes (SentimentState: 4 classes) | Yes (SentimentState: 4 classes) | Yes (SentimentState: 4 classes) |
| — *TransitionType (Level 2)* | Yes (TransitionType: 3 classes via pairwise head) | N/A — no sequence/transition head | N/A — no sequence/transition head | N/A — lacks pairwise trend head | N/A — lacks pairwise trend head | N/A — no sequence/transition head |
| — *TrajectoryType (Level 3)* | Yes (TrajectoryType: 4 classes via sequence head) | N/A — no trajectory head | N/A — no trajectory head | Yes (TrajectoryType: 4 classes) | Yes (TrajectoryType: 4 classes) | N/A — no trajectory head |
| **Module 2: Encoder Backbone** | bert-base-multilingual-cased (top 3 layers fine-tuned) | bert-base-multilingual-cased (frozen or full fine-tune) | xlm-roberta-base (SentencePiece 100-language model) | bert-base-multilingual-cased | bert-base-multilingual-cased | Static Pretrained Word Embeddings (GloVe / FastText 300d) |
| — *MLM Perplexity Applicability* | Applicable (mBERT backbone) | Applicable (identical encoder backbone) | Applicable (XLM-R backbone) | Applicable (mBERT backbone) | Applicable (mBERT backbone) | N/A — Non-transformer architecture |
| — *Vocabulary Size* | 119,547 tokens (WordPiece multilingual) | 119,547 tokens | 250,002 tokens (BPE multilingual) | 119,547 tokens | 119,547 tokens | Corpus-derived word vocabulary |
| **Module 3: Sequential Architecture** | Bidirectional LSTM (2 layers, hidden=256) + Self-Attention | None (independent review classification) | None (independent review classification) | Bidirectional LSTM (final hidden state pooled, no attention) | Multi-Head Self-Attention directly over embedding sequence (no LSTM) | 1D Convolution over word tokens (kernel sizes 3, 4, 5) |
| — *Sentiment Head* | Supported (per-review logits [batch, seq_len, 4]) | Supported (single-review logits [batch, 4]) | Supported (single-review logits [batch, 4]) | Supported (per-review sequence logits [batch, seq_len, 4]) | Supported (per-review sequence logits [batch, seq_len, 4]) | Supported (single-review logits [batch, 4]) |
| — *Transition (Trend) Head* | Supported (pairwise transition logits [batch, seq_len, 3]) | N/A — independent sentence predictions lack sequence context | N/A — independent sentence predictions lack sequence context | N/A — baseline omitted trend head to isolate trajectory ablation | N/A — baseline omitted trend head to isolate trajectory ablation | N/A — no sequence modeling |
| — *Trajectory Head* | Supported (sequence-level trajectory logits [batch, 4]) | N/A — cannot predict sequence trajectory | N/A — cannot predict sequence trajectory | Supported (sequence-level trajectory logits [batch, 4]) | Supported (sequence-level trajectory logits [batch, 4]) | N/A — no trajectory prediction |
| — *Sequence Consistency Score (SCS)* | Supported — calculates temporal prediction consistency across reviews | N/A — sentence-level model produces no sequence trajectories | N/A — sentence-level model produces no sequence trajectories | Supported — sequential predictions permit SCS computation | Supported — sequential predictions permit SCS computation | N/A — sentence-level only |
| — *Uncertainty & Entropy* | Supported (monitored per head: sentiment, trend, trajectory) | Supported for sentiment head only | Supported for sentiment head only | Supported for sentiment and trajectory heads | Supported for sentiment and trajectory heads | Supported for sentiment head only |
| — *Calibration (ECE)* | Supported — Expected Calibration Error binned over confidence | Supported for sentiment head only | Supported for sentiment head only | Supported for sentiment and trajectory heads | Supported for sentiment and trajectory heads | Supported for sentiment head only |
| **Module 4: Cross-Domain Generalization** | Both directions (Amazon -> Dravidian, Dravidian -> Amazon) | Both directions (sentiment task only) | Both directions (sentiment task only) | Both directions (sentiment + trajectory tasks) | Both directions (sentiment + trajectory tasks) | Both directions (sentiment task only) |
| — *Fuzzy Typicality Scoring* | Supported — sequence centroid distance to domain clusters | Supported at sentence level only | Supported at sentence level only | Supported at sequence level | Supported at sequence level | N/A — static embeddings do not align with contextualized centroid space |
| — *Resilience to Domain Shift* | Moderate — sequence dynamics and attention buffer against domain shifts | High — lexical shift between Amazon product reviews and Dravidian comments causes steep drop | High to Moderate — slightly better cross-lingual transfer, but still lacks sequential dynamics | Moderate to High — without attention, early/late review bias exacerbates domain shift | Moderate — attention identifies salient tokens, but unordered sequence representations drift | Severe — static embeddings suffer from high out-of-vocabulary rates across code-mixed domains |

---

## Detailed Analysis per Step

### Step B1: Structural Ontology Benefit
- **Full Model**: Integrates the ontology as a single source of truth across all 3 classification levels. Raw labels from both Amazon star ratings and Dravidian YouTube comments are mapped into `SentimentState` (0..3), while pairwise deltas map into `TransitionType` (0..2), and temporal trajectory counts map into `TrajectoryType` (0..3).
- **Group A Baselines (mBERT, XLM-R, TextCNN)**: Benefit from the ontology at Level 1 (`SentimentState`), allowing unified sentiment cross-domain evaluation. However, because they lack sequential context, they cannot utilize Levels 2 or 3.
- **Group B Baselines (LSTM-Only, Attention-Only)**: Benefit from Level 1 and Level 3, proving the ability to classify trajectory patterns while omitting Level 2 (pairwise transitions).

### Step B2: Encoder Representations & MLM Perplexity
- **mBERT vs. XLM-R**: Evaluated via `src/mlm_perplexity_eval.py`. XLM-R has a larger vocabulary (250K vs 119K) and better raw script coverage, but requires higher memory and slower throughput. mBERT with top-3 layer domain adaptation strikes the optimal balance between inference latency and subword representation.
- **TextCNN**: Marked as **N/A** because it does not utilize a transformer language model; static embeddings cannot undergo masked language modeling.

### Step B3: Sequential Opinion Modeling & Novel Metrics
- **Sequence Consistency Score (SCS)**: Only applicable to sequence-aware models (`Full Model`, `LSTM-Only`, `Attention-Only`). Sentence-level models cannot output sequence trajectories or temporal consistency metrics.
- **Ablation Insight**: The contrast between `LSTM-Only` and `Attention-Only` isolates the individual contributions: Bi-LSTM captures chronological order dependencies, while Self-Attention localizes critical turning-point reviews.

### Step B4: Cross-Domain Transfer & Typicality
- **Fuzzy Domain Typicality**: Computed via `src/fuzzy_domain_score.py`. The Full Model's multi-task representations create coherent domain centroids in sequence embedding space, allowing fuzzy membership scoring for out-of-domain sequences.
- **Degradation Pattern**: Sentence models experience severe F1 degradation under domain transfer due to lexical shifts. The Full Model's sequential and attention layers mitigate this by focusing on structural opinion evolution patterns rather than domain-specific vocabulary.

---
*Report automatically generated by `scripts/generate_module_by_baseline_matrix.py`.*