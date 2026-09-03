# Step B1 — Ontology Module Capability Comparison

## Objective
Assess how each comparison baseline interacts with the unified structural ontology (`src/ontology.py`), determining whether each baseline benefits from closed-vocabulary unified labeling or is constrained to raw dataset-specific labels.

---

## Baseline-by-Baseline Structural Analysis

| Model Identifier | Ontology-Awareness Level | SentimentState (L1) | TransitionType (L2) | TrajectoryType (L3) | Cross-Domain Compatibility |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Model (OET)** | **Fully Aware (All 3 Levels)** | **Yes (4 classes)** | **Yes (3 classes)** | **Yes (4 classes)** | **Unified cross-domain loss** |
| **mBERT Sentence** | Partially Aware (Level 1 only) | Yes (4 classes) | N/A (no sequence) | N/A (no trajectory) | Sentiment only |
| **XLM-R Sentence** | Partially Aware (Level 1 only) | Yes (4 classes) | N/A (no sequence) | N/A (no trajectory) | Sentiment only |
| **LSTM-Only** | Partially Aware (Levels 1 & 3) | Yes (4 classes) | N/A (no trend head) | Yes (4 classes) | Sentiment + Trajectory |
| **Attention-Only** | Partially Aware (Levels 1 & 3) | Yes (4 classes) | N/A (no trend head) | Yes (4 classes) | Sentiment + Trajectory |
| **TextCNN** | Partially Aware (Level 1 only) | Yes (4 classes) | N/A (no sequence) | N/A (no trajectory) | Sentiment only |

---

## Key Insights
1. **Single Source of Truth**: The Full Model is the only model whose architecture mirrors the 3-level taxonomy hierarchy of the structural ontology.
2. **Decoupled Ablation**: Group B ablation baselines (`LSTM-Only` and `Attention-Only`) demonstrate that sequence trajectory classification is possible without the intermediate pairwise transition head, isolating the specific benefit of Module 1's transition level.
3. **Closed Vocabulary Guarantee**: All transformer models benefit from `SentimentState` level-1 normalization, preventing cross-dataset vocabulary mismatch between Amazon ratings and Dravidian sentiment tags.
